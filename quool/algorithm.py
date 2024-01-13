import abc
import numpy as np
import pandas as pd
from .tool import DimFormatter
from .exception import NotRequiredDimError, UnfittedError


class Algorithm(abc.ABC):

    def __init__(self, **params) -> None:
        for key, value in params.items():
            setattr(self, key, value)
        self.fitted = False
    
    @abc.abstractmethod
    def fit(self, *args, **kwargs):
        raise NotImplementedError
    
    def transform(self, *args, **kwargs):
        raise NotImplementedError
    
    def fit_transform(self, *args, **kwargs):
        raise NotImplementedError


class Return(Algorithm):
    """
    A class to calculate the returns of financial instruments based on provided pricing data.

    Parameters:
    - price (pd.DataFrame | pd.Series): The pricing data, either as a DataFrame or a Series.
    - buy (str): The column name in the DataFrame to be used for the buy price.
    - sell (str): The column name in the DataFrame to be used for the sell price.
    - code_level (str | int): The level in the MultiIndex DataFrame or Series that represents the unique identifier for each financial instrument.
    - date_level (str | int): The level in the MultiIndex DataFrame or Series that represents the date.
    - delay (int): The delay in days for the transaction (e.g., a delay of 1 means the sell price is 1 day after the buy price).

    This class supports handling both regular indexed and MultiIndexed pandas data structures.
    """

    def __init__(
        self,
        buy: str = "open",
        sell: str = "close",
        delay: int = 1,
        code_level: str | int = 0,
        date_level: str | int = 1,
    ):
        """
        Initializes the Return object with price data and configuration.

        The initialization varies depending on the type of index (regular or MultiIndex) and the data structure (DataFrame or Series).
        - For a regular indexed DataFrame, sets the column and index names if not provided, and shifts the price data by the delay.
        - For a MultiIndexed DataFrame, groups by the code level and shifts the price data by the delay, and sets the buy and sell prices based on the specified columns.
        - For a MultiIndexed Series, groups by the code level and shifts the price data by the delay.
        """
        self.buy = buy
        self.sell = sell
        self.delay = delay
        self.code_level = code_level
        self.date_level = date_level
        self.fitted = False
    
    def fit(self, price: pd.DataFrame | pd.Series):
        formatter = DimFormatter(price)
        if formatter.ndims > 3:
            raise NotRequiredDimError(3)
        
        if formatter.naxes > 1 and price.shape[1] == 1 and formatter.rowdim > 1:
            # allow single-column dataframe with MultiIndex to be treated as series
            price = price.iloc[:, 0]
            
        if formatter.naxes > 1 and formatter.rowdim == 1:
            # DataFrame with Index
            self.code_level = price.columns.name or self.code_level
            self.date_level = price.index.name or self.date_level
            price_shift = price.shift(-self.delay)
            self.buy_price = price_shift
            self.sell_price = price
        
        elif formatter.naxes > 1 and formatter.rowdim > 1:
            # DataFrame with MultiIndex
            price_shift = price.groupby(level=self.code_level).shift(-self.delay)
            self.buy_price = price_shift[self.buy]
            self.sell_price = price[self.sell]
        
        elif formatter.naxes == 1 and formatter.rowdim > 1:
            # Series with MultiIndex
            price_shift = price.groupby(level=self.code_level).shift(-self.delay)
            self.buy_price = price_shift
            self.sell_price = price
        
        elif formatter.naxes == 1 and formatter.rowdim == 1:
            # Series with Index
            price_shift = price.shift(-self.delay)
            self.buy_price = price_shift
            self.sell_price = price
        
        self.fitted = True
        return self
    
    def transform(self, span: int) -> pd.Series | pd.DataFrame:
        """
        Calculates the return over a specified span.

        Parameters:
        - span (int): The number of periods over which to calculate the return. Negative value calculates backward.
        - log (bool): If True, calculates the logarithmic return; otherwise, calculates the simple return.

        Returns:
        - pd.Series | pd.DataFrame: The calculated return, in the same data structure format as the input price data.

        This method handles both regular and MultiIndexed data, calculating returns based on the specified span and whether to use logarithmic or simple returns.
        """
        if not self.fitted:
            raise UnfittedError("Return")
        
        formatter = DimFormatter(self.buy_price)
        if formatter.naxes > 1:
            if span < 0:
                sell_price = self.sell_price.shift(span)
                buy_price = self.buy_price
            else:
                sell_price = self.sell_price
                buy_price = self.buy_price.shift(span)
        
        else:
            if span < 0:
                sell_price = self.sell_price.groupby(level=self.code_level).shift(span)
                buy_price = self.buy_price
            else:
                sell_price = self.sell_price
                buy_price = self.buy_price.groupby(level=self.code_level).shift(span)
        
        return sell_price / buy_price - 1
    
    def transform_log(self, span: int) -> pd.Series | pd.DataFrame:
        return np.log(self.transform(span=span) + 1)
    
    def fit_transform(
        self, 
        price: pd.DataFrame | pd.Series, 
        span: int,
        log: bool = False
    ) -> pd.Series | pd.DataFrame:
        self.fit(price)
        if log:
            return self.transform_log(span=span)
        return self.transform(span=span)


class Event(Return):
    """
    A subclass of Return that focuses on analyzing financial data around specific events.

    This class is designed to work with Series data with a MultiIndex, where one level represents the identifier of the financial instrument and the other represents the date.

    Parameters:
    - price (pd.Series): The pricing data as a pandas Series with a MultiIndex.
    - buy (str): The column name to be used for the buy price.
    - sell (str): The column name to be used for the sell price.
    - code_level (str | int): The level in the MultiIndex that represents the unique identifier for each financial instrument.
    - date_level (str | int): The level in the MultiIndex that represents the date.
    """
    
    def __init__(
        self,
        buy: str = "close",
        sell: str = "close",
        delay: int = 0,
        code_level: str | int = 0,
        date_level: str | int = 1,
    ):
        """
        Initializes the Event object with price data and configuration.

        The constructor ensures that the price data is a pandas Series with a MultiIndex. It then initializes the superclass with the provided data and configuration.
        """
        super().__init__(buy, sell, delay, code_level, date_level)

    def fit(self, price: pd.Series | pd.DataFrame, event: pd.Series | pd.DataFrame):
        """
        Analyzes the price data around the given events.

        Parameters:
        - event (pd.Series): The event data as a pandas Series with the same index type as the price data.
        - span (tuple): A tuple representing the range and step for the analysis (start, end, step).

        Returns:
        - pd.DataFrame: A DataFrame containing the return data for each day in the span around each event.

        This method calculates returns for each day within the specified span around the events and aligns them with the event dates.
        """
        event_formatter = DimFormatter(event)
        if event_formatter.ndims != 2:
            raise NotRequiredDimError(2)
        if event_formatter.rowdim == 1:
            event = event_formatter.swapdim(-1, 0)
        self.event = event

        super().fit(price)
        buy_formatter, sell_formatter = DimFormatter(self.buy_price), DimFormatter(self.sell_price)
        if buy_formatter.rowdim == 1:
            self.buy_price = buy_formatter.swapdim(-1, 0).data
        if sell_formatter.rowdim == 1:
            self.sell_price = sell_formatter.swapdim(-1, 0).data
            
        return self
        
    def transform(self, span: tuple) -> pd.Series:
        """
        Provides a convenient way to call the 'call' method.

        Parameters:
        - event (pd.Series): The event data as a pandas Series.
        - span (tuple): A tuple representing the range and step for the analysis.

        Returns:
        - tuple[pd.Series, pd.Series]: A tuple containing two pandas Series. The first Series is the mean of returns for each day in the span, and the second Series is the mean of cumulative returns.

        This method is a wrapper around the 'call' method that additionally calculates the mean and cumulative mean returns for the specified span.
        """
        if not self.fitted:
            raise UnfittedError("Event")

        res = []
        r = super().transform(1)
        for i in np.arange(*span):
            res.append(r.groupby(level=self.code_level).shift(-i).loc[self.event.index])
        res = pd.concat(res, axis=1, keys=np.arange(*span)).add_prefix('day').fillna(0)
        cumres = (1 + res).cumprod(axis=1)
        cumres = cumres.div(cumres["day0"], axis=0).mean(axis=0)
        return cumres
    
    def fit_transform(
        self, 
        price: pd.DataFrame | pd.Series, 
        event: pd.DataFrame | pd.Series, 
        span: tuple = (-5, 6, 1)
    ) -> pd.Series | pd.DataFrame:
        return self.fit(price, event).transform(span)


class PeriodEvent(Return):
    """
    A subclass of Return that focuses on analyzing financial data over specific periods associated with events.

    This class is designed to work with Series data with a MultiIndex, where one level represents the identifier of the financial instrument and the other represents the date.

    Parameters:
    - price (pd.Series): The pricing data as a pandas Series with a MultiIndex.
    - buy (str): The column name to be used for the buy price.
    - sell (str): The column name to be used for the sell price.
    - code_level (str | int): The level in the MultiIndex that represents the unique identifier for each financial instrument.
    - date_level (str | int): The level in the MultiIndex that represents the date.
    - delay (int): The delay in days for the transaction.
    """
    
    def __init__(
        self,
        buy: str = "close",
        sell: str = "close",
        delay: int = 1,
        code_level: str | int = 0,
        date_level: str | int = 1,
    ):
        """
        Initializes the PeriodEvent object with price data and configuration.

        The constructor ensures that the price data is a pandas Series with a MultiIndex. It then initializes the superclass with the provided data and configuration.
        """
        super().__init__(buy, sell, delay, code_level, date_level)
    
    def _compute(
        self, _event: pd.Series, 
        start: int | float | str, 
        stop: int | float | str
    ) -> pd.Series:
        """
        Calculates the returns for a specified period defined by start and stop events.

        Parameters:
        - _event (pd.Series): A pandas Series containing the event data.
        - start (int | float | str): The value in the event series that indicates the start of the period.
        - stop (int | float | str): The value in the event series that indicates the end of the period.

        Returns:
        - pd.Series: A Series containing the calculated returns for each period.

        This method identifies the start and stop of each event period, then calculates the returns for these periods.
        """
        _event_start = _event[_event == start].index
        if _event_start.empty:
            return pd.Series(index=pd.DatetimeIndex([pd.NaT], name=self.date_level))
        _event_start = _event_start.get_level_values(self.date_level)[0]
        _event = _event.loc[_event.index.get_level_values(self.date_level) >= _event_start]

        _event_diff = _event.diff()
        _event_diff.iloc[0] = _event.iloc[0]
        _event = _event[_event_diff != 0]
        
        buy_price = self.buy_price.loc[_event.index].loc[_event == start]
        sell_price = self.sell_price.loc[_event.index].loc[_event == stop]

        if buy_price.shape[0] - sell_price.shape[0] > 1:
            raise ValueError("there are unmatched start-stop labels")
        buy_price = buy_price.iloc[:sell_price.shape[0]]
        idx = pd.IntervalIndex.from_arrays(
            left = buy_price.index.get_level_values(self.date_level),
            right = sell_price.index.get_level_values(self.date_level),
            name=self.date_level
        )
        buy_price.index = idx
        sell_price.index = idx

        if buy_price.empty:
            return pd.Series(index=pd.DatetimeIndex([pd.NaT], name=self.date_level))
        
        return sell_price / buy_price - 1
        
    def fit(self, price: pd.Series | pd.DataFrame, event: pd.Series | pd.DataFrame):
        event_formatter = DimFormatter(event)
        if event_formatter.ndims != 2:
            raise NotRequiredDimError(2)
        if event_formatter.rowdim == 1:
            event = event_formatter.swapdim(-1, 0)
        self.event = event

        super().fit(price)
        buy_formatter, sell_formatter = DimFormatter(self.buy_price), DimFormatter(self.sell_price)
        if buy_formatter.rowdim == 1:
            self.buy_price = buy_formatter.swapdim(-1, 0).data
        if sell_formatter.rowdim == 1:
            self.sell_price = sell_formatter.swapdim(-1, 0).data
            
        return self
        
    def transform(self, start: int | float | str, stop: int | float | str) -> pd.Series:
        """
        A wrapper around the __call method for applying the return calculation over multiple periods.

        Parameters:
        - event (pd.Series): A pandas Series containing the event data.
        - start (int | float | str): The value in the event series that indicates the start of the period.
        - stop (int | float | str): The value in the event series that indicates the end of the period.

        Returns:
        - pd.Series: A Series containing the calculated returns for each period.

        This method processes multiple periods in the event series and applies the return calculation to each.
        """
        if not self.fitted:
            raise UnfittedError("PeriodEvent")
        if set(self.event.unique()) - set([start, stop]):
            raise ValueError("there are labels that are not start-stop labels")
        if isinstance(start, str) or isinstance(stop, str):
            raise ValueError("start and stop should be int")

        res = self.event.groupby(level=self.code_level).apply(self._compute, start=start, stop=stop)
        return res
    
    def fit_transform(
        self,
        price: pd.Series,
        event: pd.Series,
        start: int | float | str,
        stop: int | float | str,
    ):
        return self.fit(price, event).transform(start, stop)


class Weight(Return):
    """
    A subclass of Return that calculates weighted returns of financial instruments based on given weights.

    This class is designed to work with both DataFrame and Series data structures for price and weight data.

    Parameters:
    - price (pd.DataFrame | pd.Series): The pricing data.
    - buy (str): The column name to be used for the buy price.
    - sell (str): The column name to be used for the sell price.
    - code_level (str | int): The level in the MultiIndex that represents the unique identifier for each financial instrument.
    - date_level (str | int): The level in the MultiIndex that represents the date.
    - delay (int): The delay in days for the transaction.
    """

    def __init__(
        self,
        buy: str = "open",
        sell: str = "close",
        delay: int = 1,
        code_level: str | int = 0,
        date_level: str | int = 1,
    ):
        """
        Initializes the Weight object with price data and configuration.

        The constructor calls the superclass initialization with the provided data and configuration.
        """
        super().__init__(buy, sell, delay, code_level, date_level)
    
    def fit(self, price: pd.DataFrame | pd.Series, weight: pd.DataFrame | pd.Series):
        """
        Calculates the weighted returns based on the provided weights.

        Parameters:
        - weight (pd.DataFrame | pd.Series): The weight data, either as a DataFrame or a Series.
        - rebalance (int): The number of periods after which the portfolio is rebalanced.

        Returns:
        - pd.Series: A Series containing the weighted returns.

        This method calculates returns based on the weight of each instrument at each time point, taking into account rebalancing frequency.
        """
        weight_formatter = DimFormatter(weight)
        if weight_formatter.ndims != 2:
            raise NotRequiredDimError(2)
        if weight_formatter.naxes == 1:
            weight = weight_formatter.swapdim(self.code_level, -1)
        self.weight = weight
        
        super().fit(price)
        buy_formatter = DimFormatter(self.buy_price)
        sell_formatter = DimFormatter(self.sell_price)
        if buy_formatter.rowdim == 2:
            self.buy_price = buy_formatter.swapdim(self.code_level, -1).data
        if sell_formatter.rowdim == 2:
            self.sell_price = sell_formatter.swapdim(self.code_level, -1).data
        
        return self
        
    def transform(
        self, 
        rebalance: int = -1,
        commission: float = 0.005,
        side: str = 'both',
        return_tvr: bool = False,
    ) -> tuple[pd.Series, pd.Series] | pd.Series:
        """
        Calculates the weighted returns and optionally the turnover rate (TVR).

        Parameters:
        - weight (pd.DataFrame | pd.Series): The weight data.
        - rebalance (int): The number of periods after which the portfolio is rebalanced.
        - commission (float): The commission rate applied on each transaction.
        - side (str): Indicates the side of transactions to consider for TVR ('both', 'buy', or 'sell').
        - return_tvr (bool): If True, also returns the turnover rate.

        Returns:
        - tuple[pd.Series, pd.Series] | pd.Series: The weighted returns and, optionally, the turnover rate.

        This method provides an extended functionality to calculate returns with the option to include the cost of turnover and commission.
        """        
        if not self.fitted:
            raise UnfittedError("Weight")

        delta = self.weight.fillna(0) - self.weight.shift(abs(rebalance)).fillna(0)
        if side == 'both':
            tvr = (delta.abs() / 2).sum(axis=1) / abs(rebalance)
        elif side == 'buy':
            tvr = delta.where(delta > 0).abs().sum(axis=1) / abs(rebalance)
        elif side == 'sell':
            tvr = delta.where(delta < 0).abs().sum(axis=1) / abs(rebalance)
        else:
            raise ValueError("side must be in ['both', 'buy', 'sell']")
        commission *= tvr

        r = super().transform(rebalance)
        r = (((r * self.weight).sum(axis=1) - commission) 
              / abs(rebalance)).shift(-min(0, rebalance)).fillna(0)

        if return_tvr:
            return r, tvr
        return r

    def fit_transform(
        self,
        price: pd.DataFrame | pd.Series,
        weight: pd.DataFrame | pd.Series,
        rebalance: int = -1,
        commission: float = 0.005,
        side: str = 'both',
        return_tvr: bool = False,
    ):
        return self.fit(price, weight).transform(rebalance, commission, side, return_tvr)


class Rebalance(Return):
    """
    A subclass of Return that focuses on calculating returns for a rebalanced portfolio based on provided weights.

    This class is designed to handle both DataFrame and Series data structures for price and weight data, and it calculates returns considering portfolio rebalancing.

    Parameters:
    - price (pd.DataFrame | pd.Series): The pricing data.
    - buy (str): The column name to be used for the buy price.
    - sell (str): The column name to be used for the sell price.
    - code_level (str | int): The level in the MultiIndex that represents the unique identifier for each financial instrument.
    - date_level (str | int): The level in the MultiIndex that represents the date.
    """

    def __init__(
        self,
        buy: str = "close",
        sell: str = "close",
        code_level: str | int = 0,
        date_level: str | int = 1,
    ):
        """
        Initializes the Rebalance object with price data and configuration.

        The constructor calls the superclass initialization with the provided data and configuration.
        """
        super().__init__(buy, sell, 0, code_level, date_level)
    
    def fit(self, price: pd.DataFrame | pd.Series, weight: pd.DataFrame | pd.Series):
        """
        Private method to calculate returns based on the provided weights.

        Parameters:
        - weight (pd.DataFrame | pd.Series): The weight data, either as a DataFrame or a Series.

        Returns:
        - pd.Series: A Series containing the returns for each time period.

        This method calculates returns for each time period, taking into account the weights of each instrument in the portfolio.
        """
        weight_formatter = DimFormatter(weight)
        if weight_formatter.ndims != 2:
            raise NotRequiredDimError(2)
        if weight_formatter.naxes == 1:
            weight = weight_formatter.swapdim(self.code_level, -1)
        self.weight = weight
        
        super().fit(price)
        buy_formatter = DimFormatter(self.buy_price)
        sell_formatter = DimFormatter(self.sell_price)
        if buy_formatter.rowdim == 2:
            self.buy_price = buy_formatter.swapdim(self.code_level, -1).data
        if sell_formatter.rowdim == 2:
            self.sell_price = sell_formatter.swapdim(self.code_level, -1).data
        
        return self 
    
    def transform(
        self, 
        commission: float = 0.005,
        side: str = 'both',
        return_tvr: bool = False,
    ) -> tuple[pd.Series, pd.Series] | pd.Series:
        """
        Calculates the returns for a rebalanced portfolio and optionally the turnover rate (TVR).

        Parameters:
        - weight (pd.DataFrame | pd.Series): The weight data.
        - commission (float): The commission rate applied on each transaction.
        - side (str): Indicates the side of transactions to consider for TVR ('both', 'buy', or 'sell').
        - return_tvr (bool): If True, also returns the turnover rate.

        Returns:
        - tuple[pd.Series, pd.Series] | pd.Series: The returns for the rebalanced portfolio and, optionally, the turnover rate.

        This method provides functionality to calculate returns for a rebalanced portfolio, including the option to account for the cost of turnover and commission.
        """
        if not self.fitted:
            raise UnfittedError("Rebalance")
        
        delta = self.weight.fillna(0) - self.weight.shift(1).fillna(0)
        if side == 'both':
            tvr = (delta.abs() / 2).sum(axis=1)
        elif side == 'buy':
            tvr = delta.where(delta > 0).abs().sum(axis=1)
        elif side == 'sell':
            tvr = delta.where(delta < 0).abs().sum(axis=1)
        else:
            raise ValueError("side must be in ['both', 'buy', 'sell']")
        commission *= tvr
        
        r = super().transform(span=1)
        weight = self.weight.fillna(0).reindex(r.index).ffill()
        r *= weight
        r = r.sum(axis=1) - commission.reindex(r.index).fillna(0)
        
        if return_tvr:
            return r, tvr
        return r
    
    def fit_transform(
        self,
        price: pd.DataFrame | pd.Series,
        weight: pd.DataFrame | pd.Series,
        commission: float = 0.005,
        side: str = 'both',
        return_tvr: bool = False,
    ):
        self.fit(price, weight)
        return self.transform(commission, side, return_tvr)


class RobustScaler(Algorithm):
    """
    A subclass of Preprocessor that focuses on detecting and handling outliers in financial data.

    This class provides methods to identify and adjust outliers in the data using various statistical techniques.

    Inherits:
    - All attributes and methods from Preprocessor class.
    """

    def __init__(
        self,
        method: str = "mad", 
        n: int = 5, 
        code_level: str | int = 0,
        date_level: str | int = 1,
    ) -> None:
        super().__init__(method=method, n=n, 
            code_level=code_level, date_level=date_level)
    
    def fit(self, data: pd.DataFrame | pd.Series):
        formatter = DimFormatter(data)
        if formatter.ndims != 2:
            raise NotRequiredDimError(2)
        
        if formatter.rowdim == 2:
            data = formatter.swapdim(self.code_level, -1)
        
        if self.method == "mad":
            median = data.median(axis=1)
            ad = data.sub(median, axis=0)
            mad = ad.abs().median(axis=1)
            self.thresh_up = median + self.n * mad
            self.thresh_down = median - self.n * mad
        elif self.method == "std":
            mean = data.mean(axis=1)
            std = data.std(axis=1)
            self.thresh_up = mean + std * self.n
            self.thresh_down = mean - std * self.n
        elif self.method == "iqr":
            self.thresh_down = data.quantile(self.n, axis=1)
            self.thresh_up = data.quantile(1 - self.n, axis=1)
        else:
            raise ValueError("Invalid method.")
        
        self.data = data
        self.fitted = True
        return self

    def transform(self) -> pd.DataFrame:
        """
        Processes the data to handle outliers using the specified method.

        Parameters:
        - method (str): The method to use for handling outliers ('mad', 'std', or 'nan').
        - n (int or float): The parameter for the chosen method (threshold multiplier or quantile).

        Returns:
        - pd.DataFrame: The processed data with outliers handled.

        Raises:
        - ValueError: If the method is not one of the specified options ('mad', 'std', or 'nan').

        This method selects one of the outlier handling techniques and applies it to the data.
        """
        if not self.fitted:
            raise UnfittedError("RobutScaler")

        if self.method == 'mad' or self.method == "std":
            return self.data.clip(self.thresh_down, self.thresh_up, axis=0).where(~self.data.isna())
        elif self.method == 'iqr':
            return self.data.mask(
                self.data.ge(self.thresh_up, axis=0) & self.data.le(self.thresh_down, axis=0), 
            np.nan, axis=0).where(~self.data.isna())
        else:
            raise ValueError('method must be "mad", "std" or "iqr"')

    def fit_transform(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame:
        return self.fit(data).transform()


class StandardScaler(Algorithm):

    def __init__(
        self,
        method: str = "zscore", 
        code_level: str | int = 0,
        date_level: str | int = 1,
    ) -> None:
        super().__init__(method=method, 
            code_level=code_level, date_level=date_level)
    
    def fit(self, data: pd.DataFrame | pd.Series):
        formatter = DimFormatter(data)
        if formatter.ndims != 2:
            raise NotRequiredDimError(2)
        
        if formatter.rowdim == 2:
            data = formatter.swapdim(self.code_level, -1)
        
        if self.method == "zscore":
            self.mean = data.mean(axis=1)
            self.std = data.std(axis=1)
        elif self.method == "mad":
            self.max = data.max(axis=1)
            self.min = data.min(axis=1)
        else:
            raise ValueError("Invalid method.")

        self.data = data
        self.fitted = True
        return self

    def transform(self):
        if not self.fitted:
            raise UnfittedError("StandardScaler")

        if self.method == 'zscore':
            return self.data.sub(self.mean, axis=0).div(self.std, axis=0)
        elif self.method == 'minmax':
            return self.data.sub(self.min, axis=0).div((self.max - self.min), axis=0)
        else:
            raise ValueError('method must be "zscore" or "minmax"')

    def fit_transform(self, data: pd.DataFrame):
        return self.fit(data).transform()


class Imputer(Algorithm):

    def __init__(
        self,
        method: str = "mean", 
        code_level: str | int = 0,
        date_level: str | int = 1,
    ) -> None:
        super().__init__(method=method,
            code_level=code_level, date_level=date_level)
        
    def fit(self, data: pd.DataFrame):
        formatter = DimFormatter(data)
        if formatter.ndims != 2:
            raise NotRequiredDimError(2)
        
        if formatter.rowdim == 2:
            data = formatter.swapdim(self.code_level, -1)
        
        if self.method == "median":
            self.filler = data.median(axis=1)
        elif self.method == "mod":
            self.filler = data.mode(axis=1)
        elif self.method == "mean":
            self.filler = data.mean(axis=1)
        else:
            raise ValueError("Invalid method.")
        
        self.data = data
        self.fitted = True
        return self

    def transform(self):
        if not self.fitted:
            raise UnfittedError("Imputer")

        if self.method == 'mean':
            return self.data.T.fillna(self.filler).T
        elif self.method == 'median':
            return self.data.T.fillna(self.filler).T
        elif self.method == 'mod':
            return self.data.T.fillna(self.filler).T
        else:
            raise ValueError('method must be "zero", "mean", "median", "ffill", or "bfill"')
        
    def fit_transform(self, data: pd.DataFrame):
        return self.fit(data).transform()


class Corr(Algorithm):

    def __init__(
        self,
        method: str = 'pearson',
        code_level: str | int = 0, 
        date_level: str | int = 1,
    ) -> None:
        super().__init__(method=method,
            code_level=code_level, date_level=date_level)

    def fit(self, left: pd.DataFrame | pd.Series, right: pd.DataFrame | pd.Series):
        left_formatter = DimFormatter(left)
        right_formatter = DimFormatter(right)
        if left_formatter.ndims != 2 or right_formatter.ndims != 2:
            raise ValueError("left and right must be two-dimensional")
        
        if left_formatter.rowdim == 2:
            left = left_formatter.swapdim(self.code_level, -1)
        if right_formatter.rowdim == 2:
            right = right_formatter.swapdim(self.code_level, -1)
        
        self.left = left
        self.right = right
        self.fitted = True
        return self

    def transform(self):
        """
        Calculates the correlation between the two datasets.

        Parameters:
        - method (str): The method for correlation calculation (default is 'pearson').

        Returns:
        - pd.Series: A Series containing the correlation coefficients for each time period.

        This method computes the correlation for each time period using the specified method.
        """
        if not self.fitted:
            raise ValueError('the model is not fitted yet')
        return self.left.corrwith(self.right, axis=1, method=self.method)
    
    def fit_transform(self, left: pd.DataFrame | pd.Series, right: pd.DataFrame | pd.Series):
        return self.fit(left, right).transform()


class Layer(Algorithm):

    def __init__(
        self,
        ngroup: int = 10,
        code_level: int = 0,
        date_level: int = 1,
    ) -> None:
        super().__init__(ngroup=ngroup,
            code_level=code_level, date_level=date_level)
    
    def fit(self, data: pd.DataFrame | pd.Series):
        formatter = DimFormatter(data)
        if formatter.ndims != 2:
            raise ValueError('data must be a two-dimensional')
        
        if formatter.rowdim == 2:
            data = formatter.swapdim(self.code_level, -1)
        
        self.data = data
        self.fitted = True
        return self
    
    def transform(self):
        if not self.fitted:
            raise ValueError('the model is not fitted yet')
        
        return self.data.apply(lambda x: pd.qcut(x, self.ngroup, labels=False), axis=1) + 1
    
    def fit_transform(self, data: pd.DataFrame | pd.Series):
        return self.fit(data).transform()
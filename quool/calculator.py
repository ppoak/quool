import numpy as np
import pandas as pd


class Return:
    """
    A class to calculate the returns of financial instruments based on provided pricing data.

    Parameters:
    - price (pd.DataFrame | pd.Series): The pricing data, either as a DataFrame or a Series.
    - buy_column (str): The column name in the DataFrame to be used for the buy price.
    - sell_column (str): The column name in the DataFrame to be used for the sell price.
    - code_level (str | int): The level in the MultiIndex DataFrame or Series that represents the unique identifier for each financial instrument.
    - date_level (str | int): The level in the MultiIndex DataFrame or Series that represents the date.
    - delay (int): The delay in days for the transaction (e.g., a delay of 1 means the sell price is 1 day after the buy price).

    This class supports handling both regular indexed and MultiIndexed pandas data structures.
    """

    def __init__(
        self,
        price: pd.DataFrame | pd.Series,
        buy_column: str = "open",
        sell_column: str = "close",
        code_level: str | int = 0,
        date_level: str | int = 1,
        delay: int = 1,
    ):
        """
        Initializes the Return object with price data and configuration.

        The initialization varies depending on the type of index (regular or MultiIndex) and the data structure (DataFrame or Series).
        - For a regular indexed DataFrame, sets the column and index names if not provided, and shifts the price data by the delay.
        - For a MultiIndexed DataFrame, groups by the code level and shifts the price data by the delay, and sets the buy and sell prices based on the specified columns.
        - For a MultiIndexed Series, groups by the code level and shifts the price data by the delay.
        """
        if isinstance(price, pd.DataFrame) and not \
            isinstance(price.index, pd.MultiIndex):
            self.date_level = price.index.name
            self.code_level = price.columns.name
            if self.code_level is None:
                self.code_level = code_level
                price.columns.name = self.code_level
            if self.date_level is None:
                self.date_level = date_level
                price.index.name = self.date_level
            self.price = price.shift(-delay)
            self.buy_price = self.price
            self.sell_price = self.price
        
        elif isinstance(price, pd.DataFrame) and \
            isinstance(price.index, pd.MultiIndex):
            self.date_level = date_level
            self.code_level = code_level
            self.price = price.groupby(level=self.code_level).shift(-delay)
            self.buy_price = self.price[buy_column]
            self.sell_price = self.price[sell_column]
        
        elif isinstance(price, pd.Series) and \
            isinstance(price.index, pd.MultiIndex):
            self.date_level = date_level
            self.code_level = code_level
            self.price = price.groupby(level=self.code_level).shift(-delay)
            self.buy_price = self.price
            self.sell_price = self.price
    
    def __call__(self, span: int = -1, log: bool = False) -> pd.Series | pd.DataFrame:
        """
        Calculates the return over a specified span.

        Parameters:
        - span (int): The number of periods over which to calculate the return. Negative value calculates backward.
        - log (bool): If True, calculates the logarithmic return; otherwise, calculates the simple return.

        Returns:
        - pd.Series | pd.DataFrame: The calculated return, in the same data structure format as the input price data.

        This method handles both regular and MultiIndexed data, calculating returns based on the specified span and whether to use logarithmic or simple returns.
        """
        if not isinstance(self.price.index, pd.MultiIndex):
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
        
        if log:
            return np.log(sell_price / buy_price)
        return sell_price / buy_price - 1


class Event(Return):
    """
    A subclass of Return that focuses on analyzing financial data around specific events.

    This class is designed to work with Series data with a MultiIndex, where one level represents the identifier of the financial instrument and the other represents the date.

    Parameters:
    - price (pd.Series): The pricing data as a pandas Series with a MultiIndex.
    - buy_column (str): The column name to be used for the buy price.
    - sell_column (str): The column name to be used for the sell price.
    - code_level (str | int): The level in the MultiIndex that represents the unique identifier for each financial instrument.
    - date_level (str | int): The level in the MultiIndex that represents the date.
    """
    
    def __init__(
        self,
        price: pd.Series,
        buy_column: str = "close",
        sell_column: str = "close",
        code_level: str | int = 0,
        date_level: str | int = 1,
    ):
        """
        Initializes the Event object with price data and configuration.

        The constructor ensures that the price data is a pandas Series with a MultiIndex. It then initializes the superclass with the provided data and configuration.
        """
        if not isinstance(price.index, pd.MultiIndex):
            raise ValueError("price must be a Series or DataFrame with MultiIndex")
        super().__init__(price, buy_column, sell_column,
            code_level, date_level, 0)

    def call(
        self,
        event: pd.Series,
        span: tuple = (-5, 6, 1),
    ):
        """
        Analyzes the price data around the given events.

        Parameters:
        - event (pd.Series): The event data as a pandas Series with the same index type as the price data.
        - span (tuple): A tuple representing the range and step for the analysis (start, end, step).

        Returns:
        - pd.DataFrame: A DataFrame containing the return data for each day in the span around each event.

        This method calculates returns for each day within the specified span around the events and aligns them with the event dates.
        """
        if not isinstance(self.price.index, type(event.index)):
            raise ValueError("the type of price and event must be the same")
        
        res = []
        r = super()(span=1)
        for i in np.arange(*span):
            res.append(r.groupby(level=self.code_level).shift(-i).loc[event.index])
        res = pd.concat(res, axis=1, keys=np.arange(*span)).add_prefix('day').fillna(0)
        return res
        
    def __call__(
        self,
        event: pd.Series,
        span: tuple = (-5, 6, 1),
    ) -> tuple[pd.Series, pd.Series]:
        """
        Provides a convenient way to call the 'call' method.

        Parameters:
        - event (pd.Series): The event data as a pandas Series.
        - span (tuple): A tuple representing the range and step for the analysis.

        Returns:
        - tuple[pd.Series, pd.Series]: A tuple containing two pandas Series. The first Series is the mean of returns for each day in the span, and the second Series is the mean of cumulative returns.

        This method is a wrapper around the 'call' method that additionally calculates the mean and cumulative mean returns for the specified span.
        """
        res = self.call(event, span)
        dayres = res.mean(axis=0)
        cumres = (1 + res).cumprod(axis=1)
        cumres = cumres.div(cumres["day0"], axis=0).mean(axis=0)
        return dayres, cumres


class PeriodEvent(Return):
    """
    A subclass of Return that focuses on analyzing financial data over specific periods associated with events.

    This class is designed to work with Series data with a MultiIndex, where one level represents the identifier of the financial instrument and the other represents the date.

    Parameters:
    - price (pd.Series): The pricing data as a pandas Series with a MultiIndex.
    - buy_column (str): The column name to be used for the buy price.
    - sell_column (str): The column name to be used for the sell price.
    - code_level (str | int): The level in the MultiIndex that represents the unique identifier for each financial instrument.
    - date_level (str | int): The level in the MultiIndex that represents the date.
    - delay (int): The delay in days for the transaction.
    """
    
    def __init__(
        self,
        price: pd.Series,
        buy_column: str = "close",
        sell_column: str = "close",
        code_level: str | int = 0,
        date_level: str | int = 1,
        delay: int = 1,
    ):
        """
        Initializes the PeriodEvent object with price data and configuration.

        The constructor ensures that the price data is a pandas Series with a MultiIndex. It then initializes the superclass with the provided data and configuration.
        """
        if not isinstance(price.index, pd.MultiIndex):
            raise ValueError("price must be a Series or DataFrame with MultiIndex")
        super().__init__(price, buy_column, sell_column,
            code_level, date_level, delay)
    
    def __call(
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
        
    def __call__(
        self,
        event: pd.Series,
        start: int | float | str,
        stop: int | float | str,
    ) -> pd.Series:
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
        if not isinstance(self.price.index, type(event.index)):
            raise ValueError("the type of price and event must be the same")
        if set(event.unique()) - set([start, stop]):
            raise ValueError("there are labels that are not start-stop labels")

        res = event.groupby(level=self.code_level).apply(
            self.__call, start=start, stop=stop)
        return res


class Weight(Return):
    """
    A subclass of Return that calculates weighted returns of financial instruments based on given weights.

    This class is designed to work with both DataFrame and Series data structures for price and weight data.

    Parameters:
    - price (pd.DataFrame | pd.Series): The pricing data.
    - buy_column (str): The column name to be used for the buy price.
    - sell_column (str): The column name to be used for the sell price.
    - code_level (str | int): The level in the MultiIndex that represents the unique identifier for each financial instrument.
    - date_level (str | int): The level in the MultiIndex that represents the date.
    - delay (int): The delay in days for the transaction.
    """

    def __init__(
        self,
        price: pd.DataFrame | pd.Series,
        buy_column: str = "open",
        sell_column: str = "close",
        code_level: str | int = 0,
        date_level: str | int = 1,
        delay: int = 1,
    ):
        """
        Initializes the Weight object with price data and configuration.

        The constructor calls the superclass initialization with the provided data and configuration.
        """
        super().__init__(price, buy_column, sell_column,
            code_level, date_level, delay)
    
    def call(
        self,
        weight: pd.DataFrame | pd.Series, 
        rebalance: int = -1,
    ):
        """
        Calculates the weighted returns based on the provided weights.

        Parameters:
        - weight (pd.DataFrame | pd.Series): The weight data, either as a DataFrame or a Series.
        - rebalance (int): The number of periods after which the portfolio is rebalanced.

        Returns:
        - pd.Series: A Series containing the weighted returns.

        This method calculates returns based on the weight of each instrument at each time point, taking into account rebalancing frequency.
        """
        if isinstance(weight.index, type(self.price.index)):
            r = super().__call__(rebalance)
        elif isinstance(weight.index, pd.MultiIndex):
            r = super().__call__(rebalance).stack().reorder_levels(
                [self.code_level, self.date_level])
        elif isinstance(weight.index, pd.Index):
            r = super().__call__(rebalance).unstack(
                level=self.code_level)

        return r * weight
    
    def __call__(
        self, 
        weight: pd.DataFrame | pd.Series, 
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
        if isinstance(weight, pd.Series) and \
            isinstance(weight.index, pd.MultiIndex):
            weight = weight.unstack(level=self.code_level)
        elif not (isinstance(weight, pd.DataFrame) and
            not isinstance(weight.index, pd.MultiIndex)):
            raise ValueError("the type of weight must be one-level "
                             f"DataFrame or multi-level Series")
        
        delta = weight.fillna(0) - weight.shift(abs(rebalance)).fillna(0)
        if side == 'both':
            tvr = (delta.abs() / 2).sum(axis=1) / abs(rebalance)
        elif side == 'buy':
            tvr = delta.where(delta > 0).abs().sum(axis=1) / abs(rebalance)
        elif side == 'sell':
            tvr = delta.where(delta < 0).abs().sum(axis=1) / abs(rebalance)
        else:
            raise ValueError("side must be in ['both', 'buy', 'sell']")
        commission *= tvr

        r = self.call(weight, rebalance)
        r = ((r.sum(axis=1) - commission) / 
             abs(rebalance)).shift(-min(0, rebalance)).fillna(0)

        if return_tvr:
            return r, tvr
        return r


class Rebalance(Return):
    """
    A subclass of Return that focuses on calculating returns for a rebalanced portfolio based on provided weights.

    This class is designed to handle both DataFrame and Series data structures for price and weight data, and it calculates returns considering portfolio rebalancing.

    Parameters:
    - price (pd.DataFrame | pd.Series): The pricing data.
    - buy_column (str): The column name to be used for the buy price.
    - sell_column (str): The column name to be used for the sell price.
    - code_level (str | int): The level in the MultiIndex that represents the unique identifier for each financial instrument.
    - date_level (str | int): The level in the MultiIndex that represents the date.
    """

    def __init__(
        self,
        price: pd.DataFrame | pd.Series,
        buy_column: str = "close",
        sell_column: str = "close",
        code_level: str | int = 0,
        date_level: str | int = 1,
    ):
        """
        Initializes the Rebalance object with price data and configuration.

        The constructor calls the superclass initialization with the provided data and configuration.
        """
        super().__init__(price, buy_column, sell_column, 
            code_level, date_level, 0)
    
    def call(
        self,
        weight: pd.DataFrame | pd.Series,
    ):
        """
        Private method to calculate returns based on the provided weights.

        Parameters:
        - weight (pd.DataFrame | pd.Series): The weight data, either as a DataFrame or a Series.

        Returns:
        - pd.Series: A Series containing the returns for each time period.

        This method calculates returns for each time period, taking into account the weights of each instrument in the portfolio.
        """
        if isinstance(weight.index, type(self.price.index)):
            r = super().__call__(span=1)
        elif isinstance(weight.index, pd.MultiIndex):
            r = super().__call__(span=1).stack().reorder_levels(
                [self.code_level, self.date_level])
        elif isinstance(weight.index, pd.Index):
            r = super().__call__(span=1).unstack(
                level=self.code_level)
        
        weight = weight.fillna(0).reindex(r.index)
        if isinstance(weight.index, pd.MultiIndex):
            weight = weight.groupby(self.code_level).ffill()
        
        r = r.fillna(0) * weight
        return r
    
    def __call__(
        self, 
        weight: pd.DataFrame | pd.Series, 
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
        if isinstance(weight, pd.Series) and \
            isinstance(weight.index, pd.MultiIndex):
            weight = weight.unstack(level=self.code_level)
        elif not (isinstance(weight, pd.DataFrame) and 
            not isinstance(weight.index, pd.MultiIndex)):
            raise ValueError("the type of weight must be one-level "
                             f"DataFrame or multi-level Series")
        
        delta = weight.fillna(0) - weight.shift(1).fillna(0)
        if side == 'both':
            tvr = (delta.abs() / 2).sum(axis=1)
        elif side == 'buy':
            tvr = delta.where(delta > 0).abs().sum(axis=1)
        elif side == 'sell':
            tvr = delta.where(delta < 0).abs().sum(axis=1)
        else:
            raise ValueError("side must be in ['both', 'buy', 'sell']")
        commission *= tvr

        r = self.call(weight)
        r = r.sum(axis=1) - commission.reindex(r.index).fillna(0)
        
        if return_tvr:
            return r, tvr
        return r


class Preprocessor:
    """
    A utility class for preprocessing financial data stored in pandas DataFrame or Series.

    This class handles the conversion between DataFrame and Series formats, particularly useful when the data needs to be in a specific format for certain analyses.

    Parameters:
    - data (pd.DataFrame | pd.Series): The financial data to be preprocessed.
    - code_level (str | int): The level in the MultiIndex that represents the unique identifier for each financial instrument.
    - date_level (str | int): The level in the MultiIndex that represents the date.
    """

    def __init__(
        self, 
        data: pd.DataFrame | pd.Series,
        code_level: str | int = 0,
        date_level: str | int = 1,
    ) -> None:
        """
        Initializes the Preprocessor object with data and configuration.

        The constructor checks the type of data and initializes the preprocessing structure accordingly. It also keeps track of whether the data was initially stacked (Series) or not (DataFrame).

        Raises:
        - ValueError: If the input data is neither a pandas DataFrame nor a Series.
        """
        self.code_level = code_level
        self.date_level = date_level
        self.data = self.__transform(data)

    def __transform(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame:
        """
        Private method to transform the input data into the desired format for preprocessing.

        This method handles the conversion between DataFrame and Series, based on the structure of the input data.

        Parameters:
        - data (pd.DataFrame | pd.Series): The data to be transformed.

        Raises:
        - ValueError: If the input data is neither a pandas DataFrame nor a Series.
        """
        if isinstance(data, pd.DataFrame) and isinstance(data, pd.Index):
            data = data
            self.stacked = False
        elif isinstance(data, pd.Series) and isinstance(data, pd.MultiIndex):
            data = data.unstck(level=self.code_level)
            self.stacked = True
        else:
            raise ValueError("data must be pd.DataFrame or pd.Series")
        return data
    
    def __recover(self, data: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
        """
        Recovers the original format of the data after preprocessing.

        Parameters:
        - data (pd.DataFrame | pd.Series): The data to be recovered to its original format.

        Returns:
        - pd.DataFrame | pd.Series: The data in its original format (either DataFrame or Series).

        This method reverses the preprocessing steps applied to the data, returning it to its initial structure.
        """
        if self.stacked and isinstance(data, pd.DataFrame) and isinstance(data, pd.Index):
            return data.stack().reorder_levels([self.code_level, self.date_level]).sort_index()
        elif not self.stacked and isinstance(data, pd.Series) and isinstance(data, pd.MultiIndex):
            return data.unstack(level=self.code_level)


class DeOutlier(Preprocessor):
    """
    A subclass of Preprocessor that focuses on detecting and handling outliers in financial data.

    This class provides methods to identify and adjust outliers in the data using various statistical techniques.

    Inherits:
    - All attributes and methods from Preprocessor class.
    """

    def _mad(self, data: pd.DataFrame, n: int = 5):
        """
        Detects and adjusts outliers based on Median Absolute Deviation (MAD).

        Parameters:
        - data (pd.DataFrame): The data in which to detect outliers.
        - n (int): The threshold multiplier for defining what constitutes an outlier.

        Returns:
        - pd.DataFrame: The adjusted data with outliers handled according to MAD.

        This method calculates the MAD and adjusts the values that are n times the MAD away from the median.
        """
        median = data.median(axis=1)
        ad = data.sub(median, axis=0)
        mad = ad.abs().median(axis=1)
        thresh = mad * n
        data = data.mask(
            ad.ge(thresh, axis=0), median + thresh, axis=0
        ).mask(data.isna(), np.nan)
        data = data.mask(
            ad.le(-thresh, axis=0), median - thresh, axis=0
        ).mask(data.isna(), np.nan)
        return data
    
    def _std(self, data: pd.DataFrame, n: int = 3):
        """
        Detects and adjusts outliers based on standard deviation.

        Parameters:
        - data (pd.DataFrame): The data in which to detect outliers.
        - n (int): The number of standard deviations to use as the threshold for defining outliers.

        Returns:
        - pd.DataFrame: The adjusted data with outliers handled according to standard deviation.

        This method calculates the standard deviation and adjusts the values that are n standard deviations away from the mean.
        """
        mean = data.mean(axis=1)
        std = data.std(axis=1)
        thresh = std * n
        threshup = mean + thresh
        threshdown = mean - thresh
        data = data.mask(
            data.ge(threshup, axis=0), threshup, axis=0
        ).mask(data.isna(), np.nan)
        data = data.mask(
            data.le(threshdown, axis=0), threshdown, axis=0
        ).mask(data.isna(), np.nan)
        return data

    def _nan(self, data: pd.DataFrame, n: int = 0.1):
        """
        Identifies outliers and replaces them with NaN based on quantiles.

        Parameters:
        - data (pd.DataFrame): The data in which to detect outliers.
        - n (float): The quantile to use for defining the upper and lower bounds of outliers.

        Returns:
        - pd.DataFrame: The data with outliers replaced by NaN.

        This method uses quantiles to determine outlier thresholds and replaces outliers with NaN.
        """
        up = data.quantile(1 - n, axis=1)
        down = data.quantile(n, axis=1)
        data = data.mask(
            data.ge(up, axis=0), np.nan, axis=0
        ).mask(data.isna(), np.nan)
        data = data.mask(
            data.le(down, axis=0), np.nan, axis=0
        ).mask(data.isna(), np.nan)
        return data
    
    def __call__(self, method: str = 'mad', n: int = 5):
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
        if method == 'mad':
            data = self._mad(self.data, n)
        elif method == 'std':
            data = self._std(self.data, n)
        elif method == 'nan':
            data = self._nan(self.data, n)
        else:
            raise ValueError('method must be "mad", "std" or "nan"')
        return self.__recover(data)


class Standarize(Preprocessor):
    """
    A subclass of Preprocessor that focuses on standardizing financial data.

    This class provides methods for standardizing data using Z-score normalization or Min-Max scaling.

    Inherits:
    - All attributes and methods from Preprocessor class.
    """

    def _zscore(self, data: pd.DataFrame):
        """
        Standardizes the data using Z-score normalization.

        Parameters:
        - data (pd.DataFrame): The data to be standardized.

        Returns:
        - pd.DataFrame: The standardized data using Z-score normalization.

        This method standardizes the data by subtracting the mean and dividing by the standard deviation for each time period.
        """
        mean = data.mean(axis=1)
        std = data.std(axis=1)
        return data.sub(mean, axis=0).div(std, axis=0)

    def _minmax(self, data: pd.DataFrame):
        """
        Standardizes the data using Min-Max scaling.

        Parameters:
        - data (pd.DataFrame): The data to be standardized.

        Returns:
        - pd.DataFrame: The standardized data using Min-Max scaling.

        This method standardizes the data by subtracting the minimum and dividing by the range (max - min) for each time period.
        """
        min = data.min(axis=1)
        max = data.max(axis=1)
        return data.sub(min, axis=0).div((max - min), axis=0)

    def __call__(self, method: str = 'zscore'):
        """
        Processes the data to standardize it using the specified method.

        Parameters:
        - method (str): The method to use for standardizing ('zscore' or 'minmax').

        Returns:
        - pd.DataFrame: The processed data, standardized according to the specified method.

        Raises:
        - ValueError: If the method is not one of the specified options ('zscore' or 'minmax').

        This method selects one of the standardization techniques (Z-score or Min-Max) and applies it to the data.
        """
        if method == 'zscore':
            data = self._zscore(self.data)
        elif method == 'minmax':
            data = self._minmax(self.data)
        else:
            raise ValueError('method must be "zscore" or "minmax"')
        return self.__recover(data)


class FillNA(Preprocessor):
    """
    A subclass of Preprocessor that focuses on handling missing values (NaNs) in financial data.

    This class offers various methods for imputing missing data, suitable for different analysis scenarios.

    Inherits:
    - All attributes and methods from Preprocessor class.
    """

    def _zero(self, data: pd.DataFrame):
        """
        Fills missing values with zero.

        Parameters:
        - data (pd.DataFrame): The data with missing values.

        Returns:
        - pd.DataFrame: The data with missing values filled with zero.
        """
        return data.fillna(0)
    
    def _mean(self, data: pd.DataFrame):
        """
        Fills missing values with the mean of each column.

        Parameters:
        - data (pd.DataFrame): The data with missing values.

        Returns:
        - pd.DataFrame: The data with missing values filled with the mean of each column.
        """
        return data.T.fillna(data.mean(axis=1)).T
    
    def _median(self, data: pd.DataFrame):
        """
        Fills missing values with the median of each column.

        Parameters:
        - data (pd.DataFrame): The data with missing values.

        Returns:
        - pd.DataFrame: The data with missing values filled with the median of each column.
        """
        return data.T.fillna(data.median(axis=1)).T
    
    def _ffill(self, data: pd.DataFrame):
        """
        Forward fills missing values.

        Parameters:
        - data (pd.DataFrame): The data with missing values.

        Returns:
        - pd.DataFrame: The data with missing values forward filled.
        """
        return data.ffill()
    
    def _bfill(self, data: pd.DataFrame):
        """
        Backward fills missing values.

        Parameters:
        - data (pd.DataFrame): The data with missing values.

        Returns:
        - pd.DataFrame: The data with missing values backward filled.
        """
        return data.bfill()
    
    def __call__(self, method: str = 'zero'):
        """
        Processes the data to handle missing values using the specified method.

        Parameters:
        - method (str): The method to use for handling missing values ('zero', 'mean', 'median', 'ffill', or 'bfill').

        Returns:
        - pd.DataFrame: The processed data with missing values handled according to the specified method.

        Raises:
        - ValueError: If the method is not one of the specified options.
        """
        if method == 'zero':
            data = self._zero(self.data)
        elif method == 'mean':
            data = self._mean(self.data)
        elif method == 'median':
            data = self._median(self.data)
        elif method == 'ffill':
            data = self._ffill(self.data)
        elif method == 'bfill':
            data = self._bfill(self.data)
        else:
            raise ValueError('method must be "zero", "mean", "median", "ffill", or "bfill"')
        return self.__recover(data)


class Corr(Preprocessor):
    """
    A subclass of Preprocessor that calculates the correlation between two datasets.

    This class is designed to compare two sets of financial data (for example, stock prices and returns) and compute their correlation.

    Inherits:
    - All attributes and methods from Preprocessor class.
    """

    def __init__(
        self, 
        data_left: pd.DataFrame | pd.Series,
        data_right: pd.DataFrame | pd.Series,
        code_level: str | int = 0, 
        date_level: str | int = 1,
    ) -> None:
        """
        Initializes the Corr object with two datasets and configuration for processing.

        Parameters:
        - data_left (pd.DataFrame | pd.Series): The first dataset for correlation calculation.
        - data_right (pd.DataFrame | pd.Series): The second dataset for correlation calculation.
        - code_level (str | int): The level in the MultiIndex that represents the unique identifier for each financial instrument.
        - date_level (str | int): The level in the MultiIndex that represents the date.

        The constructor transforms the input data using the __transform method inherited from Preprocessor.
        """
        self.code_level = code_level
        self.date_level = date_level

        self.data_right = self.__transform(data_right)
        self.data_left = self.__transform(data_left)

    def __call__(self, method: str = 'pearson'):
        """
        Calculates the correlation between the two datasets.

        Parameters:
        - method (str): The method for correlation calculation (default is 'pearson').

        Returns:
        - pd.Series: A Series containing the correlation coefficients for each time period.

        This method computes the correlation for each time period using the specified method.
        """
        data = self.data_left.corrwith(
            self.data_right, axis=1, method=method)
        return self.__recover(data)


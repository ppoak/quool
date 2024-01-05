import numpy as np
import pandas as pd


class Return:

    def __init__(
        self,
        price: pd.DataFrame | pd.Series,
        buy_column: str = "open",
        sell_column: str = "close",
        code_level: str = "code",
        date_level: str = "date",
        delay: int = 1,
    ):
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
    
    def ret(self, span: int = -1, log: bool = False) -> pd.Series | pd.DataFrame:
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
    
    def __init__(
        self,
        price: pd.Series,
        buy_column: str = "close",
        sell_column: str = "close",
        code_level: str = 'code',
        date_level: str = 'date',
    ):
        if not isinstance(price.index, pd.MultiIndex):
            raise ValueError("price must be a Series or DataFrame with MultiIndex")
        super().__init__(price, buy_column, sell_column,
            code_level, date_level, 0)

    def _ret(
        self,
        event: pd.Series,
        span: tuple = (-5, 6, 1),
    ):
        if not isinstance(self.price.index, type(event.index)):
            raise ValueError("the type of price and event must be the same")
        
        res = []
        r = super().ret(span=1)
        for i in np.arange(*span):
            res.append(r.groupby(level=self.code_level).shift(-i).loc[event.index])
        res = pd.concat(res, axis=1, keys=np.arange(*span)).add_prefix('day').fillna(0)
        return res
        
    def ret(
        self,
        event: pd.Series,
        span: tuple = (-5, 6, 1),
    ) -> tuple[pd.Series, pd.Series]:
        res = self._ret(event, span)
        dayres = res.mean(axis=0)
        cumres = (1 + res).cumprod(axis=1)
        cumres = cumres.div(cumres["day0"], axis=0).mean(axis=0)
        return dayres, cumres


class PeriodEvent(Return):
    
    def __init__(
        self,
        price: pd.Series,
        buy_column: str = "close",
        sell_column: str = "close",
        code_level: str = 'code',
        date_level: str = 'date',
        delay: int = 1,
    ):
        if not isinstance(price.index, pd.MultiIndex):
            raise ValueError("price must be a Series or DataFrame with MultiIndex")
        super().__init__(price, buy_column, sell_column,
            code_level, date_level, delay)
    
    def __ret(
        self, _event: pd.Series, 
        start: int | float | str, 
        stop: int | float | str
    ) -> pd.Series:
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
        
    def ret(
        self,
        event: pd.Series,
        start: int | float | str,
        stop: int | float | str,
    ) -> pd.Series:
        if not isinstance(self.price.index, type(event.index)):
            raise ValueError("the type of price and event must be the same")
        if set(event.unique()) - set([start, stop]):
            raise ValueError("there are labels that are not start-stop labels")

        res = event.groupby(level=self.code_level).apply(
            self.__ret, start=start, stop=stop)
        return res


class Weight(Return):

    def __init__(
        self,
        price: pd.DataFrame | pd.Series,
        buy_column: str = "open",
        sell_column: str = "close",
        code_level: str = 'code',
        date_level: str = 'date',
        delay: int = 1,
    ):
        super().__init__(price, buy_column, sell_column,
            code_level, date_level, delay)
    
    def _ret(
        self,
        weight: pd.DataFrame | pd.Series, 
        span: int = -1,
    ):
        if isinstance(weight.index, type(self.price.index)):
            r = super().ret(span)
        elif isinstance(weight.index, pd.MultiIndex):
            r = super().ret(span).stack().reorder_levels(
                [self.code_level, self.date_level])
        elif isinstance(weight.index, pd.Index):
            r = super().ret(span).unstack(
                level=self.code_level)

        return r * weight
    
    def ret(
        self, 
        weight: pd.DataFrame | pd.Series, 
        span: int = -1,
        commission: float = 0.005,
        side: str = 'both',
        return_tvr: bool = False,
    ) -> tuple[pd.Series, pd.Series] | pd.Series:
        if isinstance(weight, pd.Series) and \
            isinstance(weight.index, pd.MultiIndex):
            weight = weight.unstack(level=self.code_level)
        elif not (isinstance(weight, pd.DataFrame) and
            not isinstance(weight.index, pd.MultiIndex)):
            raise ValueError("the type of weight must be one-level "
                             f"DataFrame or multi-level Series")
        
        delta = weight.fillna(0) - weight.shift(abs(span)).fillna(0)
        if side == 'both':
            tvr = (delta.abs() / 2).sum(axis=1)
        elif side == 'buy':
            tvr = delta.where(delta > 0).abs().sum(axis=1)
        elif side == 'sell':
            tvr = delta.where(delta < 0).abs().sum(axis=1)
        else:
            raise ValueError("side must be in ['both', 'buy', 'sell']")
        commission *= tvr

        r = self._ret(weight, span)
        r = ((r.sum(axis=1) - commission) / 
             abs(span)).shift(-min(0, span)).fillna(0)

        if return_tvr:
            return r, tvr
        return r


class NetValue(Return):

    def __init__(
        self,
        price: pd.DataFrame | pd.Series,
        buy_column: str = "close",
        sell_column: str = "close",
        code_level: str = 'code',
        date_level: str = 'date',
    ):
        super().__init__(price, buy_column, sell_column, 
            code_level, date_level, 0)
    
    def _ret(
        self,
        weight: pd.DataFrame | pd.Series,
    ):
        if isinstance(weight.index, type(self.price.index)):
            r = super().ret(span=1)
        elif isinstance(weight.index, pd.MultiIndex):
            r = super().ret(span=1).stack().reorder_levels(
                [self.code_level, self.date_level])
        elif isinstance(weight.index, pd.Index):
            r = super().ret(span=1).unstack(
                level=self.code_level)
        
        weight = weight.fillna(0).reindex(r.index)
        if isinstance(weight.index, pd.MultiIndex):
            weight = weight.groupby(self.code_level).ffill()
        
        r = r.fillna(0) * weight
        return r
    
    def ret(
        self, 
        weight: pd.DataFrame | pd.Series, 
        commission: float = 0.005,
        side: str = 'both',
        return_tvr: bool = False,
    ) -> tuple[pd.Series, pd.Series] | pd.Series:
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

        r = self._ret(weight)
        r = r.sum(axis=1) - commission.reindex(r.index).fillna(0)
        
        if return_tvr:
            return r, tvr
        return r


import numpy as np
import pandas as pd
from .database import Data, Dim2Frame, Dim2Series
from .exception import NotRequiredDimError


class Return(Data):

    def __init__(
        self,
        data: pd.DataFrame | pd.Series,
        buyon: str = "open",
        sellon: str = "close",
        code_level: str | int = 0,
    ):
        self._code_level = code_level
        Data.__init__(self, data)
        if self.ndim > 3:
            raise NotRequiredDimError(3)
        
        if self.naxes > 1:
            self._pbuy, self._psell = data[buyon], data[sellon]
        elif self.naxes == 1:
            self._pbuy, self._psell = data, data
    
    def __call__(self, buyat: int, sellat: int, log: bool = False):
        if self._pbuy.index.nlevels == 1 and self._psell.index.nlevels == 1:
            r = self._psell.shift(-sellat) / self._pbuy.shift(-buyat)
        
        elif self._pbuy.index.nlevels == 2 and self._psell.index.nlevels == 2:
            r = (self._psell.groupby(level=self._code_level).shift(-sellat) / 
                self._pbuy.groupby(level=self._code_level).shift(-buyat))
        
        else:
            raise ValueError("internal data broken")
        
        return r - 1 if not log else np.log(r)


class Event(Return, Dim2Series):
    
    def __init__(
        self,
        data: pd.DataFrame | pd.Series,
        buyon: str = "close",
        sellon: str = "close",
        code_level: int | str = 0,
    ):
        Return.__init__(self, data, buyon, sellon, code_level)
        Dim2Series.__init__(self, data)

    def __call__(self, event: pd.Series, span: tuple = (-5, 6, 1), delay: int = 0):
        event = Dim2Series(event, self._code_level)._data
        res = []
        r = super(Return, self)(delay, delay + 1)
        for i in np.arange(*span):
            res.append(r.groupby(level=self._code_level).shift(-i).loc[event.index])
        res = pd.concat(res, axis=1, keys=np.arange(*span)).add_prefix('day').fillna(0)
        cumres = (1 + res).cumprod(axis=1)
        cumres = cumres.div(cumres["day0"], axis=0).mean(axis=0)
        return cumres


class PeriodEvent(Event, Dim2Series):
    
    def __init__(
        self,
        data: pd.DataFrame | pd.Series,
        buyon: str = "close",
        sellon: str = "close",
        code_level: int | str = 0,
        date_level: int | str = 1,
    ):
        self._date_level = date_level
        Return.__init__(self, data, buyon, sellon, code_level)
        Dim2Series.__init__(self, data)
    
    def __compute(
        self, _event: pd.Series, 
        start: int | float | str, 
        stop: int | float | str
    ) -> pd.Series:
        _event_start = _event[_event == start].index
        if _event_start.empty:
            return pd.Series(index=pd.DatetimeIndex([pd.NaT], name=self._date_level))
        _event_start = _event_start.get_level_values(self._date_level)[0]
        _event = _event.loc[_event.index.get_level_values(self._date_level) >= _event_start]

        _event_diff = _event.diff()
        _event_diff.iloc[0] = _event.iloc[0]
        _event = _event[_event_diff != 0]
        
        pbuy = self._pbuy.loc[_event.index].loc[_event == start]
        psell = self._psell.loc[_event.index].loc[_event == stop]

        if pbuy.shape[0] - psell.shape[0] > 1:
            raise ValueError("there are unmatched start-stop labels")
        pbuy = pbuy.iloc[:psell.shape[0]]
        idx = pd.IntervalIndex.from_arrays(
            left = pbuy.index.get_level_values(self._date_level),
            right = psell.index.get_level_values(self._date_level),
            name=self._date_level
        )
        pbuy.index = idx
        psell.index = idx

        if pbuy.empty:
            return pd.Series(index=pd.DatetimeIndex([pd.NaT], name=self._date_level))
        
        return psell / pbuy - 1
        
    def __call__(self, start: int | float, stop: int | float) -> pd.Series:
        event = Dim2Series(event, self._code_level)._data
        if set(event.unique()) - set([start, stop]):
            raise ValueError("there are labels that are not start-stop labels")

        res = event.groupby(level=self._code_level).apply(self.__compute, start=start, stop=stop)
        return res


class Weight(Return, Dim2Frame):

    def __init__(
        self,
        data: pd.DataFrame | pd.Series,
        buyon: str = "open",
        sellon: str = "close",
        code_level: str | int = 0,
    ):
        Return.__init__(self, data, buyon, sellon, code_level)
        Dim2Frame.__init__(self, data, code_level)
        
    def transform(
        self, 
        weight: pd.DataFrame | pd.Series,
        buyat: int = 1,
        sellat: int = 20,
        commission: float = 0.005,
        side: str = 'both',
        return_tvr: bool = False,
    ) -> tuple[pd.Series, pd.Series] | pd.Series:    
        weight = Dim2Frame(weight, self._code_level)._data

        delta = weight.fillna(0) - weight.shift(abs(sellat)).fillna(0)
        if side == 'both':
            tvr = (delta.abs() / 2).sum(axis=1) / abs(sellat)
        elif side == 'buy':
            tvr = delta.where(delta > 0).abs().sum(axis=1) / abs(sellat)
        elif side == 'sell':
            tvr = delta.where(delta < 0).abs().sum(axis=1) / abs(sellat)
        else:
            raise ValueError("side must be in ['both', 'buy', 'sell']")
        commission *= tvr

        r = Return.__call__(self, buyat, sellat)
        r = (((r * weight).sum(axis=1) - commission) 
             / abs(sellat)).shift(-min(0, sellat)).fillna(0)

        if return_tvr:
            return r, tvr
        return r


class Rebalance(Return, Dim2Frame):

    def __init__(
        self,
        data: pd.DataFrame | pd.Series,
        buyon: str = "open",
        sellon: str = "close",
        code_level: str | int = 0,
    ):
        Return.__init__(self, data, buyon, sellon, code_level)
        Dim2Frame.__init__(self, data, code_level)
    
    def transform(
        self, 
        weight: pd.DataFrame | pd.Series,
        commission: float = 0.005,
        side: str = 'both',
        return_tvr: bool = False,
    ) -> tuple[pd.Series, pd.Series] | pd.Series:
        weight = Dim2Frame(weight, self._code_level)._data

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
        
        r = Return.__call__(self, -1, 0)
        weight = weight.fillna(0).reindex(r.index).ffill()
        r *= weight
        r = r.sum(axis=1) - commission.reindex(r.index).fillna(0)
        
        if return_tvr:
            return r, tvr
        return r


class RobustScaler(Dim2Frame):
    
    def __call__(self, method: str, n: int):
        if method == "mad":
            median = self._data.median(axis=1)
            ad = self._data.sub(median, axis=0)
            mad = ad.abs().median(axis=1)
            thresh_up = median + n * mad
            thresh_down = median - n * mad
        elif method == "std":
            mean = self._data.mean(axis=1)
            std = self._data.std(axis=1)
            thresh_up = mean + std * n
            thresh_down = mean - std * n
        elif method == "iqr":
            thresh_down = self._data.quantile(n, axis=1)
            thresh_up = self._data.quantile(1 - n, axis=1)
        else:
            raise ValueError("Invalid method.")
        
        if method == 'mad' or method == "std":
            self._data.clip(thresh_down, thresh_up, axis=0).where(~self._data.isna())
        elif method == 'iqr':
            self._data.mask(
                self._data.ge(thresh_up, axis=0) & self._data.le(thresh_down, axis=0), 
            np.nan, axis=0).where(~self._data.isna())
        else:
            raise ValueError("Invalid method.")
        
        return self._data


class StandardScaler(Data):
    
    def __call__(self, method: str):
        if method == "zscore":
            mean = self._data.mean(axis=1)
            std = self._data.std(axis=1)
        elif method == "mad":
            max = self._data.max(axis=1)
            min = self._data.min(axis=1)
        else:
            raise ValueError("Invalid method.")

        if method == 'zscore':
            return self._data.sub(mean, axis=0).div(std, axis=0)
        elif method == 'minmax':
            return self._data.sub(min, axis=0).div((max - min), axis=0)
        else:
            raise ValueError("Invalid method.")

class Imputer(Dim2Frame):
        
    def fit(self, method: str):
        if method == "median":
            filler = self._data.median(axis=1)
        elif method == "mod":
            filler = self._data.mode(axis=1)
        elif method == "mean":
            filler = self._data.mean(axis=1)
        else:
            raise ValueError("Invalid method.")
        
        if method == 'mean':
            return self._data.T.fillna(filler).T
        elif method == 'median':
            return self._data.T.fillna(filler).T
        elif method == 'mod':
            return self._data.T.fillna(filler).T
        else:
            raise ValueError('method must be "zero", "mean", "median", "ffill", or "bfill"')
        

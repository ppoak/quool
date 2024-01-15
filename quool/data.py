import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from .core.data import Data
from .core.exception import NotRequiredDimError
from .operate import Shift, LevelShift, CsCorr


class Dim1Data(Data):

    def __init__(self, data: pd.Series | pd.DataFrame):
        super().__init__(data)
        if self.ndims != 1:
            raise NotRequiredDimError(1)


class Dim2Data(Data):

    def __init__(self, data: pd.Series | pd.DataFrame):
        super().__init__(data)
        if self.ndims != 2:
            raise NotRequiredDimError(2)


class Dim2Frame(Dim2Data):

    def __init__(
        self, 
        data: pd.Series | pd.DataFrame, 
        level: int | str = 0
    ):
        self._level = level
        super().__init__(data)
        if self.rowdim == 2:
            self.swapdim(level, -1)


class Dim2Series(Dim2Data):

    def __init__(
        self, 
        data: pd.Series | pd.DataFrame, 
        level: int | str = 0
    ):
        self._level = level
        super().__init__(data)
        if self.rowdim == 1:
            self.swapdim(-1, level)


class Dim3Data(Data):

    def __init__(self, data: pd.Series | pd.DataFrame):
        super().__init__(data)
        if self.ndims != 3:
            raise NotRequiredDimError(3)


class Dim3Frame(Dim3Data):

    def __init__(self, data: pd.Series | pd.DataFrame, level: int | str = 0):
        self._level = level
        super().__init__(data)
        if self.rowdim == 3:
            self.swapdim(level, -1)


class Price(Data):

    def __init__(
        self,
        data: pd.DataFrame | pd.Series,
        code_level: str | int = 0,
    ):
        self._code_level = code_level
        super().__init__(data)
        if self.ndims > 3:
            raise NotRequiredDimError(3)
        if self.rowdim > 2:
            raise NotRequiredDimError(2)
    
    def __rshift__(self, n: int):
        if self.rowdim == 1:
            return Shift()(self, n)
        elif self.rowdim == 2:
            return LevelShift()(self, self._code_level, n)
    
    def __lshift__(self, n: int):
        return self.__rshift__(-n)


class Dim2PriceFrame(Dim2Frame, Price):
    pass


class Dim2PriceSeries(Dim2Series, Price):
    pass


class Event(Dim2Series):

    def __call__(
        self, 
        buy: Dim2PriceSeries, 
        sell: Dim2PriceSeries = None, 
        span: tuple = (-5, 6, 1), 
    ):
        sell = sell or buy
        res = []
        r = sell.shift(1, level=self._level) / buy - 1
        for i in np.arange(*span):
            res.append(r.groupby(level=self._level).shift(-i).loc[self._data.index])
        res = pd.concat(res, axis=1, keys=np.arange(*span)).add_prefix('day').fillna(0)
        cumres = (1 + res).cumprod(axis=1)
        cumres = cumres.div(cumres["day0"], axis=0).mean(axis=0)
        return cumres


class PairEvent(Dim2Series):

    def __init__(
        self, 
        data: pd.Series | pd.DataFrame, 
        code_level: int | str = 0, 
        date_level: int | str = 1
    ):
        super().__init__(data, code_level)
        self._date_level = date_level
        self._code_level = code_level

    def __compute(
        self, 
        _buy: pd.Series, 
        _sell: pd.Series,
        _event: pd.Series, 
        _start: int | float | str, 
        _stop: int | float | str
    ) -> pd.Series:
        _event_start = _event[_event == _start].index
        if _event_start.empty:
            return pd.Series(index=pd.DatetimeIndex([pd.NaT], name=self._date_level))
        _event_start = _event_start.get_level_values(self._date_level)[0]
        _event = _event.loc[_event.index.get_level_values(self._date_level) >= _event_start]

        _event_diff = _event.diff()
        _event_diff.iloc[0] = _event.iloc[0]
        _event = _event[_event_diff != 0]
        
        pbuy = _buy.loc[_event.index].loc[_event == _start]
        psell = _sell.loc[_event.index].loc[_event == _stop]

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
    
    def __call__(
        self, 
        buy: Dim2PriceSeries, 
        sell: Dim2PriceSeries = None, 
        start: int = 1, 
        stop: int = -1
    ):
        sell = sell or buy
        if set(self._data.unique()) - {start, stop}:
            raise ValueError("extra label other than start and stop found")
        return self._data.groupby(level=self._code_level).apply(
            self.__compute, _buy=buy(), _sell=sell(),
            _start=start, _stop=stop
        )


class Weight(Dim2Frame):

    def __init__(
        self, 
        data: pd.Series | pd.DataFrame, 
        level: int | str = 0
    ):
        super().__init__(data, level)
        self._data = self._data.div(self._data.sum(axis=1), axis=0)

    def __call__(
        self, 
        returns: Dim2Frame,
        commission: float = 0.005,
        side: str = 'both',
    ):
        delta = self._data.fillna(0) - self._data.shift(1).fillna(0)
        if side == 'both':
            turnover = (delta.abs() / 2).sum(axis=1)
        elif side == 'buy':
            turnover = delta.where(delta > 0).abs().sum(axis=1)
        elif side == 'sell':
            turnover = delta.where(delta < 0).abs().sum(axis=1)
        else:
            raise ValueError("side must be in ['both', 'buy', 'sell']")
        commission *= turnover
        
        weight = self._data.fillna(0).reindex(returns.index).ffill()
        returns *= weight
        returns = returns.sum(axis=1) - commission.reindex(returns.index).fillna(0)
        
        return pd.concat([returns, turnover], axis=1, keys=["returns", "turnover"])


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


class StandardScaler(Dim2Frame):
    
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
        
    def __call__(self, method: str | float | int):
        if method == "median":
            filler = self._data.median(axis=1)
        elif method == "mod":
            filler = self._data.mode(axis=1)
        elif method == "mean":
            filler = self._data.mean(axis=1)
        elif isinstance(method, (float, int)):
            filler = method
        else:
            raise ValueError("Invalid method.")
        
        if method == 'mean':
            return self._data.T.fillna(filler).T
        elif method == 'median':
            return self._data.T.fillna(filler).T
        elif method == 'mod':
            return self._data.T.fillna(filler).T
        elif isinstance(method, (int, float)):
            return self._data.T.fillna(method).T
        else:
            raise ValueError('method must be "mean", "median", "mod" or a number')


class Factor(Dim2Frame):

    def preprocess(
        self, 
        robust_method: str = "mad",
        n: int = 5,
        standard_method: str = "zscore",
        imputer_method: str = np.nan,
    ):
        self._data = RobustScaler(self._data, self._level)(robust_method, n)
        self._data = StandardScaler(self._data, self._level)(standard_method)
        self._data = Imputer(self._data, self._level)(imputer_method)
        return self
    
    def report_cross_section(
        self,
        future_return: Dim2Frame,
        crossdate: str,
        image_path: str = None,
        data_path: str = None,
    ):
        crossdata = self._data.loc[crossdate]
        frdata = future_return().loc[crossdate]
        data = pd.concat([crossdata, frdata], axis=1, keys=['factor', 'future return'])
        if image_path:
            fig, axes = plt.subplots(2, 1, figsize=(20, 20))
            axeslist = axes.tolist()
            data.iloc[:, 0].plot.hist(bins=300, ax=axeslist.pop(0), title=f'distribution')
            data.plot.scatter(x=data.columns[0], y=data.columns[1], ax=axeslist.pop(0), title=f"scatter")
            fig.tight_layout()
            fig.savefig(image_path)
        if data_path:
            data.to_excel(data_path, sheet_name=f'cross-section')
        
        return data

    def report_information_coef(
        self, 
        future_return: Dim2Frame,
        method: str = 'pearson',
        image_path: str = None,
        data_path: str = None,
    ):
        inforcoef = CsCorr()(self, future_return, method=method)
        if image_path:
            fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            ax = inforcoef.plot.bar(title=f'information coefficient', ax=ax, figsize=(20, 10))
            ax.set_xticks(
                [i for i in range(0, inforcoef.shape[0], int(inforcoef.shape[0] * 0.1))], 
                [inforcoef.index[i].strftime(r'%Y-%m-%d') 
                    for i in range(0, inforcoef.shape[0], int(inforcoef.shape[0] * 0.1))])
            fig.tight_layout()
            fig.savefig(image_path)
        if data_path:
            inforcoef.to_excel(data_path, sheet_name=f'information coefficient')
        return inforcoef
    
    def report_group_test(
        self,
        future_return: Dim2Frame,
        benchmark: Dim1Data = None,
        ngroup: int = 10,
        commission: float = 0.005,
        side: str = 'both',
        n_jobs: int = -1,
        image_path: str = None,
        data_path: str = None,
    ):
        @delayed
        def _backtest(n):
            group = Weight(groups.mask(groups != n, np.nan))
            return group(future_return, commission, side)
        
        if benchmark is not None:
            benchmark = benchmark().pct_change().fillna(0)
        
        groups = self().apply(lambda x: pd.qcut(x, ngroup, labels=False), axis=1) + 1
        result = Parallel(n_jobs=n_jobs, backend='loky')(_backtest(n) for n in range(1, ngroup + 1))

        profit = pd.concat([res.iloc[:, 0] for res in result], axis=1, keys=range(1, ngroup + 1)).add_prefix('Group')
        turnover = pd.concat([res.iloc[:, 1] for res in result], axis=1, keys=range(1, ngroup + 1)).add_prefix('Group')
        if benchmark is not None:
            exprofit = profit.sub(benchmark, axis=0)
            profit = pd.concat([profit, benchmark], axis=1)
            excumprofit = (exprofit + 1).cumprod()
        cumprofit = (profit + 1).cumprod()

        if image_path:
            fig, axes = plt.subplots(2 + int(benchmark is not None), 1, 
                figsize=(20, 20 + 10 * int(benchmark is not None)))
            axeslist = axes.tolist()
            cumprofit.plot(ax=axeslist.pop(0), title=f'cumulative netvalue')
            if benchmark is not None:
                excumprofit.plot(ax=axeslist.pop(0), title=f'execess cumulative netvalue')
            turnover.plot(ax=axeslist.pop(0), title=f'turnover')
            fig.tight_layout()
            fig.savefig(image_path)
        
        if data_path:
            with pd.ExcelWriter(data_path) as writer:
                profit.to_excel(writer, sheet_name=f'profit')
                cumprofit.to_excel(writer, sheet_name=f'cumprofit')
                if benchmark is not None:
                    excumprofit.to_excel(writer, sheet_name=f'excumprofit')
                turnover.to_excel(writer, sheet_name=f'turnover')
        
        return result


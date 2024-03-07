import quool
import pandas as pd
from config import qtd


class Factor(quool.Factor):

    def get_future(
        self, 
        period: int = 1, 
        ptype: str = "close",
        start: str | pd.Timestamp = None,
        stop: str | pd.Timestamp = None,
    ):
        if stop is not None:
            stop = self.get_trading_days_rollback(stop, -period - 1)
        price = qtd.read([ptype, "st", "suspended"], start=start, stop=stop)
        price = price[ptype].where(~(price["st"].fillna(True) | price["suspended"].fillna(True)))
        price = price.unstack(self._code_level)
        future = price.shift(-1 - period) / price.shift(-1) - 1
        return future.dropna(axis=0, how='all').squeeze()

    def perform_crosssection(
        self, name: str, 
        date: str | pd.Timestamp,
        period: int = 1,
        ptype: str = "close",
        image: str | bool = True, 
        result: str = None
    ):
        future = self.get_future(period, ptype, date, date)
        return super().perform_crosssection(name, future, image, result) 
    
    def perform_inforcoef(
        self, name: str, 
        period: int = 1,
        start: str = None,
        stop: str = None,
        ptype: str = "close",
        rolling: int = 20, 
        method: str = 'pearson', 
        image: str | bool = True, 
        result: str = None
    ):
        future = self.get_future(period, ptype, start, stop)
        return super().perform_inforcoef(name, future, rolling, method, image, result)
    
    def perform_grouping(
        self, 
        name: str, 
        period: int = 1,
        start: str = None,
        stop: str = None,
        ptype: str = "close",
        ngroup: int = 5, 
        commission: float = 0.002, 
        image: str | bool = True, 
        result: str = None
    ):
        future = self.get_future(period, ptype, start, stop)
        return super().perform_grouping(name, future, ngroup, commission, image, result)
    
    def perform_topk(
        self, 
        name: str, 
        period: int = 1,
        start: str = None,
        stop: str = None,
        ptype: str = "close",
        topk: int = 100, 
        commission: float = 0.002, 
        image: str | bool = True, 
        result: str = None
    ):
        future = self.get_future(period, ptype, start, stop)
        return super().perform_topk(name, future, topk, commission, image, result)

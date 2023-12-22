import numpy as np
import pandas as pd


class Return:

    def __init__(
        self,
        price: pd.DataFrame | pd.Series,
        buy_column: str = "open",
        sell_column: str = "close",
        code_index: str = "code",
        date_index: str = "date",
        delay: int = 1,
    ):
        if isinstance(price, pd.DataFrame) and not \
            isinstance(price.index, pd.MultiIndex):
            self.date_index = price.index.name
            self.code_index = price.columns.name
            if self.code_index is None:
                self.code_index = code_index
                price.columns.name = self.code_index
            if self.date_index is None:
                self.date_index = date_index
                price.index.name = self.date_index
            self.price = price.shift(-delay)
            self.buy_price = self.price
            self.sell_price = self.price
        
        elif isinstance(price, pd.DataFrame) and \
            isinstance(price.index, pd.MultiIndex):
            self.date_index = date_index
            self.code_index = code_index
            self.price = price.groupby(level=self.code_index).shift(-delay)
            self.buy_price = self.price[buy_column]
            self.sell_price = self.price[sell_column]
        
        elif isinstance(price, pd.Series) and \
            isinstance(price.index, pd.MultiIndex):
            self.date_index = date_index
            self.code_index = code_index
            self.price = price.groupby(level=self.code_index).shift(-delay)
            self.buy_price = self.price
            self.sell_price = self.price
    
    def ret(self, span: int = -1, log: bool = False) -> pd.Series:
        if not isinstance(self.price.index, pd.MultiIndex):
            if span < 0:
                sell_price = self.sell_price.shift(span)
                buy_price = self.buy_price
            else:
                sell_price = self.sell_price
                buy_price = self.buy_price.shift(span)
        
        else:
            if span < 0:
                sell_price = self.sell_price.groupby(level=self.code_index).shift(span)
                buy_price = self.buy_price
            else:
                sell_price = self.sell_price
                buy_price = self.buy_price.groupby(level=self.code_index).shift(span)
        
        if log:
            return np.log(sell_price / buy_price)
        return sell_price / buy_price - 1


class Event(Return):
    
    def __init__(
        self,
        price: pd.Series,
        buy_column: str = "close",
        sell_column: str = "close",
        code_index: str = 'code',
        date_index: str = 'date',
    ):
        if not isinstance(price.index, pd.MultiIndex):
            raise ValueError("price must be a Series with MultiIndex")
        super().__init__(price, buy_column, sell_column,
            code_index, date_index, 0)

    def ret(
        self,
        event: pd.Series,
        span: tuple = (-5, 6, 1),
    ):
        if not isinstance(self.price.index, type(event.index)):
            raise ValueError("the type of price and event must be the same")

        res = []
        r = super().ret(span=1)

        for i in np.arange(*span):
            res.append(r.groupby(level=self.code_index).shift(-i).loc[event.index])
                
        res = pd.concat(res, axis=1, keys=np.arange(*span)).add_prefix('day').fillna(0)
        return res


class Weight(Return):

    def __init__(
        self,
        price: pd.DataFrame | pd.Series,
        buy_column: str = "open",
        sell_column: str = "close",
        code_index: str = 'code',
        date_index: str = 'date',
        delay: int = 1,
    ):
        super().__init__(price, buy_column, sell_column,
            code_index, date_index, delay)
    
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
            weight = weight.unstack(level=self.code_index)
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

        r = super().ret(span=span)
        if isinstance(self.price.index, pd.MultiIndex):
            r = r.unstack(level=self.code_index)

        r = (((r * weight).sum(axis=1) - commission) 
             / abs(span)).shift(-min(0, span)).fillna(0)

        if return_tvr:
            return r, tvr
        return r


class NetValue(Return):

    def __init__(
        self,
        price: pd.DataFrame | pd.Series,
        buy_column: str = "close",
        sell_column: str = "close",
        code_index: str = 'code',
        date_index: str = 'date',
    ):
        super().__init__(price, buy_column, sell_column, 
            code_index, date_index, 0)
    
    def ret(
        self, 
        weight: pd.DataFrame | pd.Series, 
        commission: float = 0.005,
        side: str = 'both',
        return_tvr: bool = False,
    ) -> tuple[pd.Series, pd.Series] | pd.Series:
        if isinstance(weight, pd.Series) and \
            isinstance(weight.index, pd.MultiIndex):
            weight = weight.unstack(level=self.code_index)
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

        r = super().ret(span=1).unstack(level=self.code_index)
        r = (r.fillna(0) * weight.fillna(0).reindex(r.index).ffill()).sum(axis=1) - \
            commission.reindex(r.index).fillna(0)
            
        
        if return_tvr:
            return r, tvr
        return r


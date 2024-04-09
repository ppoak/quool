import numpy as np
import pandas as pd
from base import (
    fqtd, BaseFactor
)

    
class MomentumFactor(BaseFactor):
    def get_strev(self, date: str):
        rollback = fqtd.get_trading_days_rollback(date, 21)
        price = fqtd.read("close", start=rollback, stop=date)
        ret  = np.log(1 + price.pct_change()).ewm(halflife=4).mean()
        res = ret.rolling(21).mean().dropna(how='all')
        return res
    
    def get_rstr(self, date: str):
        rollback = fqtd.get_trading_days_rollback(date, 504)
        price = fqtd.read("close", start=rollback, stop=date)
        ret  = np.log(1 + price.pct_change()).ewm(halflife=126).mean()
        res = ret.rolling(504).mean().dropna(how='all')
        return res
    
    def get_momentum(self, date: str):
        lt = self.get_rstr(date)
        st = self.get_strev(date)

        res = lt - st 
        return res.stack().droplevel(level='date')
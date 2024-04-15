import numpy as np
import pandas as pd
from .base import (
    fqtd, BaseFactor
)

    
class MomentumFactor(BaseFactor):
    def get_rstr(self, date: str): # with a lag of L=21 tradingdays
        rollback = fqtd.get_trading_days_rollback(date, 525)
        prices = fqtd.read("close", start=rollback, stop=date)
        adjfactor = fqtd.read("adjfactor", start=rollback, stop=date)
        prices_adj = prices * adjfactor
        ret  = np.log(1 + prices_adj.pct_change(fill_method=None)).ewm(halflife=126).mean()
        res = ret.iloc[:504].sum()
        res.name = date
        return res
    
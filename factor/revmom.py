import numpy as np
import pandas as pd
from base import (
    fqtd, BaseFactor
)

    
class MomentumFactor(BaseFactor):
    def get_rstr(self, date: str): # with a lag of L=21 tradingdays
        rollback = fqtd.get_trading_days_rollback(date, 504+21)
        prices = fqtd.read("close", start=rollback, stop=date)
        adjfactor = fqtd.read("adjfactor", start=rollback, stop=date)
        prices_adj = prices * adjfactor
        ret  = np.log(1 + prices_adj.pct_change()).ewm(halflife=126).mean()

        for end_date in ret.index[21:]:
            start_date = ret.index[ret.index.get_loc(end_date) - 504-21]
            res = ret.loc[start_date:end_date].iloc[:-21]
        res = res.sum()
        res.name = date
        return res
    
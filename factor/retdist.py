import numpy as np
import pandas as pd
from .base import (
    fqtd, fqtm, Factor
)


class RetDistFactor(Factor):

    def get_intraday_distribution(self, date: str) -> pd.DataFrame:
        data = fqtm.read("close", start=date, stop=date + pd.Timedelta(days=1))
        ret = data.pct_change(fill_method=None)
        res = pd.concat([ret.skew(), ret.kurt()], axis=1, 
            keys=['intraday_return_skew', 'intraday_return_kurt'])
        res.index = pd.MultiIndex.from_product([
            res.index, [date]], names=["order_book_id", "date"])
        return res

    def get_down_trend_volatility(self, date: str) -> pd.DataFrame:
        price = fqtm.read("close", start=date, stop=date + pd.Timedelta(days=1))
        ret = price.pct_change(fill_method=None)
        res = ret.apply(lambda x: x[x < 0].pow(2).sum() / x.pow(2).sum())
        res.name = date
        return res
        
    def get_long_short_ratio(self, date: str) -> pd.DataFrame:
        rollback = fqtd.get_trading_days_rollback(date, 5)
        price = fqtm.read("close", start=rollback, stop=date + pd.Timedelta(days=1))
        vol = fqtm.read("volume", start=rollback, stop=date + pd.Timedelta(days=1))
        ret = price.pct_change(fill_method=None)
        vol_per_unit = abs(vol / ret).replace([np.inf, -np.inf], np.nan)
        tot_ret = (price.iloc[-1] / price.iloc[0] - 1).abs()
        res = (tot_ret * vol_per_unit.mean()) / vol.sum()
        res.name = date
        return res


rdf = RetDistFactor("./data/factor", code_level="order_book_id", date_level="date")

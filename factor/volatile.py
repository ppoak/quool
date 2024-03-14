import pandas as pd
from .base import (
    fqtd, fqtm, Factor
)


class VolatileFactor(Factor):

    def get_information_distribution_uniformity(self, date: str):
        rollback = fqtd.get_trading_days_rollback(date, 20)
        price = fqtm.read("close", start=rollback, stop=date + pd.Timedelta(days=1))
        ret = price.groupby(price.index.date).pct_change(fill_method=None)
        std = ret.groupby(ret.index.date).std()
        res = std.std() / std.mean()
        res.name = date
        return res

vtf = VolatileFactor("./data/factor", code_level="order_book_id", date_level="date")
import numpy as np
import pandas as pd
from .base import (
    fcon, fqtd, Factor
)


class CapFlowFactor(Factor):

    def get_stock_connect_stableinc(self, date: pd.Timestamp) -> pd.Series:
        rollback = fqtd.get_trading_days_rollback(date, 20)
        chold = fcon.read("shares_holding", start=rollback, stop=date)
        shares = fqtd.read("circulation_a", start=rollback, stop=date)
        per = chold / shares
        res = (per.iloc[-1] - per.iloc[0]) / per.std()
        res.name = date
        return res


cff = CapFlowFactor("./data/factor", code_level="order_book_id", date_level="date")

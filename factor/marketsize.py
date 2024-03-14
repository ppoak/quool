import numpy as np
import pandas as pd
from .base import (
    fqtd, Factor
)


class MarketSizeFactor(Factor):

    def get_barra_log_marketcap(self, date: str):
        shares = fqtd.read("circulation_a", start=date, stop=date)
        price = fqtd.read("close", start=date, stop=date)
        adjfactor = fqtd.read("adjfactor", start=date, stop=date)
        res = np.log(shares * price * adjfactor).loc[date]
        return res

msf = MarketSizeFactor("./data/factor", code_level="order_book_id", date_level="date")

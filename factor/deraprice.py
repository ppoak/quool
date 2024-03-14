import numpy as np
import pandas as pd
from .base import (
    fqtd, fqtm, Factor
)


class DeraPriceFactor(Factor):

    def get_volume_weighted_price(self, date: pd.Timestamp):
        p = fqtm.read("close", start=date, stop=date + pd.Timedelta(days=1))
        vol = fqtm.read("volume", start=date, stop=date + pd.Timedelta(days=1))
        stsus = fqtd.read("st, suspended", start=date, stop=date)
        adjfactor = fqtd.read("adjfactor", start=date, stop=date).squeeze()
        nontradable = stsus["st"] | stsus["suspended"]
        nontradable = nontradable.unstack(level=fqtd._code_level).fillna(True).squeeze()
        w = vol / vol.sum()
        res = ((p * w).sum() * adjfactor).where(~nontradable)
        res.name = date
        return res

    def get_price_volume_corr(self, date: pd.Timestamp) -> pd.DataFrame:
        price = fqtm.read("close", start=date, stop=date + pd.Timedelta(days=1))
        volume = fqtm.read("volume", start=date, stop=date + pd.Timedelta(days=1))
        res = price.corrwith(volume, axis=0).replace([np.inf, -np.inf], np.nan)
        res.name = date
        return res

    def get_average_relative_price_percent(self, date: pd.Timestamp) -> pd.DataFrame:
        df = fqtm.read("open, high, low, close", start=date, stop=date + pd.Timedelta(days=1))
        twap = df.mean(axis=1).groupby(level=fqtm._code_level).mean()
        high = df["high"].groupby(level=fqtm._code_level).max()
        low = df["low"].groupby(level=fqtm._code_level).min()
        arrp = (twap - low) / (high - low)
        arrp.name = date
        return arrp


dpf = DeraPriceFactor("./data/factor", code_level="order_book_id", date_level="date")

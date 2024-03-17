import pandas as pd
from .base import (
    fqtm, BaseFactor
)


class VolDistFactor(BaseFactor):

    def get_tail_volume_percent(self, date: pd.Timestamp):
        data = fqtm.read("volume", start=date, stop=date + pd.Timedelta(days=1))
        tail_vol = data.between_time("14:31", "14:57").sum()
        day_vol = data.sum()
        res = tail_vol / day_vol
        res.name = date
        return res

    def get_volume_peak_count(self, date: pd.Timestamp):
        volume = fqtm.read("volume", start=date, stop=date + pd.Timedelta(days=1))
        mean = volume.mean()
        std = volume.std()
        peaks = volume[volume > mean + std]
        filt = peaks.notna() & peaks.shift().notna()
        peaks = peaks.where(~filt)
        res = peaks.count()
        res.name = date
        return res

    def get_foggy_amount_ratio(self, date: pd.Timestamp) -> pd.Series:
        price = fqtm.read("close", start=date, stop=date + pd.Timedelta(days=1))
        vol = fqtm.read("volume", start=date, stop=date + pd.Timedelta(days=1))
        amount = price * vol
        ret = price.pct_change(fill_method=None)
        std = ret.rolling(5).std()
        blur = std.rolling(5).std()
        foggy = blur[blur > blur.mean()]
        foggyamt = amount.where(foggy.notna())
        res = foggyamt.sum() / amount.sum()
        res.name = date
        return res


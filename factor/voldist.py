import pandas as pd
from .base import (
    fqtm, Factor
)


class VolDistFactor(Factor):

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


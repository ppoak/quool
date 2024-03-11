import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from .base import (
    Factor,
    fqtd, fqtm,
)


class MinFreqFactor(Factor):

    def get_tail_volume_percent(self, start: str = None, stop: str = None) -> pd.DataFrame:
        def _get(date: pd.Timestamp):
            data = fqtm.read("volume", start=date, stop=date + pd.Timedelta(days=1))
            tail_vol = data.between_time("14:31", "14:57").sum()
            day_vol = data.sum()
            res = tail_vol / day_vol
            res.name = date
            return res
            
        start = start or pd.to_datetime('now').strftime(r"%Y-%m-%d")
        stop = stop or pd.to_datetime('now').strftime(r"%Y-%m-%d")
        trading_days = fqtd.get_trading_days(start, stop)
        result = Parallel(n_jobs=-1, backend='loky')(
            delayed(_get)(date) for date in tqdm(list(trading_days))
        )
        return pd.concat(result, axis=1).T.loc[start:stop]

    def get_intraday_distribution(self, start: str = None, stop: str = None) -> pd.DataFrame:
        def _get(date: pd.Timestamp):
            data = fqtm.read("close", start=date, stop=date + pd.Timedelta(days=1))
            ret = data.pct_change(fill_method=None)
            res = pd.concat([ret.skew(), ret.kurt()], axis=1, 
                keys=['intraday_return_skew', 'intraday_return_kurt'])
            res.index = pd.MultiIndex.from_product([
                res.index, [date]], names=["order_book_id", "date"])
            return res
        
        start = start or pd.to_datetime('now').strftime(r"%Y-%m-%d")
        stop = stop or pd.to_datetime('now').strftime(r"%Y-%m-%d")
        trading_days = fqtd.get_trading_days(start, stop)
        result = Parallel(n_jobs=-1, backend='loky')(
            delayed(_get)(date) for date in tqdm(list(trading_days))
        )
        return pd.concat(result, axis=0).sort_index().loc(axis=0)[:, start:stop]

    def get_long_short_ratio(self, start: str = None, stop: str = None):
        def _get(start, stop):
            price = fqtm.read("close", start=start, stop=stop + pd.Timedelta(days=1))
            vol = fqtm.read("volume", start=start, stop=stop + pd.Timedelta(days=1))
            ret = price.pct_change(fill_method=None)
            vol_per_unit = abs(vol / ret).replace([np.inf, -np.inf], np.nan)
            tot_ret = (price.iloc[-1] / price.iloc[0] - 1).abs()
            res = (tot_ret * vol_per_unit.mean()) / vol.sum()
            res.name = stop
            return res
        
        start = start or pd.to_datetime('now').strftime(r"%Y-%m-%d")
        stop = stop or pd.to_datetime('now').strftime(r"%Y-%m-%d")
        rollback = fqtd.get_trading_days_rollback(start, 5)
        trading_days = fqtd.get_trading_days(rollback, stop)
        result = Parallel(n_jobs=-1, backend='loky')(delayed(_get)
            (_start, _stop) for _start, _stop in  tqdm(list(
                zip(trading_days[:-4], trading_days[4:])
        )))
        return pd.concat(result, axis=1).T.loc[start:stop]

    def get_price_volume_corr(self, start: str = None, stop: str = None) -> pd.DataFrame:
        def _get(date: pd.Timestamp):
            price = fqtm.read("close", start=date, stop=date + pd.Timedelta(days=1))
            volume = fqtm.read("volume", start=date, stop=date + pd.Timedelta(days=1))
            res = price.corrwith(volume, axis=0).replace([np.inf, -np.inf], np.nan)
            res.name = date
            return res

        start = start or pd.to_datetime('now').strftime(r"%Y-%m-%d")
        stop = stop or pd.to_datetime('now').strftime(r"%Y-%m-%d")
        trading_days = fqtd.get_trading_days(start, stop)
        return pd.concat(Parallel(n_jobs=-1, backend='loky')(delayed(_get)
            (date) for date in tqdm(list(trading_days))), axis=1).T.loc[start:stop]


mff = MinFreqFactor("./data/minfreq", code_level="order_book_id", date_level="date")

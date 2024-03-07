import pandas as pd
from tqdm import tqdm
from .base import (
    Factor,
    fqtd, fqtm,
)
from joblib import Parallel, delayed


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


mff = MinFreqFactor("./data/minfreq", code_level="order_book_id", date_level="date")

import quool
import config
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed


class MinFreqFactor(quool.Factor):

    def get_tail_volume_percent(self, start: str = None, stop: str = None) -> pd.DataFrame:
        def _get(date: pd.Timestamp):
            data = config.fqtm.read("volume", start=date, stop=date + pd.Timedelta(days=1))
            tail_vol = data.between_time("14:31", "14:57").sum()
            day_vol = data.sum()
            res = tail_vol / day_vol
            res.name = date
            return res
            
        start = start or pd.to_datetime('now').strftime(r"%Y-%m-%d")
        stop = stop or pd.to_datetime('now').strftime(r"%Y-%m-%d")
        trading_days = config.fqtd.get_trading_days(start, stop)
        return pd.concat(Parallel(n_jobs=-1, backend='loky')(
            delayed(_get)(date) for date in tqdm(list(trading_days))
        ), axis=1).T.loc[start:stop]

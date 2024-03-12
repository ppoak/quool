import quool as q
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed


def zscore(df: pd.DataFrame):
    return df.sub(df.mean(axis=1), axis=0
        ).div(df.std(axis=1), axis=0)

def minmax(df: pd.DataFrame):
    return df.sub(df.min(axis=1), axis=0).div(
        df.max(axis=1) - df.min(axis=1), axis=0)

def madoutlier(
    df: pd.DataFrame, 
    dev: int, 
    drop: bool = False
):
    median = df.median(axis=1)
    ad = df.sub(median, axis=0)
    mad = ad.abs().median(axis=1)
    thresh_down = median - dev * mad
    thresh_up = median + dev * mad
    if not drop:
        return df.clip(thresh_down, thresh_up, axis=0).where(~df.isna())
    return df.where(
        df.le(thresh_up, axis=0) & df.ge(thresh_down, axis=0),
        other=np.nan, axis=0).where(~df.isna())

def stdoutlier(
    df: pd.DataFrame, 
    dev: int, 
    drop: bool = False
):
    mean = df.mean(axis=1)
    std = df.std(axis=1)
    thresh_down = mean - dev * std
    thresh_up = mean + dev * std
    if not drop:
        return df.clip(thresh_down, thresh_up, axis=0).where(~df.isna())
    return df.where(
        df.le(thresh_up, axis=0) & df.ge(thresh_down, axis=0),
        other=np.nan, axis=0).where(~df.isna())

def iqroutlier(
    df: pd.DataFrame, 
    dev: int, 
    drop: bool = False
):
    thresh_up = df.quantile(1 - dev / 2, axis=1)
    thresh_down = df.quantile(dev / 2, axis=1)
    if not drop:
        return df.clip(thresh_down, thresh_up, axis=0).where(~df.isna())
    return df.where(
        df.le(thresh_up, axis=0) & df.ge(thresh_down, axis=0),
        other=np.nan, axis=0).where(~df.isna())

def fillna(
    df: pd.DataFrame,
    val: int | str = 0,
):
    return df.fillna(val)

def log(df: pd.DataFrame, dropinf: bool = True):
    if dropinf:
        return np.log(df).replace([np.inf, -np.inf], np.nan)
    return np.log(df)

def tsmean(df: pd.DataFrame, n: int = 20):
    return df.rolling(n).mean()


class Factor(q.Factor):

    def get_vwap(self, start: str = None, stop: str = None) -> pd.DataFrame:
        def _get(date: pd.Timestamp):
            p = fqtm.read("close", start=date, stop=date + pd.Timedelta(days=1))
            vol = fqtm.read("volume", start=date, stop=date + pd.Timedelta(days=1))
            w = vol / vol.sum()
            res = (p * w).sum() * adjfactor.loc[date]
            res.name = date
            return res
        
        start = start or pd.to_datetime('now').strftime(r"%Y-%m-%d")
        stop = stop or pd.to_datetime('now').strftime(r"%Y-%m-%d")
        trading_days = fqtd.get_trading_days(start, stop)
        adjfactor = fqtd.read("adjfactor", start=start, stop=stop)
        stsus = fqtd.read("st, suspended", start=start, stop=stop)
        nontradable = stsus["st"] | stsus["suspended"]
        nontradable = nontradable.unstack(level=fqtd._code_level).fillna(True)
        adjfactor = adjfactor.where(~nontradable)
        return pd.concat(Parallel(n_jobs=-1, backend='loky')(delayed(_get)
            (date) for date in tqdm(list(trading_days))), axis=1).T.loc[start:stop]


fqtd = q.Factor("./data/quotes-day", code_level="order_book_id", date_level="date")
fqtm = q.Factor("./data/quotes-min", code_level="order_book_id", date_level="datetime")

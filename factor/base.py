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


class BaseFactor(q.Factor):

    def get(self, name: str, start: str = None, stop: str = None, n_jobs: int = -1):
        start = start or pd.to_datetime('now').strftime(r"%Y-%m-%d")
        stop = stop or pd.to_datetime('now').strftime(r"%Y-%m-%d")
        trading_days = fqtd.get_trading_days(start, stop)
        result = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(getattr(self, "get_" + name))(date) for date in tqdm(list(trading_days))
        )
        if isinstance(result[0], pd.Series):
            return pd.concat(result, axis=1).T.loc[start:stop]
        elif isinstance(result[0], pd.DataFrame):
            return pd.concat(result, axis=0).sort_index().loc(axis=0)[:, start:stop]


fqtd = q.Factor("./data/quotes-day", code_level="order_book_id", date_level="date")
fqtm = q.Factor("./data/quotes-min", code_level="order_book_id", date_level="datetime")
fcon = q.Factor("./data/stock-connect", code_level="order_book_id", date_level="date")

import numpy as np
import pandas as pd
from .base import (
    fqtd, fqtm, BaseFactor
)


class DeraPriceFactor(BaseFactor):

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


    def get_liquidity(self, date: pd.Timestamp) -> pd.DataFrame:
        stom = self.get_stom(date)
        stoq = self.get_stoq(date)
        stoa = self.get_stoa(date)
        res = stom + stoq + stoa
        res.name = date
        return res

    def get_stom(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = fqtd.get_trading_days_rollback(date, 21)
        volume = fqtd.read("volume", start=rollback, stop=date)
        shares = fqtd.read("circulation_a", start=rollback, stop=date)
        res = np.log(np.sum(volume/shares))
        return res*0.35
    
    def get_stoq(self, date: str | pd.Timestamp) -> pd.Series:
        current_date = pd.Timestamp(date)
        res = pd.Series()
        for _ in range(3):  
            stom = np.exp(self.get_stom(current_date))
            current_date -= pd.DateOffset(days=21)
            res = pd.concat([res, stom])
        res = res.groupby(res.index).sum()
        res = np.log(res)
        return res * 0.35 
    
    def get_stoa(self, date: str | pd.Timestamp) -> pd.Series:
        current_date = pd.Timestamp(date)
        res = pd.Series()
        for _ in range(12):  
            stom = np.exp(self.get_stom(current_date))
            current_date -= pd.DateOffset(days=21)
            res = pd.concat([res, stom])
        res = res.groupby(res.index).sum()
        res = np.log(res)
        return res * 0.3 


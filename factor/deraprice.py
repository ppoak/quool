import numpy as np
import pandas as pd
from .base import (
    fqtd, fqtm, fidxwgt, BaseFactor
)


class DeraPriceFactor(BaseFactor):
    def standardize(self, data: pd.Series ,date: pd.Timestamp) -> pd.Series:
        weight = fidxwgt.read('000001.XSHG',start=date, stop=date).loc[date]
        mean = np.sum(weight * data)
        std = data.std()
        res = (data - mean) / std
        return res

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
        stom = self.standardize(self.get_stom(date),date).fillna(0)
        stoq = self.standardize(self.get_stoq(date),date).fillna(0)
        stoa = self.standardize(self.get_stoa(date),date).fillna(0)
        res = stom*0.35 + stoq*0.35 + stoa*0.3 
        res.name = date
        return res

    def get_stom(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = fqtd.get_trading_days_rollback(date, 20)
        volume = fqtd.read("volume", start=rollback, stop=date)
        shares = fqtd.read("circulation_a", start=rollback, stop=date)
        res = np.log((volume/shares).sum(axis=0) + 1e-6)
        return res
    
    def get_stoq(self, date: str | pd.Timestamp) -> pd.Series:
        stom = np.exp(self.get_stom(date))
        for i in range(1, 3):
             rollback = fqtd.get_trading_days_rollback(date, 21*i)
             stom_2 = np.exp(self.get_stom(rollback))
             stom += stom_2
        res = np.log(stom/3 + 1e-6)
        return res  
    
    def get_stoa(self, date: str | pd.Timestamp) -> pd.Series:
        stom = np.exp(self.get_stom(date))
        for i in range(1, 12):
             rollback = fqtd.get_trading_days_rollback(date, 21*i)
             stom_2 = np.exp(self.get_stom(rollback))
             stom += stom_2
        res = np.log(stom/12 + 1e-6)
        return res 


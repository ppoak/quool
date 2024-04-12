import pandas as pd
import numpy as np
import statsmodels.api as sm
from .base import (
    fqtd, ffin, BaseFactor
)

class EvaluationFactor(BaseFactor):

    def get_barra_bp_ratio(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = fqtd.get_trading_days_rollback(date, rollback=252)
        trading_days = fqtd.get_trading_days(start=rollback, stop=date)
        equity = ffin.read('total_equity', start=rollback, stop=date)
        equity = equity.reindex(trading_days).ffill()
        equity = equity.loc[date]
        price = fqtd.read('close', start=date, stop=date).loc[date]
        adjfactor = fqtd.read('adjfactor', start=date, stop=date).loc[date]
        shares = fqtd.read('circulation_a', start=date, stop=date).loc[date]
        size = price * adjfactor * shares
        res = equity / size
        res.name = date
        return res
    
    def get_barra_earning_yield(self, date: str | pd.Timestamp) -> pd.Series:
        cetop = self.get_cetop_ratio(date)
        etop = self.get_etop_ratio(date)
        res = cetop + etop
        res.name = date
        return res
    
    def get_etop_ratio(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = fqtd.get_trading_days_rollback(date, rollback=252)
        eps = ffin.read('basic_earnings_per_share',start=rollback, stop=date).sum()
        price = fqtd.read('close', start=date, stop=date).loc[date]
        adjfactor = fqtd.read('adjfactor', start=date, stop=date).loc[date]
        price_adj = price * adjfactor
        res = eps / price_adj
        return res * 0.33
    
    def get_cetop_ratio(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = fqtd.get_trading_days_rollback(date, rollback=252)
        ce = ffin.read('cash_flow_from_operating_activities',start=rollback, stop=date).sum()
        price = fqtd.read('close', start=date, stop=date).loc[date]
        adjfactor = fqtd.read('adjfactor', start=date, stop=date).loc[date]
        shares = fqtd.read('circulation_a', start=date, stop=date).loc[date]
        size = price * adjfactor * shares
        res = ce / size
        return res * 0.64
    
    def get_egro(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = fqtd.get_trading_days_rollback(date, rollback=252*5)
        eps = ffin.read('basic_earnings_per_share',start=rollback, stop=date).ffill().bfill()
        X = sm.add_constant(np.arange(len(eps)))
        res = []
        for code in eps.columns:
            slopes = sm.OLS(eps[code], X).fit().params[1]
            res.append(slopes)
        res = pd.Series(res, index=eps.columns)
        res = res/ eps.mean()
        return res * 0.34
    
    def get_sgro(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = fqtd.get_trading_days_rollback(date, rollback=252*5)
        eps = ffin.read('revenue',start=rollback, stop=date).ffill().bfill()
        X = sm.add_constant(np.arange(len(eps)))
        res = []
        for code in eps.columns:
            slopes = sm.OLS(eps[code], X).fit().params[1]
            res.append(slopes)
        res = pd.Series(res, index=eps.columns)
        res = res/ eps.mean()
        return res * 0.65

    def get_barra_growth(self, date: str | pd.Timestamp) -> pd.Series:
        sgro = self.get_sgro(date)
        egro = self.get_egro(date)
        res = sgro + egro
        res.name = date
        return res

import pandas as pd
import numpy as np
import statsmodels.api as sm
from .base import (
    fqtd, ffin,fidxwgt, BaseFactor
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

    def get_mlev(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = fqtd.get_trading_days_rollback(date, rollback=252)
        trading_days = fqtd.get_trading_days(start=rollback, stop=date)
        pe = ffin.read('equity_preferred_stock', start=rollback, stop=date)
        pe = pe.reindex(trading_days).ffill()
        pe = pe.loc[date].fillna(0)
        ld = ffin.read('non_current_liabilities', start=rollback, stop=date)
        ld = ld.reindex(trading_days).ffill()
        ld = ld.loc[date].fillna(0)
        price = fqtd.read('close', start=date, stop=date).loc[date]
        adjfactor = fqtd.read('adjfactor', start=date, stop=date).loc[date]
        shares = fqtd.read('circulation_a', start=date, stop=date).loc[date]
        me = price * adjfactor * shares
        res = (me + ld + pe)/ me
        return res * 0.38

    def get_dtoa(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = fqtd.get_trading_days_rollback(date, rollback=252)
        trading_days = fqtd.get_trading_days(start=rollback, stop=date)
        ta = ffin.read('total_assets', start=rollback, stop=date)
        ta = ta.reindex(trading_days).ffill()
        ta = ta.loc[date].fillna(0)
        td = ffin.read('total_liabilities', start=rollback, stop=date)
        td = td.reindex(trading_days).ffill()
        td = td.loc[date].fillna(0)
        res = td / ta
        return res * 0.35
    
    def get_blev(self, date: str | pd.Timestamp) -> pd.Series:
        rollback = fqtd.get_trading_days_rollback(date, rollback=252)
        trading_days = fqtd.get_trading_days(start=rollback, stop=date)
        pe = ffin.read('equity_preferred_stock', start=rollback, stop=date)
        pe = pe.reindex(trading_days).ffill()
        pe = pe.loc[date].fillna(0)
        ld = ffin.read('non_current_liabilities', start=rollback, stop=date)
        ld = ld.reindex(trading_days).ffill()
        ld = ld.loc[date].fillna(0)
        total_equity = ffin.read('non_current_liabilities', start=rollback, stop=date)
        total_equity = total_equity.reindex(trading_days).ffill()
        total_equity = total_equity.loc[date].fillna(0)
        shares = fqtd.read('circulation_a', start=date, stop=date).loc[date]
        be = (total_equity - pe) / shares
        res = (be + ld + pe) / be
        return res * 0.27

    def get_barra_leverage(self, date: str | pd.Timestamp) -> pd.Series:
        mlev = self.standardize(self.get_mlev(date),date).fillna(0)
        blev = self.standardize(self.get_blev(date),date).fillna(0)
        dtoa = self.standardize(self.get_dtoa(date),date).fillna(0)
        res = mlev + blev + dtoa
        return res
    
    def standardize(self, data: pd.Series ,date: pd.Timestamp) -> pd.Series:
        weight = fidxwgt.read('000001.XSHG',start=date, stop=date).loc[date]
        mean = np.sum(weight * data)
        std = data.std()
        res = (data - mean) / std
        return res
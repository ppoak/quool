import numpy as np
import pandas as pd
import statsmodels.api as sm
from base import (
    fqtd, fina, BaseFactor
)


class MarketSizeFactor(BaseFactor):

    def get_barra_log_marketcap(self, date: str):
        shares = fqtd.read("circulation_a", start=date, stop=date)
        price = fqtd.read("close", start=date, stop=date)
        adjfactor = fqtd.read("adjfactor", start=date, stop=date)
        res = np.log(shares * price * adjfactor).loc[date]
        return res

    def get_NLsize(self, date: str):
        log_marketcap = self.get_barra_log_marketcap(date)
        Y = (log_marketcap ** 3)
        X = sm.add_constant(log_marketcap.dropna()) 
        model = sm.OLS(Y, X)
        res = model.fit()
        res = res.resid
        res = self.winsorize(res)
        res = (res - res.mean()) / res.std()
        res.name = date
        return res
    
    def winsorize(self, data):
        mean = data.mean()
        std = data.std()
        lower_bound = mean - 3 * std
        upper_bound = mean + 3 * std
        # lower_bound = np.percentile(data, 2.5)
        # upper_bound = np.percentile(data, 97.5)
        res = np.clip(data, lower_bound, upper_bound)
        return res
    
    def get_book_to_price(self, date: str):
        bv = fina.read('total_assets,total_liabilities',start=date, stop=date)
        bv['book_value'] = bv['total_assets'] - bv['total_liabilities']
        bv = bv.reset_index().set_index('order_book_id')['book_value']
        log_marketcap = self.get_barra_log_marketcap(date)
        res = bv / log_marketcap
        res.name = date
        return res
    
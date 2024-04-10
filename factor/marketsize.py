import numpy as np
import pandas as pd
import statsmodels.api as sm
from base import (
    fqtd, BaseFactor
)


class MarketSizeFactor(BaseFactor):

    def get_barra_log_marketcap(self, date: str):
        shares = fqtd.read("circulation_a", start=date, stop=date)
        price = fqtd.read("close", start=date, stop=date)
        adjfactor = fqtd.read("adjfactor", start=date, stop=date)
        res = np.log(shares * price * adjfactor).loc[date]
        return res

    def get_NLsize(self, date: str):
        log_marketcap = self.get_barra_log_marketcap(date).dropna()
        Y = (log_marketcap ** 3)
        X = sm.add_constant(log_marketcap) 
        model = sm.OLS(Y, X)
        res = model.fit()
        res = res.resid
        res = self.winsorize(res)
        res = (res - res.mean()) / res.std()
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
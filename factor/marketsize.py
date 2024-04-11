import numpy as np
import pandas as pd
import statsmodels.api as sm
from .base import (
    fqtd, BaseFactor
)


class MarketSizeFactor(BaseFactor):

    def get_barra_log_marketcap(self, date: str | pd.Timestamp) -> pd.Series:
        shares = fqtd.read("circulation_a", start=date, stop=date)
        price = fqtd.read("close", start=date, stop=date)
        adjfactor = fqtd.read("adjfactor", start=date, stop=date)
        res = np.log(shares * price * adjfactor).loc[date]
        return res

    def get_barra_nonlinear_size(self, date: str | pd.Timestamp) -> pd.Series:
        marketcap = self.get_barra_log_marketcap(date)
        y = (marketcap ** 3).dropna()
        x = sm.add_constant(marketcap).dropna()
        model = sm.OLS(y, x)
        res = model.fit()
        res = res.resid
        mean = res.mean()
        std = res.std()
        res = res.clip(mean - 3 * std, mean + 3 * std)
        res = (res - res.mean()) / res.std()
        res.name = date
        return res
        
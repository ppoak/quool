import numpy as np
import pandas as pd
from base import (
    fqtd, fidx_c, BaseFactor
)


class BetaFactor(BaseFactor):
    def get_returns(self, price):
        return price.pct_change()

    def calculate_beta(self, stock_returns, market_returns):
        res = {}
        var = np.var(market_returns)
        for code in stock_returns.columns:
            covr = np.cov(stock_returns[code],market_returns)[0][1]
            result = covr/var
            res[code] = result
        return pd.Series(res)

    def get_beta_factor(self, date: str):
        rollback = fqtd.get_trading_days_rollback(date, 252)
        prices = fqtd.read("close", start=rollback, stop=date)
        market_prices = fidx_c.read('close',start=rollback, stop=date).loc[:,'000001.XSHG']
        stock_returns = self.get_returns(prices).tail(252).ewm(halflife=63).mean()
        market_returns = self.get_returns(market_prices).tail(252).ewm(halflife=63).mean()
        res = self.calculate_beta(stock_returns, market_returns)
        res.index.name = 'order_book_id'
        return res

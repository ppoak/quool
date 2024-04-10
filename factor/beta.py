import numpy as np
import pandas as pd
from base import (
    fqtd, fidx_c, BaseFactor
)


class BetaFactor(BaseFactor):
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
        adjfactor = fqtd.read("adjfactor", start=rollback, stop=date)
        prices_adj = prices * adjfactor
        market_prices = fidx_c.read('close',start=rollback, stop=date).loc[:,'000001.XSHG']
        stock_returns = prices_adj.pct_change().tail(252).ewm(halflife=63).mean()
        market_returns = market_prices.pct_change().tail(252).ewm(halflife=63).mean()
        res = self.calculate_beta(stock_returns, market_returns)
        res.index.name = 'order_book_id'
        return res

import numpy as np
import pandas as pd
from base import (
    fqtd, fidx_c, BaseFactor
)


class BetaFactor(BaseFactor):
    def get_returns(self, price):
        return price.pct_change()

    def calculate_beta(self, stock_returns, market_returns):
        beta_values = {}
        var = np.var(market_returns)
        for stock in stock_returns.columns:
            covr = np.cov(stock_returns[stock],market_returns)[0][1]
            result = covr/var
            beta_values[stock] = result
        return pd.Series(beta_values)

    def get_beta_factor(self, date: str):
        rollback = fqtd.get_trading_days_rollback(date, 252)
        stock_prices = fqtd.read("close", start=rollback, stop=date)
        market_prices = fidx_c.read('close',start=rollback, stop=date).loc[:,'000001.XSHG']
        
        stock_returns = self.get_returns(stock_prices).tail(252).ewm(halflife=63).mean()
        market_returns = self.get_returns(market_prices).tail(252).ewm(halflife=63).mean()

        beta = self.calculate_beta(stock_returns, market_returns)
        beta.index.name = 'order_book_id'
        return beta

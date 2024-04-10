import numpy as np
import pandas as pd
import statsmodels.api as sm
from base import (
    fqtd, fqtm, fidx_c, BaseFactor
)


class VolatileFactor(BaseFactor):

    def get_information_distribution_uniformity(self, date: str):
        rollback = fqtd.get_trading_days_rollback(date, 20)
        price = fqtm.read("close", start=rollback, stop=date + pd.Timedelta(days=1))
        ret = price.groupby(price.index.date).pct_change(fill_method=None)
        std = ret.groupby(ret.index.date).std()
        res = std.std() / std.mean()
        res.name = date
        return res

    def get_dastd(self, date: str):
        rollback = fqtd.get_trading_days_rollback(date, 252)
        prices = fqtd.read("close", start=rollback, stop=date)
        adjfactor = fqtd.read("adjfactor", start=rollback, stop=date)
        prices_adj = prices * adjfactor
        market_prices = fidx_c.read('close',start=rollback, stop=date).loc[:,'000001.XSHG']
        stock_ret = prices_adj.pct_change()
        market_ret = market_prices.pct_change()
        excess_ret = stock_ret.subtract(market_ret, axis=0)
        res = np.sqrt(np.sum(((excess_ret - excess_ret.mean()) ** 2).ewm(halflife=42).mean()))
        return res * 0.74
    
    def get_cmra(self, date: str):
        rollback = fqtd.get_trading_days_rollback(date, 252)
        prices = fqtd.read("close", start=rollback, stop=date)
        adjfactor = fqtd.read("adjfactor", start=rollback, stop=date)
        prices_adj = prices * adjfactor
        market_prices = fidx_c.read('close',start=rollback, stop=date).loc[:,'000001.XSHG']
        stock_ret = np.log(1 + prices_adj.pct_change())
        market_ret = np.log(1 + market_prices.pct_change())
        zt = stock_ret.subtract(market_ret, axis=0).rolling(window=21).sum()
        res = zt.max() - zt.min()
        return res * 0.16
    
    def get_hsigma(self, date: str):
        rollback = fqtd.get_trading_days_rollback(date, 252)
        prices = fqtd.read("close", start=rollback, stop=date)
        adjfactor = fqtd.read("adjfactor", start=rollback, stop=date)
        prices_adj = prices * adjfactor
        market_prices = fidx_c.read('close',start=rollback, stop=date).loc[:,'000001.XSHG']
        stock_returns = prices_adj.pct_change().tail(252).ewm(halflife=63).mean()
        market_returns = market_prices.pct_change().tail(252).ewm(halflife=63).mean()
        res = self.calculate_resid(stock_returns, market_returns).std()
        res.index.name = 'order_book_id'
        return res * 0.1

    def calculate_resid(self, stock_returns, market_returns):
        X = sm.add_constant(market_returns)
        res = pd.DataFrame()
        list = []
        for code in stock_returns.columns:
            list.append(sm.OLS(stock_returns[code], X).fit().resid)
        res = pd.concat(list, axis=1)
        res.columns = stock_returns.columns
        return res


    def get_residual_volatility(self, date: str):
        hsigma = self.get_hsigma(date).fillna(0)
        cmra = self.get_cmra(date).fillna(0)
        dastd = self.get_dastd(date).fillna(0)
        res = dastd + cmra + hsigma
        res.name = date
        return res
    
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
        res.name = date
        return res
import numpy as np
import pandas as pd
import statsmodels.api as sm
from base import (
    fqtd,fidx_c, BaseFactor
)

    
class Volatility(BaseFactor):
    def get_dastd(self, date: str):
        rollback = fqtd.get_trading_days_rollback(date, 252)
        price = fqtd.read("close", start=rollback, stop=date)
        market_prices = fidx_c.read('close',start=rollback, stop=date).loc[:,'000001.XSHG']
        stock_ret = price.pct_change()
        market_ret = market_prices.pct_change()
        excess_ret = stock_ret.subtract(market_ret, axis=0)
        res = np.sqrt(np.sum(((excess_ret - excess_ret.mean()) ** 2).ewm(halflife=42).mean()))
        return res * 0.74
    
    def get_cmra(self, date: str):
        rollback = fqtd.get_trading_days_rollback(date, 252)
        price = fqtd.read("close", start=rollback, stop=date)
        market_prices = fidx_c.read('close',start=rollback, stop=date).loc[:,'000001.XSHG']
        stock_ret = np.log(1 + price.pct_change())
        market_ret = np.log(1 + market_prices.pct_change())
        zt = stock_ret.subtract(market_ret, axis=0).rolling(window=21).sum()
        res = zt.max() - zt.min()
        return res * 0.16
    
    def get_hsigma(self, date: str):
        rollback = fqtd.get_trading_days_rollback(date, 252)
        prices = fqtd.read("close", start=rollback, stop=date)
        market_prices = fidx_c.read('close',start=rollback, stop=date).loc[:,'000001.XSHG']
        stock_returns = prices.pct_change().tail(252).ewm(halflife=63).mean()
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


    def get_volatility(self, date: str):
        date = '20040104'
        hsigma = self.get_hsigma(date).fillna(0)
        cmra = self.get_cmra(date).fillna(0)
        dastd = self.get_dastd(date).fillna(0)
        res = dastd + cmra + hsigma
        return res



import logging
import numpy as np
import pandas as pd
import backtrader as bt
from .util import Logger


def strategy(
    weight: pd.DataFrame, 
    price: pd.DataFrame, 
    delay: int = 1,
    side: str = 'both',
    commission: float = 0.005,
) -> dict[str, pd.Series]:
    # compute turnover and commission
    delta = weight - weight.shift(1).fillna(0)
    if side == "both":
        turnover = delta.abs().sum(axis=1) / 2
    elif side == "long":
        turnover = delta.where(delta > 0).abs().sum(axis=1)
    elif side == "short":
        turnover = delta.where(delta < 0).abs().sum(axis=1)
    turnover = turnover.shift(delay).fillna(0)
    commission *= turnover

    # compute the daily return
    returns = price.pct_change(fill_method=None).fillna(0)
    returns = (weight.shift(delay + 1) * returns).sum(axis=1)
    returns -= commission
    returns.name = 'return'
    turnover.name = 'turnover'

    return {'returns': returns, 'turnover': turnover}

def evaluate(
    returns: pd.Series, 
    turnover: pd.Series = None,    
    benchmark: pd.Series = None,
) -> pd.Series:
    value = (returns + 1).cumprod()
    if benchmark is not None:
        benchmark = benchmark.squeeze()
        benchmark_returns = benchmark.pct_change(fill_method=None).fillna(0)
    
    # evaluation indicators
    evaluation = pd.Series(name='evaluation')
    evaluation['total_return(%)'] = (value.iloc[-1] - 1) * 100
    evaluation['annual_return(%)'] = (value.iloc[-1] ** (252 / value.shape[0]) - 1) * 100
    evaluation['annual_volatility(%)'] = (returns.std() * np.sqrt(252)) * 100
    cumdrawdown = -(value / value.cummax() - 1)
    maxdate = cumdrawdown.idxmax()
    startdate = cumdrawdown.loc[:maxdate].idxmin()
    evaluation['max_drawdown(%)'] = (cumdrawdown.max()) * 100
    evaluation['max_drawdown_period(days)'] = maxdate - startdate
    evaluation['daily_turnover(%)'] = turnover.mean() * 100 if turnover is not None else np.nan
    evaluation['sharpe_ratio'] = returns.mean() / returns.std()
    evaluation['sortino_ratio'] = returns.mean() / returns[returns < 0].std()
    evaluation['calmar_ratio'] = evaluation['annual_return(%)'] / evaluation['max_drawdown(%)']
    if benchmark is not None:
        exreturns = returns - benchmark_returns
        exvalue = (1 + exreturns).cumprod()
        evaluation['total_exreturn(%)'] = (exvalue.iloc[-1] - 1) * 100
        evaluation['annual_exreturn(%)'] = (exvalue.iloc[-1] ** (252 / exvalue.shape[0]) - 1) * 100
        evaluation['annual_exvolatility(%)'] = (exreturns.std() * np.sqrt(252)) * 100
        evaluation['beta'] = returns.cov(benchmark_returns) / benchmark_returns.var()
        evaluation['alpha(%)'] = (returns.mean() - (evaluation['beta'] * (benchmark_returns.mean()))) * 100
        evaluation['treynor_ratio(%)'] = (exreturns.mean() / evaluation['beta']) * 100
        evaluation['information_ratio'] = exreturns.mean() / benchmark_returns.std()
    
    return evaluation

            
class Strategy(bt.Strategy):

    params = (
        ("minstake", 1), 
        ("minshare", 100), 
        ("splitfactor", "splitfactor"),
        ("divfactor", "divfactor"),
    )
    logger = Logger("QuoolStrategy", level=logging.DEBUG, display_time=False, display_name=False)

    def log(self, text: str, level: int = logging.DEBUG, datetime: pd.Timestamp = None):
        datetime = datetime or self.data.datetime.date(0)
        self.logger.log(level=level, msg=f'[{datetime}] {text}')

    def split_dividend(self):
        if not (self.params.divfactor or self.params.splitfactor):
            raise ValueError("at least one factor should be provided")
        
        for data in self.datas:
            divfactor = getattr(data, self.params.divfactor)[0] \
                if self.params.divfactor else np.nan
            splitfactor = getattr(data, self.params.splitfactor)[0] \
                if self.params.splitfactor else np.nan
            position = self.getposition(data)
            size = position.size
            # hold the stock which has dividend
            if size and not np.isnan(divfactor):
                dividend = divfactor * size
                self.broker.add_cash(dividend)
            # hold the stock which has split
            if size and not np.isnan(splitfactor):
                splitsize = int(size * splitfactor)
                position.update(size=splitsize, price=data.close[0])
    
    def resize(self, size: int):
        minstake = self.params._getkwargs().get("minstake", 1)
        if size is not None:
            size = max(minstake, (size // minstake) * minstake)
        return size

    def buy(
        self, data=None, size=None, price=None, plimit=None, 
        exectype=None, valid=None, tradeid=0, oco=None, trailamount=None, 
        trailpercent=None, parent=None, transmit=True, **kwargs
    ):
        minsize = self.params._getkwargs().get("minsize", 1)
        size = self.resize(size)
        if size is not None and size < minsize:
            self.logger.warning(f'{data._name} buy {size} < {minsize} failed')
            size = 0
        return super().buy(data, size, price, plimit, 
            exectype, valid, tradeid, oco, trailamount, 
            trailpercent, parent, transmit, **kwargs)
    
    def sell(
        self, data=None, size=None, price=None, plimit=None, 
        exectype=None, valid=None, tradeid=0, oco=None, trailamount=None, 
        trailpercent=None, parent=None, transmit=True, **kwargs
    ):
        size = self.resize(size)
        return super().sell(data, size, price, plimit, 
            exectype, valid, tradeid, oco, trailamount, 
            trailpercent, parent, transmit, **kwargs)

    def notify_order(self, order: bt.Order):
        # order possible status:
        # 'Created'、'Submitted'、'Accepted'、'Partial'、'Completed'、
        # 'Canceled'、'Expired'、'Margin'、'Rejected'
        # broker submitted or accepted order do nothing
        if order.status in [order.Submitted, order.Accepted, order.Created]:
            return

        # broker completed order, just hint
        elif order.status in [order.Completed]:
            self.log(f'{order.data._name} ref.{order.ref} order {order.executed.size} at {order.executed.price:.2f}')

        elif order.status in [order.Canceled, order.Margin, order.Rejected, order.Expired]:
            self.log(f'{order.data._name} ref.{order.ref} canceled, margin, rejected or expired')

    def notify_trade(self, trade):
        if not trade.isclosed:
            # trade not closed, skip
            return
        # else, log it
        self.log(f'{trade.data._name} gross {trade.pnl:.2f}, net {trade.pnlcomm:.2f}')


class Indicator(bt.Indicator):
    logger = Logger('QuoolIndicator', display_time=False, display_name=False)
    
    def log(self, text: str, level: int = logging.INFO, datetime: pd.Timestamp = None):
        datetime = datetime or self.data.datetime.date(0)
        self.logger.log(level=level, msg=f'[{datetime}]: {text}')


class Analyzer(bt.Analyzer):
    logger = Logger('QuoolAnalyzer', display_time=False, display_name=False)

    def log(self, text: str, level: int = logging.INFO, datetime: pd.Timestamp = None):
        datetime = datetime or self.data.datetime.date(0)
        self.logger.log(level=level, msg=f'[{datetime}]: {text}')


class Observer(bt.Observer):
    logger = Logger('QuoolObserver', display_time=False, display_name=False)

    def log(self, text: str, level: int = logging.INFO, datetime: pd.Timestamp = None):
        datetime = datetime or self.data.datetime.date(0)
        self.logger.log(level=level, msg=f'[{datetime}]: {text}')


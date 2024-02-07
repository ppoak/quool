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
    commission *= turnover

    # compute the daily return
    price = price.shift(-delay).loc[weight.index]
    returns = price.shift(-1) / price - 1
    returns = (weight * returns).sum(axis=1)
    returns -= commission
    value = (1 + returns).cumprod()
    value.name = 'value'
    turnover.name = 'turnover'

    return {'value': value, 'turnover': turnover}

def evaluate(
    value: pd.Series, 
    turnover: pd.Series = None,    
    benchmark: pd.Series = None,
) -> pd.Series:
    returns = value.pct_change().fillna(0)
    if benchmark is not None:
        benchmark = benchmark.squeeze()
        benchmark_returns = benchmark.pct_change(fill_method=None).fillna(0)
    
    # evaluation indicators
    evaluation = pd.Series(name='evaluation')
    evaluation['total_return(%)'] = (value.iloc[-1] / value.iloc[0] - 1) * 100
    evaluation['annual_return(%)'] = ((evaluation['total_return(%)'] / 100 + 1) ** (252 / value.shape[0]) - 1) * 100
    evaluation['annual_volatility(%)'] = (returns.std() * np.sqrt(252)) * 100
    down_volatility = (returns[returns < 0].std() * np.sqrt(252)) * 100
    cumdrawdown = -(value / value.cummax() - 1)
    maxdate = cumdrawdown.idxmax()
    startdate = cumdrawdown.loc[:maxdate][cumdrawdown.loc[:maxdate] == 0].index[-1]
    evaluation['max_drawdown(%)'] = (cumdrawdown.max()) * 100
    evaluation['max_drawdown_period(days)'] = maxdate - startdate
    evaluation['max_drawdown_start'] = startdate
    evaluation['max_drawdown_stop'] = maxdate
    evaluation['daily_turnover(%)'] = turnover.mean() * 100 if turnover is not None else np.nan
    evaluation['sharpe_ratio'] = evaluation['annual_return(%)'] / evaluation['annual_volatility(%)'] \
        if evaluation['annual_volatility(%)'] != 0 else np.nan
    evaluation['sortino_ratio'] = evaluation['annual_return(%)'] / down_volatility \
        if down_volatility != 0 else np.nan
    evaluation['calmar_ratio'] = evaluation['annual_return(%)'] / evaluation['max_drawdown(%)'] \
        if evaluation['max_drawdown(%)'] != 0 else np.nan
    if benchmark is not None:
        exreturns = returns - benchmark_returns
        benchmark_volatility = (benchmark_returns.std() * np.sqrt(252)) * 100
        exvalue = (1 + exreturns).cumprod()
        evaluation['total_exreturn(%)'] = (exvalue.iloc[-1] - exvalue.iloc[0]) * 100
        evaluation['annual_exreturn(%)'] = ((evaluation['total_exreturn(%)'] / 100 + 1) ** (252 / exvalue.shape[0]) - 1) * 100
        evaluation['annual_exvolatility(%)'] = (exreturns.std() * np.sqrt(252)) * 100
        evaluation['beta'] = returns.cov(benchmark_returns) / benchmark_returns.var()
        evaluation['alpha(%)'] = (returns.mean() - (evaluation['beta'] * (benchmark_returns.mean()))) * 100
        evaluation['treynor_ratio(%)'] = (evaluation['annual_exreturn(%)'] / evaluation['beta'])
        evaluation['information_ratio'] = evaluation['annual_exreturn(%)'] / benchmark_volatility \
            if benchmark_volatility != 0 else np.nan
    
    return evaluation

            
class Strategy(bt.Strategy):

    logger = Logger("QuoolStrategy", level=logging.DEBUG, display_time=False, display_name=False)

    def log(self, text: str, level: int = logging.DEBUG, datetime: pd.Timestamp = None):
        datetime = datetime or self.data.datetime.date(0)
        self.logger.log(level=level, msg=f'[{datetime}] {text}')
    
    def notify_order(self, order: bt.Order):
        # order possible status:
        # 'Created'、'Submitted'、'Accepted'、'Partial'、'Completed'、
        # 'Canceled'、'Expired'、'Margin'、'Rejected'
        # broker submitted or accepted order do nothing
        if order.status in [order.Submitted, order.Accepted, order.Created]:
            return

        # broker completed order, just hint
        elif order.status in [order.Completed]:
            self.log(f'{order.data._name} ref.{order.ref} order {order.executed.size:.0f} at {order.executed.price:.2f}')

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


class Broker(bt.BackBroker):

    params = (
        ("minstake", 1), 
        ("minshare", 100), 
        ("divfactor", "divfactor"), 
        ("splitfactor", "splitfactor"),
    )
    logger = Logger("QuoolBroker", level=logging.DEBUG, display_time=False, display_name=False)

    def split_dividend(self):
        for data, pos in self.positions.items():
            divline = getattr(data, self.params.divfactor, None)
            splitline = getattr(data, self.params.splitfactor, None)
            divfactor = divline[0] if divline else np.nan
            splitfactor = splitline[0] if splitline else np.nan
            size = pos.size
            # hold the stock which has dividend
            if size and not np.isnan(divfactor):
                dividend = divfactor * size
                self.logger.debug(f"[{data.datetime.date(0)}] {data._name} dividend cash: {dividend:.2f}")
                self.add_cash(dividend)
            # hold the stock which has split
            if size and not np.isnan(splitfactor):
                splitsize = int(size * splitfactor)
                self.logger.debug(f"[{data.datetime.date(0)}] {data._name} split size: {splitsize:.0f}")
                pos.update(size=splitsize, price=data.close[0])
    
    def resize(self, size: int):
        minstake = self.params._getkwargs().get("minstake", 1)
        if size is not None:
            size = max(minstake, (size // minstake) * minstake)
        return size

    def buy(
        self, owner, data, size, price=None, plimit=None,
        exectype=None, valid=None, tradeid=0, oco=None, 
        trailamount=None, trailpercent=None, parent=None, 
        transmit=True, histnotify=False, _checksubmit=True, **kwargs
    ):
        minshare = self.params._getkwargs().get("minshare", 1)
        size = self.resize(size)
        if size is not None and size < minshare:
            self.logger.warning(f'[{data.datetime.date(0)}] {data._name} buy {size} < {minshare} failed')
            size = 0
        return super().buy(owner, data, size, price, plimit,
            exectype, valid, tradeid, oco, trailamount, trailpercent, 
            parent, transmit, histnotify, _checksubmit, **kwargs)
    
    def sell(
        self, owner, data, size, price=None, plimit=None,
        exectype=None, valid=None, tradeid=0, oco=None, 
        trailamount=None, trailpercent=None, parent=None, 
        transmit=True, histnotify=False, _checksubmit=True, **kwargs
    ):
        size = self.resize(size)
        return super().sell(owner, data, size, price, plimit,
            exectype, valid, tradeid, oco, trailamount, trailpercent, 
            parent, transmit, histnotify, _checksubmit, **kwargs)
    
    def next(self):
        self.split_dividend()
        return super().next()


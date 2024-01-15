import logging
import numpy as np
import pandas as pd
import backtrader as bt
from .util import Logger


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
                orig_value = self.broker.get_value()
                orig_cash = self.broker.get_cash()
                dividend = divfactor * size
                self.broker.set_cash(orig_cash + dividend)
                self.broker._value = orig_value + dividend
            # hold the stock which has split
            if size and not np.isnan(splitfactor):
                splitsize = int(size * splitfactor)
                splitvalue = splitsize * data.close[0]
                orig_value = size * position.price
                position.set(size=size + splitsize, 
                    price=(orig_value + splitvalue) / (size + splitsize))
    
    def resize(self, size: int):
        minstake = self.params._getkwargs().get("minstake", 1)
        minshare = self.params._getkwargs().get("minshare", 1)
        if size is not None:
            size = max(minstake, (size // minstake) * minstake)
        if size is not None and size < minshare:
            raise ValueError(f"order size {size} is smaller than {minshare}.")
        return size

    def buy(
        self, data=None, size=None, price=None, plimit=None, 
        exectype=None, valid=None, tradeid=0, oco=None, trailamount=None, 
        trailpercent=None, parent=None, transmit=True, **kwargs
    ):
        size = self.resize(size)
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
        """Logging function"""
        datetime = datetime or self.data.datetime.date(0)
        self.logger.log(level=level, msg=f'[{datetime}]: {text}')


class Analyzer(bt.Analyzer):
    logger = Logger('QuoolAnalyzer', display_time=False, display_name=False)

    def log(self, text: str, level: int = logging.INFO, datetime: pd.Timestamp = None):
        """Logging function"""
        datetime = datetime or self.data.datetime.date(0)
        self.logger.log(level=level, msg=f'[{datetime}]: {text}')


class Observer(bt.Observer):
    logger = Logger('QuoolObserver', display_time=False, display_name=False)

    def log(self, text: str, level: int = logging.INFO, datetime: pd.Timestamp = None):
        """Logging function"""
        datetime = datetime or self.data.datetime.date(0)
        self.logger.log(level=level, msg=f'[{datetime}]: {text}')


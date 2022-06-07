import datetime
import pandas as pd
import numpy as np
import backtrader as bt
from ..tools import *

class BackTesterError(FrameWorkError):
    pass

@pd.api.extensions.register_dataframe_accessor("relocator")
@pd.api.extensions.register_series_accessor("relocator")
class Relocator(Worker):

    def profit(self, forward: pd.Series = None, weight_col: str = None, forward_col: str = None):
        '''calculate profit from weight and forward
        ---------------------------------------------

        weight_col: str, the column name of weight
        forward_col: str, the column name of forward
        '''
        if self.hint == Worker.TS:
            raise BackTesterError('profit', 'Please transform your data into multiindex data')
        
        elif self.hint == Worker.CS:
            raise BackTesterError('profit', 'We cannot calculate the profit by cross section data')

        if self.is_frame:
            if weight_col is None or forward_col is None:
                raise BackTesterError('profit', 'Please specify the weight and forward column')
            return self.data.groupby(level=0).apply(lambda x:
                (x.loc[:, weight_col] * x.loc[:, forward_col]).sum()
                / x.loc[:, weight_col].sum()
            )
        
        else:
            if forward is None:
                raise BackTesterError('profit', 'Please specify the forward return')
            else:
                self.data.name = self.data.name or 'forward'
                forward.name = forward.name or 'weight'
                data = pd.concat([forward, self.data], axis=1)
                return data.groupby(level=0).apply(
                    lambda x: (x.iloc[:, 0] * x.iloc[:, -1]).sum() / x.iloc[:, -1].sum()
                    if not x.iloc[:, -1].sum() == 0 else np.nan)
        
    def turnover(self, side: str = 'both'):
        '''calculate turnover
        ---------------------

        side: str, choice between "buy", "short" or "both"
        '''
        if self.hint == Worker.TS:
            raise BackTesterError('turnover', 'Please transform your data into multiindex data')
        
        elif self.hint == Worker.CS:
            raise BackTesterError('turnover', 'We cannot calculate the turnover by cross section data')

        datetime_index = self.data.index.get_level_values(0).unique()
        ret = pd.Series(index=datetime_index, dtype='float64')
        ret.loc[datetime_index[0]] = 1
        for i, d in enumerate(datetime_index[1:]):
            delta_frame = pd.concat([self.data.loc[d], self.data.loc[datetime_index[i]]], 
                axis=1, join='outer').fillna(0)
            delta = delta_frame.iloc[:, 0] - delta_frame.iloc[:, -1]
            if side == 'both':
                delta = delta.abs().sum() / self.data.loc[datetime_index[i]].abs().sum()
            if side == 'buy':
                delta = delta[delta > 0].sum() / self.data.loc[datetime_index[i]].abs().sum()
            if side == 'sell':
                delta = -delta[delta < 0].sum() / self.data.loc[datetime_index[i]].abs().sum()
            ret.loc[d] = delta
        return ret


class Strategy(bt.Strategy):

    def log(self, text: str, datetime: datetime.datetime = None, hint: str = 'INFO'):
        '''Logging function'''
        datetime = datetime or self.data.datetime.date(0)
        datetime = time2str(datetime)
        print(f'[{hint}] {datetime}: {text}')

    def notify_order(self, order: bt.Order):
        '''order notification'''
        # order possible status:
        # 'Created'、'Submitted'、'Accepted'、'Partial'、'Completed'、
        # 'Canceled'、'Expired'、'Margin'、'Rejected'
        # broker submitted or accepted order do nothing
        if order.status in [order.Submitted, order.Accepted, order.Created]:
            return

        # broker completed order, just hint
        elif order.status in [order.Completed]:
            self.log(f'Trade <{order.executed.size}> at <{order.executed.price:.2f}>')
            # record current bar number
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected, order.Expired]:
            self.log('Order canceled, margin, rejected or expired', hint='warn')

        # except the submitted, accepted, and created status,
        # other order status should reset order variable
        self.order = None

    def notify_trade(self, trade):
        '''trade notification'''
        if not trade.isclosed:
            # trade not closed, skip
            return
        # else, log it
        self.log(f'Gross Profit: {trade.pnl:.2f}, Net Profit {trade.pnlcomm:.2f}')

class TradeTable(bt.Analyzer):

    def __init__(self):
        self.trades = []
        self.cumprofit = 0.0

    def notify_trade(self, trade):
        if trade.isclosed:
            brokervalue = self.strategy.broker.getvalue()

            dir = 'short'
            if trade.history[0].event.size > 0: dir = 'long'

            pricein = trade.history[len(trade.history)-1].status.price
            priceout = trade.history[len(trade.history)-1].event.price
            datein = bt.num2date(trade.history[0].status.dt)
            dateout = bt.num2date(trade.history[len(trade.history)-1].status.dt)
            if trade.data._timeframe >= bt.TimeFrame.Days:
                datein = datein.date()
                dateout = dateout.date()

            pcntchange = 100 * priceout / pricein - 100
            pnl = trade.history[len(trade.history)-1].status.pnlcomm
            pnlpcnt = 100 * pnl / brokervalue
            barlen = trade.history[len(trade.history)-1].status.barlen
            pbar = pnl / barlen
            self.cumprofit += pnl

            size = value = 0.0
            for record in trade.history:
                if abs(size) < abs(record.status.size):
                    size = record.status.size
                    value = record.status.value

            highest_in_trade = max(trade.data.high.get(ago=0, size=barlen+1))
            lowest_in_trade = min(trade.data.low.get(ago=0, size=barlen+1))
            hp = 100 * (highest_in_trade - pricein) / pricein
            lp = 100 * (lowest_in_trade - pricein) / pricein
            if dir == 'long':
                mfe = hp
                mae = lp
            if dir == 'short':
                mfe = -lp
                mae = -hp

            self.trades.append({'ref': trade.ref, 
             'ticker': trade.data._name, 
             'dir': dir,
             'datein': datein, 
             'pricein': pricein, 
             'dateout': dateout, 
             'priceout': priceout,
             'chng%': round(pcntchange, 2), 
             'pnl': pnl, 'pnl%': round(pnlpcnt, 2),
             'size': size, 
             'value': value, 
             'cumpnl': self.cumprofit,
             'nbars': barlen, 'pnl/bar': round(pbar, 2),
             'mfe%': round(mfe, 2), 'mae%': round(mae, 2)})
            
    def get_analysis(self):
        return self.trades

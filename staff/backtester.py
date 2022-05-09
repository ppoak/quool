import datetime
import pandas as pd
import backtrader as bt
from ..tools import *

@pd.api.extensions.register_dataframe_accessor("relocator")
@pd.api.extensions.register_series_accessor("relocator")
class Relocator(Worker):

    def profit(self, weight: pd.Series = None, weight_col: str = None, forward_col: str = None):
        '''calculate profit from weight and forward
        ---------------------------------------------

        weight_col: str, the column name of weight
        forward_col: str, the column name of forward
        '''
        if self.type_ == Worker.TS:
            raise TypeError('Please transform your data into multiindex data')
        
        elif self.type_ == Worker.CS:
            raise TypeError('We cannot calculate the profit by cross section data')
        
        elif self.type_ == Worker.PN and self.is_frame:
            if weight_col is None or forward_col is None:
                raise ValueError('Please specify the weight and forward column')
            return self.data.groupby(level=0).apply(lambda x:
                (x.loc[:, weight_col] * x.loc[:, forward_col]).sum()
                / x.loc[:, weight_col].sum()
            )
        
        elif self.type_ == Worker.PN and not self.is_frame:
            # if you pass a Series in a panel form without weight, we assume that 
            # value is the forward return and weights are all equal
            if weight is None:
                return self.data.groupby(level=0).mean()
            # if other is not None, and data is a Series, we assume that weight is
            # the weight and data is the forward return
            else:
                if self.data.name is None:
                    self.data.name = 'forward'
                if weight.name is None:
                    weight.name = 'weight'
                data = pd.merge(self.data, weight, left_index=True, right_index=True)
                return data.groupby(level=0).apply(
                    lambda x: (x.iloc[:, 0] * x.iloc[:, 1]).sum() / x.iloc[:, 1].sum()
                )

class Strategy(bt.Strategy):

    def log(self, text: str, datetime: datetime.datetime = None, type_: str = 'info'):
        '''Logging function'''
        datetime = datetime or self.data.datetime.date(0)
        datetime = time2str(datetime)
        if type_ == 'info':
            hint = '[*]'
        elif type_ == 'warn':
            hint = '[!]'
        elif type_ == 'error':
            hint = '[x]'
        print(f'{hint} {datetime}: {text}')

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
            if order.isbuy():
                self.log(f'Buy at {order.executed.price:.2f}')
            elif order.issell():
                self.log(f'Sell at {order.executed.price:.2f}')
            # record current bar number
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected, order.Expired]:
            self.log('Order canceled, margin, rejected or expired', type_='warn')

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

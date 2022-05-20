import datetime
import pandas as pd
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
        if self.type_ == Worker.TS:
            raise BackTesterError('profit', 'Please transform your data into multiindex data')
        
        elif self.type_ == Worker.CS:
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
                )
        
    def turnover(self):
        '''calculate turnover'''
        if self.type_ == Worker.TS:
            raise BackTesterError('turnover', 'Please transform your data into multiindex data')
        
        elif self.type_ == Worker.CS:
            raise BackTesterError('turnover', 'We cannot calculate the turnover by cross section data')

        datetime_index = self.data.index.get_level_values(0).unique()
        ret = pd.Series(index=datetime_index, dtype='float64')
        ret.loc[datetime_index[0]] = 1
        for i, d in enumerate(datetime_index[1:]):
            delta_frame = pd.concat([self.data.loc[d], self.data.loc[datetime_index[i]]], 
                axis=1, join='outer').fillna(0)
            delta = delta_frame.iloc[:, 0] - delta_frame.iloc[:, -1]
            delta = delta.abs().sum() / self.data.loc[datetime_index[i]].abs().sum()
            ret.loc[d] = delta
        return ret


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

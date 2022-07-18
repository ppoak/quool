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

    def profit(self, 
        ret: pd.Series, 
        portfolio: pd.Series = None,
        ):
        '''calculate profit from weight and forward
        ---------------------------------------------

        ret: pd.Series, the return data in either PN series or TS frame form
        portfolio: pd.Series, the portfolio tag marked by a series, 
            only available when passing a PN
        '''
        if self.type_ == Worker.TS and self.is_frame:
            weight = self.data.stack()
        elif self.type_ == Worker.PN and not self.is_frame:
            weight = self.data.copy()
        else:
            raise BackTesterError('profit', 'Your weight data should either be in PN series or TS frame form')
        
        if isinstance(ret, pd.Series) and isinstance(ret.index, 
            pd.MultiIndex) and isinstance(ret.index.levels[0], pd.DatetimeIndex):
            pass
        elif isinstance(ret, pd.DataFrame) and isinstance(ret.index, 
            pd.DatetimeIndex) and ret.columns.size > 1:
            ret = ret.stack()
        else:
            raise BackTesterError('profit', 'Your return data should either be in PN series or TS frame form')
        
        if portfolio is not None:
            if isinstance(portfolio, pd.Series) and isinstance(portfolio.index, 
                pd.MultiIndex) and isinstance(portfolio.index.levels[0], pd.DatetimeIndex):
                pass
            if isinstance(portfolio, pd.DataFrame) and isinstance(portfolio.index, 
                pd.DatetimeIndex) and portfolio.columns.size > 1:
                portfolio = portfolio.stack()
            else:
                raise BackTesterError('profit', 'Your portofolio data should either be in PN series or TS frame form')
                
        if portfolio is not None:
            grouper = [portfolio, pd.Grouper(level=0)]
        else:
            grouper = pd.Grouper(level=0) 
        
        return weight.groupby(grouper).apply(lambda x: 
            (ret.loc[x.index] * x).sum() / x.sum())
    
    def networth(self, price: 'pd.Series | pd.DataFrame'):
        """Calculate the networth curve using normal price data
        --------------------------------------------------------

        price: pd.Series or pd.DataFrame, the price data either in
            MultiIndex form or the TS Matrix form
        return: pd.Series, the networth curve
        """
        if self.type_ == Worker.TS and self.is_frame:
            relocate_date = self.data.index
        elif self.type_ == Worker.PN and not self.is_frame:
            relocate_date = self.data.index.levels[0]
        else:
            raise BackTesterError('networth', 'Portfolio data should be in PN series or TS frame form')
        
        if isinstance(price, pd.DataFrame) and price.columns.size > 1:
            datetime_index = price.index
        elif isinstance(price.index, pd.MultiIndex) and isinstance(price.index.levels[0], 
            pd.DatetimeIndex) and isinstance(price, pd.Series):
            datetime_index = price.index.levels[0]
        else:
            raise BackTesterError('networth', 'Price data should be in PN series or TS frame form')
            
        lrd = relocate_date[0]
        lnet = (price.loc[d] * self.data.loc[lrd]).sum()
        lcnet = 1
        net = pd.Series(np.ones(datetime_index.size), index=datetime_index)
        for d in datetime_index[1:]:
            cnet = (price.loc[d] * self.data.loc[lrd]).sum() / lnet * lcnet
            lrd = relocate_date[relocate_date <= d][-1]
            if d == lrd:
                lcnet = cnet
                lnet = (price.loc[d] * self.data.loc[lrd]).sum()
            net.loc[d] = cnet
        return net
        
    def turnover(self, side: str = 'both'):
        '''calculate turnover
        ---------------------

        side: str, choice between "buy", "short" or "both"
        '''
        if self.type_ == Worker.TS:
            raise BackTesterError('turnover', 'Please transform your data into multiindex data')
        
        elif self.type_ == Worker.CS:
            raise BackTesterError('turnover', 'We cannot calculate the turnover by cross section data')

        datetime_index = self.data.index.levels[0]
        ret = pd.Series(np.ones(datetime_index.size), index=datetime_index)
        for i, d in enumerate(datetime_index[1:]):
            delta_frame = pd.concat([self.data.loc[d], 
                self.data.loc[datetime_index[i]]], axis=1, join='outer').fillna(0)
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
        if hint == "INFO":
            color = "color"
        elif hint == "WARN":
            color = "yellow"
        elif hint == "ERROR":
            color = "red"
        else:
            color = "blue"
        CONSOLE.print(f'[{color}][{hint}][/{color}] {datetime}: {text}')

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
            self.log('Order canceled, margin, rejected or expired', hint='WARN')

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


class OrderTable(bt.Analyzer):

    def __init__(self):
        self.orders = pd.DataFrame(columns=['asset', 'size', 'price', 'direction'])
        self.orders.index.name = 'datetime'

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.orders.loc[self.data.datetime.date(0)] = [
                    self.data._name, order.executed.size, 
                    order.executed.price, 'BUY']
            elif order.issell():
                self.orders.loc[self.data.datetime.date(0)] = [
                    self.data._name, order.executed.size, 
                    order.executed.price, 'SELL']
        
    def get_analysis(self):
        self.rets = self.orders
        return self.orders


if __name__ == "__main__":
    data = pd.Series(np.random.rand(100), index=pd.MultiIndex.from_product(
        [pd.date_range('20200101', periods=20, freq='3d'), list('abcde')]))
    ret = pd.Series(np.random.rand(300), index=pd.MultiIndex.from_product(
        [pd.date_range('20200101', periods=60), list('abcde')]))
    port = data.groupby(level=0).apply(pd.qcut, labels=False, q=2) + 1
    position = pd.Series(np.random.rand(10), index=pd.MultiIndex.from_tuples([
        (pd.to_datetime('20200101'), 'a'),
        (pd.to_datetime('20200101'), 'b'),
        (pd.to_datetime('20200102'), 'c'),
        (pd.to_datetime('20200102'), 'a'),
        (pd.to_datetime('20200103'), 'd'),
        (pd.to_datetime('20200103'), 'e'),
        (pd.to_datetime('20200104'), 'a'),
        (pd.to_datetime('20200104'), 'e'),
        (pd.to_datetime('20200105'), 'b'),
        (pd.to_datetime('20200105'), 'e'),
    ]))
    # print(data.relocator.profit(ret))
    print(position.relocator.turnover())
    
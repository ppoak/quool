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

    def __init__(self, data: 'pd.DataFrame | pd.Series'):
        super().__init__(data)
        weight = self.make_available(self.data)
        if isinstance(weight, bool):
            raise BackTesterError('profit', 'Your weight data should either be in PN series or TS frame form')
        self.weight = weight.groupby(level=0).apply(lambda x: x / x.sum())


    def make_available(self, data: 'pd.DataFrame | pd.Series'):
        if self.ists(data) and self.isframe(data):
            return data.stack()
        elif self.ispanel(data) and self.isseries(data):
            return data
        elif self.ispanel(data) and self.isframe(data) and data.columns.size == 1:
            return data.iloc[:, 0]
        else:
            return False
    
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
        
        weight = self.weight.copy()
        
        ret = self.make_available(ret)
        if isinstance(ret, bool):
            raise BackTesterError('profit', 'Your return data should either be in PN series or TS frame form')
        
        if portfolio is not None:
            portfolio = self.make_available(portfolio)
            if portfolio == False:
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
        weight = self.weight.copy()
        price = self.make_available(price)
        if isinstance(price, bool):
            raise BackTesterError('networth', 'Price data should be in PN series or TS frame form')
            
        relocate_date = weight.index.levels[0]
        datetime_index = price.index.levels[0]
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
        weight = self.weight.copy()
        weight = weight.reindex(pd.MultiIndex.from_product(
            [weight.index.levels[0], weight.index.levels[1]],
            names = ['date', 'asset'],
        ))
        delta = pd.concat([weight, weight.groupby(level=1).shift(1)], axis=1, join='outer').fillna(0)
        delta = delta.iloc[:, 0] - delta.iloc[:, -1]
        if side == 'both':
            return delta.groupby(level=0).apply(lambda x: x.abs().sum())
        elif side == 'buy':
            return delta.groupby(level=0).apply(lambda x: x[x > 0].abs().sum())
        elif side == 'sell':
            return delta.groupby(level=0).apply(lambda x: x[x < 0].abs().sum())


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
    print(position.relocator.turnover(side='sell'))
    
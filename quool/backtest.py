import logging
import numpy as np
import pandas as pd
import backtrader as bt
import matplotlib.pyplot as plt
from pathlib import Path
from .tools import parse_date, Logger


class Strategy(bt.Strategy):
    params = (("minstake", 100), )
    logger = Logger("QuoolStrategy", display_time=False, display_name=False)

    def log(self, text: str, level: int = logging.INFO, datetime: pd.Timestamp = None):
        """Logging function"""
        datetime = datetime or self.data.datetime.date(0)
        self.logger.log(level=level, msg=f'[{datetime}]: {text}')
    
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
        """order notification"""
        # order possible status:
        # 'Created'、'Submitted'、'Accepted'、'Partial'、'Completed'、
        # 'Canceled'、'Expired'、'Margin'、'Rejected'
        # broker submitted or accepted order do nothing
        if order.status in [order.Submitted, order.Accepted, order.Created]:
            return

        # broker completed order, just hint
        elif order.status in [order.Completed]:
            self.log(f'Order <{order.executed.size}> <{order.data._name}> at <{order.executed.price:.2f}>')
            # record current bar number
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected, order.Expired]:
            self.log('Order canceled, margin, rejected or expired')

        # except the submitted, accepted, and created status,
        # other order status should reset order variable
        self.order = None

    def notify_trade(self, trade):
        """trade notification"""
        if not trade.isclosed:
            # trade not closed, skip
            return
        # else, log it
        self.log(f'Gross Profit: {trade.pnl:.2f}, Net Profit {trade.pnlcomm:.2f}')


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


class OrderTable(Analyzer):

    def __init__(self):
        self.orders = []

    def notify_order(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                self.orders.append([
                    self.data.datetime.date(0),
                    order.data._name, order.executed.size, 
                    order.executed.price, 'BUY']
                )
            elif order.issell():
                self.orders.append([
                    self.data.datetime.date(0),
                    order.data._name, order.executed.size, 
                    order.executed.price, 'SELL']
                )
        
    def get_analysis(self):
        self.rets = pd.DataFrame(self.orders, columns=['datetime', 'code', 'size', 'price', 'direction'])
        self.rets["datetime"] = pd.to_datetime(self.rets["datetime"])
        self.rets = self.rets.set_index(['datetime', 'code'])
        return self.rets


class CashValueRecorder(Analyzer):
    def __init__(self):
        self.cash = []
        self.value = []
        self.date = []

    def next(self):
        self.cash.append(self.strategy.broker.get_cash())
        self.value.append(self.strategy.broker.get_value())
        self.date.append(self.strategy.data.datetime.date(0))

    def get_analysis(self):
        self.rets = pd.DataFrame(
            {'cash': self.cash, 'value': self.value}, 
            index=pd.to_datetime(self.date)
        )
        self.rets.index.name = "datetime"
        return self.rets 


class Relocator:

    def _format(self, data: pd.DataFrame):
        if isinstance(data.index, pd.MultiIndex):
            return data
        elif isinstance(data, pd.DataFrame) and isinstance(data.index, pd.DatetimeIndex):
            return data.stack()
        else:
            raise ValueError("Malformed format of data")

    def __init__(
        self,
        price: pd.DataFrame,
        code_index: str = 'code',
        date_index: str = 'date',
        buy_column: str = "open",
        sell_column: str = "close",
        commision: float = 0.005,
    ):
        self.price = self._format(price)
        self.buy_price = self.price[buy_column] if isinstance(self.price, pd.DataFrame) else self.price
        self.sell_price = self.price[sell_column] if isinstance(self.price, pd.DataFrame) else self.price
        self.code_index = code_index
        self.date_index = date_index
        self.commision = commision

    def turnover(
        self, 
        weight: pd.DataFrame | pd.Series, 
        side: str = 'both'
    ):
        weight = weight.reindex(pd.MultiIndex.from_product([
            weight.index.get_level_values(self.code_index).unique(), 
            weight.index.get_level_values(self.date_index).unique()
        ], names = [self.code_index, self.date_index])).fillna(0)

        preweight = weight.groupby(level=self.code_index).shift(1).fillna(0)
        delta = weight - preweight
        if side == 'both':
            return delta.groupby(level=self.date_index).apply(lambda x: x.abs().sum() / 2)
        elif side == 'buy':
            return delta.groupby(level=self.date_index).apply(lambda x: x[x > 0].abs().sum())
        elif side == 'sell':
            return delta.groupby(level=self.date_index).apply(lambda x: x[x < 0].abs().sum())
    
    def profit(
        self, 
        weight: pd.DataFrame | pd.Series, 
    ):
        weight = self._format(weight)
        commision = (self.turnover(weight) * self.commision)
        buy_price = self.buy_price.loc[weight.index]
        sell_price = self.sell_price.loc[weight.index].groupby(level=self.code_index).shift(-1)
        ret = (sell_price - buy_price) / buy_price
        return weight.groupby(level=self.date_index, group_keys=False).apply(lambda x: 
            (ret.loc[x.index] * x).sum() - commision.loc[x.index.get_level_values(self.date_index)[0]]).shift(1)
    
    def netvalue(
        self,
        weight: pd.DataFrame | pd.Series,    
    ):
        weight = self._format(weight)
        relocate_date = weight.index.get_level_values(self.date_index).unique()
        datetime_index = self.price.index.get_level_values(self.date_index).unique()
        lrd = relocate_date[0]
        lnet = (self.price.loc[d] * self.weight.loc[lrd]).sum()
        lcnet = 1
        net = pd.Series(np.ones(datetime_index.size), index=datetime_index)
        for d in datetime_index[1:]:
            cnet = (self.price.loc[d] * self.weight.loc[lrd]).sum() / lnet * lcnet
            lrd = relocate_date[relocate_date <= d][-1]
            if d == lrd:
                lcnet = cnet
                lnet = (self.price.loc[d] * self.weight.loc[lrd]).sum()
            net.loc[d] = cnet
        return net


class BackTrader:

    def __init__(
        self, 
        data: pd.DataFrame, 
        code_index: str = 'code',
        date_index: str = 'date',
    ):
        self.logger = Logger("QuoolBackTrader")
        self.data = data
        self.data = self._valid(data)
        self.data = self.data.reindex(pd.MultiIndex.from_product([
            self.data.index.get_level_values(code_index).unique(),
            self.data.index.get_level_values(date_index).unique(),
        ], names=[code_index, date_index])).fillna(0)
        self.code_index = code_index
        self.date_index = date_index
    
    def _valid(self, data: pd.DataFrame) -> pd.DataFrame:
        if isinstance(data, pd.DataFrame) and not 'close' in data.columns:
            raise ValueError('Your data should at least have a column named close')
        
        required_col = ['open', 'high', 'low']
        if isinstance(data, pd.DataFrame):
            # you should at least have a column named close
            for col in required_col:
                if not col in data.columns and col != 'volume':
                    data[col] = data['close']
            if not 'volume' in data.columns:
                data['volume'] = 0
        else:
            # just a series, all ohlc data will be the same, volume set to 0
            data = data.to_frame(name='close')
            for col in required_col:
                data[col] = col
            data['volume'] = 0

        return data

    def run(
        self, 
        strategy: bt.Strategy, 
        start: str = None,
        stop: str = None,
        cash: float = 1e6,
        indicators: 'bt.Indicator | list' = None,
        analyzers: 'bt.Analyzer | list' = None,
        observers: 'bt.Observer | list' = None,
        coc: bool = False,
        minstake: int = 100,
        commission: float = 0.005,
        verbose: bool = True,
        detail_img: str | Path = None,
        simple_img: str | Path = None,
        data_path: str | Path = None,
        maxcpu: int = None,
        **kwargs
    ):
        start = parse_date(start) if start is not None else\
            self.data.index.get_level_values(self.date_index).min()
        stop = parse_date(stop) if stop is not None else\
            self.data.index.get_level_values(self.date_index).max()
        cerebro = bt.Cerebro()
        cerebro.broker.setcash(cash)
        if coc:
            cerebro.broker.set_coc(True)
        cerebro.broker.setcommission(commission=commission)

        indicators = [indicators] if not isinstance(indicators, list) else indicators
        analyzers = analyzers if isinstance(analyzers, list) else [analyzers]
        analyzers += [bt.analyzers.SharpeRatio, bt.analyzers.TimeDrawDown, bt.analyzers.AnnualReturn,
                     bt.analyzers.TimeReturn, OrderTable, CashValueRecorder]
        observers = observers if isinstance(observers, list) else [observers]
        observers += [bt.observers.DrawDown]
        
        more = set(self.data.columns.to_list()) - set(['open', 'high', 'low', 'close', 'volume'])
        PandasData = type("_PandasData", (bt.feeds.PandasData,), {"lines": tuple(more), "params": tuple(zip(more, [-1] * len(more)))})
        # without setting bt.metabase._PandasData, PandasData cannot be pickled
        bt.metabase._PandasData = PandasData
        # add data
        if isinstance(self.data.index, pd.MultiIndex):
            datanames = self.data.index.get_level_values(self.code_index).unique().to_list()
        else:
            datanames = ['data']
        for dn in datanames:
            d = self.data.xs(dn, level=self.code_index) if \
                isinstance(self.data.index, pd.MultiIndex) else self.data
            feed = PandasData(dataname=d, fromdate=start, todate=stop)
            cerebro.adddata(feed, name=dn)
        
        for indicator in indicators:
            if indicator is not None:
                cerebro.addindicator(indicator)
        if 'minstake' not in strategy.params._getkeys():
            strategy.params.add('minstake', 1)
        if maxcpu is None:
            cerebro.addstrategy(strategy, minstake=minstake, **kwargs)
        else:
            cerebro.optstrategy(strategy, minstake=minstake, **kwargs)
        for analyzer in analyzers:
            if analyzer is not None:
                cerebro.addanalyzer(analyzer)
        for observer in observers:
            if observer is not None:
                cerebro.addobserver(observer)
        
        strats = cerebro.run(maxcpu=maxcpu)
        if maxcpu:
            strats = [strat[0] for strat in strats]
            # return strats
        params = ['_'.join([f'{key}({strat.params._getkwargs()[key]})' 
            for key in kwargs.keys()]) for strat in strats]
        timereturn = pd.DataFrame(
            [strat.analyzers.timereturn.rets for strat in strats],
        ).T
        timereturn.columns = params
        timereturn.index.name = "datetime"
        netvalue = (timereturn + 1).cumprod()

        if verbose:
            self.logger.info('=' * 15 + " Return " + '=' * 15)
            for i, strat in enumerate(strats):
                self.logger.info("-" * 15 + " total return " + "-" * 15)
                self.logger.info(f"total return {params[i]}: {(netvalue.iloc[-1, i] - 1) * 100:.2f} (%)")
                self.logger.info(f"annual return {params[i]}: "
                    f"<{start.strftime('%Y')}> - <{stop.strftime('%Y')}>: "
                    f"{strat.analyzers.annualreturn.rets}")
                self.logger.info('-' * 15 + " Time Drawdown " + '-' * 15)
                self.logger.info(f'{params[i]}: {dict(strat.analyzers.timedrawdown.rets)}')
                self.logger.info('-' * 15 + " Sharpe " + '-' * 15)
                self.logger.info(f'{params[i]}: {dict(strat.analyzers.sharperatio.rets)}')
        
        if detail_img is not None and maxcpu is None:
            if len(datanames) > 3:
                self.logger.warning(f"There are {len(datanames)} stocks, the image "
                      "may be nested and takes a long time to draw")
            figs = cerebro.plot(style='candel')
            fig = figs[0][0]
            fig.set_size_inches(18, 3 + 6 * len(datanames))
            fig.savefig(detail_img, dpi=300)

        if simple_img is not None:
            netvalue.plot()
            plt.savefig(simple_img)
        
        if data_path is not None:
            with pd.ExcelWriter(data_path) as writer:
                timereturn.to_excel(writer, sheet_name='RET')
                netvalue.to_excel(writer, sheet_name='VAL')
                for i, strat in enumerate(strats):
                    strat.analyzers.ordertable.rets.to_excel(writer, sheet_name='ORD_' + params[i])
                    strat.analyzers.cashvaluerecorder.rets.to_excel(writer, sheet_name='CASH_' + params[i])

        return strats

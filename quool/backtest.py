import logging
import pandas as pd
import backtrader as bt
import matplotlib.pyplot as plt
from pathlib import Path
from .equipment import parse_date, Logger


class Strategy(bt.Strategy):
    params = (("minstake", 100), )
    logger = Logger("QuoolStrategy", level=logging.DEBUG, display_time=False, display_name=False)

    def log(self, text: str, level: int = logging.DEBUG, datetime: pd.Timestamp = None):
        """Logging function"""
        datetime = datetime or self.data.datetime.date(0)
        self.logger.log(level=level, msg=f'[{datetime}] {text}')
    
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
            self.log(f'{order.data._name} ref.{order.ref} order {order.executed.size} at {order.executed.price:.2f}')

        elif order.status in [order.Canceled, order.Margin, order.Rejected, order.Expired]:
            self.log(f'{order.data._name} ref.{order.ref} canceled, margin, rejected or expired')

    def notify_trade(self, trade):
        """trade notification"""
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


class OrderTable(Analyzer):

    def __init__(self):
        self.orders = []

    def notify_order(self, order):
        if not order.alive():
            self.orders.append({
                'datetime': order.data.datetime.date(0),
                'code': order.data._name,
                'ref': order.ref,
                'type': order.ordtypename(),
                'status': order.getstatusname(),
                'createdprice': order.created.price,
                'createdsize': order.created.size,
                'excecutedprice': order.executed.price,
                'excecutedsize': order.executed.size,
                'pricelimit': order.pricelimit,
                'trailamount': order.trailamount,
                'trailpercent': order.trailpercent,
                'exectype': order.getordername(),
            })
        
    def get_analysis(self):
        self.rets = pd.DataFrame(self.orders)
        self.rets["datetime"] = pd.to_datetime(self.rets["datetime"])
        self.rets = self.rets.set_index(['datetime', 'code'])
        return self.rets


class CashValueRecorder(Analyzer):
    def __init__(self):
        self.cashvalue = []

    def next(self):
        self.cashvalue.append({
            'datetime': self.data.datetime.date(0),
            'cash': self.strategy.broker.get_cash(),
            'value': self.strategy.broker.get_value(),
        })

    def get_analysis(self):
        self.rets = pd.DataFrame(self.cashvalue)
        self.rets['datetime'] = pd.to_datetime(self.rets['datetime'])
        self.rets = self.rets.set_index('datetime')
        self.rets.index.name = "datetime"
        return self.rets


class Cerebro:

    def __init__(
        self, 
        data: pd.DataFrame, 
        code_index: str = 'code',
        date_index: str = 'date',
    ):
        self.logger = Logger("QuoolCerebro", display_time=False)
        self.data = data
        self.data = self._valid(data)
        self.data = self.data.reindex(pd.MultiIndex.from_product([
            self.data.index.get_level_values(code_index).unique(),
            self.data.index.get_level_values(date_index).unique(),
        ], names=[code_index, date_index])).sort_index()
        self.data.loc[:, ["open", "high", "low", "close", "volume"]] = \
            self.data.loc[:, ["open", "high", "low", "close", "volume"]].groupby(
                level=code_index).ffill()
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
        *,
        start: str = None,
        stop: str = None,
        cash: float = 1e6,
        benchmark: pd.DataFrame | pd.Series = None,
        indicators: 'bt.Indicator | list' = None,
        analyzers: 'bt.Analyzer | list' = None,
        observers: 'bt.Observer | list' = None,
        coc: bool = False,
        minstake: int = 100,
        commission: float = 0.005,
        maxcpus: int = None,
        preload: bool = True,
        runonce: bool = True,
        exactbars: bool = False,
        optdatas: bool = True,
        optreturn: bool = True,
        param_format: str = None,
        verbose: bool = True,
        detail_img: str | Path = None,
        simple_img: str | Path = None,
        data_path: str | Path = None,
        **kwargs
    ):
        start = parse_date(start) if start is not None else\
            self.data.index.get_level_values(self.date_index).min()
        stop = parse_date(stop) if stop is not None else\
            self.data.index.get_level_values(self.date_index).max()
        cerebro = bt.Cerebro(stdstats=False)
        cerebro.broker.setcash(cash)
        if coc:
            cerebro.broker.set_coc(True)
        cerebro.broker.setcommission(commission=commission)

        indicators = [indicators] if not isinstance(indicators, list) else indicators
        analyzers = analyzers if isinstance(analyzers, list) else [analyzers]
        analyzers += [bt.analyzers.SharpeRatio, bt.analyzers.TimeDrawDown, 
                      OrderTable, CashValueRecorder]
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
        if maxcpus is None:
            cerebro.addstrategy(strategy, minstake=minstake, **kwargs)
        else:
            cerebro.optstrategy(strategy, minstake=minstake, **kwargs)
        for analyzer in analyzers:
            if analyzer is not None:
                cerebro.addanalyzer(analyzer)
        for observer in observers:
            if observer is not None:
                cerebro.addobserver(observer)
        
        strats = cerebro.run(
            maxcpus = maxcpus,
            preload = preload,
            runonce = runonce,
            exactbars = exactbars,
            optdatas = optdatas,
            optreturn = optreturn,
        )
        if maxcpus is not None:
            strats = [strat[0] for strat in strats]
        
        if param_format is None:
            param_format = f"_".join([f"{key}{{{key}}}" for key in kwargs.keys()])
        params = [param_format.format(**strat.params._getkwargs()) for strat in strats]
        cashvalue = [strat.analyzers.cashvaluerecorder.get_analysis() for strat in strats]
        if benchmark is not None:
            benchmark = benchmark / benchmark.groupby(level=self.code_index).apply(lambda x: x.iloc[0])
            benchmark = benchmark["close"].unstack(level=self.code_index) * cash
            cashvalue_ = []
            for cv in cashvalue:
                cvb = pd.concat([cv, benchmark, (
                    -benchmark.pct_change().sub(
                        cv["value"].pct_change(), axis=0
                    ).fillna(0) + 1
                ).cumprod().add_prefix("ex-") * cash], axis=1).ffill()
                cvb.index.name = 'datetime'
                cashvalue_.append(cvb)
            cashvalue = cashvalue_
        
        abstract = []
        for i, strat in enumerate(strats):
            ab = strat.params._getkwargs()
            ret = (cashvalue[i].dropna().iloc[-1] /
                cashvalue[i].dropna().iloc[0] - 1) * 100
            ret = ret.drop(index="cash").add_prefix("return_").add_suffix("(%)")
            ab.update(ret.to_dict())
            for analyzer in strat.analyzers:
                ret = analyzer.get_analysis()
                if isinstance(ret, dict):
                    ab.update(ret)
            abstract.append(ab)
        abstract = pd.DataFrame(abstract)
        abstract = abstract.set_index(keys=list(kwargs.keys()))
        
        if verbose:
            self.logger.info(abstract)
        
        if detail_img is not None and maxcpus is None:
            if len(datanames) > 3:
                self.logger.warning(f"There are {len(datanames)} stocks, the image "
                      "may be nested and takes a long time to draw")
            figs = cerebro.plot(style='candel')
            fig = figs[0][0]
            fig.set_size_inches(18, 3 + 6 * len(datanames))
            if not isinstance(detail_img, bool):
                fig.savefig(detail_img, dpi=300)

        if simple_img is not None:
            fig, axes = plt.subplots(nrows=len(params), figsize=(20, 10 * len(params)))
            axes = [axes] if len(params) == 1 else axes
            for i, (name, cv) in enumerate(zip(params, cashvalue)):
                cv.plot(ax=axes[i], title=name)
            if not isinstance(simple_img, bool):
                fig.tight_layout()
                fig.savefig(simple_img)
        
        if data_path is not None:
            with pd.ExcelWriter(data_path) as writer:
                abstract.reset_index().to_excel(writer, sheet_name='ABSTRACT', index=False)
                for name, cv, strat in zip(params, cashvalue, strats):
                    cv.to_excel(writer, sheet_name='CV_' + name)
                    strat.analyzers.ordertable.rets.reset_index().to_excel(
                        writer, sheet_name='ORD_' + name, index=False)
    
        return strats

import logging
import pandas as pd
import numpy as np
import backtrader as bt
from pathlib import Path
from .tool import Logger, evaluate


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


class CBroker(bt.BackBroker):

    params = (
        ("minstake", 100), 
        ("minshare", 100), 
        ("divfactor", "divfactor"), 
        ("splitfactor", "splitfactor"),
        ("split_div_recorder", "tradeorderrecorder")
    )
    logger = Logger("CBroker", level=logging.DEBUG, display_time=False, display_name=False)

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
                self.add_cash(dividend)
                analyzer = getattr(self.cerebro.runningstrats[0].analyzers, self.params.split_div_recorder, None)
                if analyzer:
                    info = {
                        "notify_time": data.datetime.date(0),
                        "code": "Cash",
                        'type': "Dividend",
                        'reference': next(bt.Order.refbasis),
                        'status': "Completed",
                        'created_time': data.datetime.date(0),
                        'created_price': 1,
                        'created_size': dividend,
                        'executed_time': data.datetime.date(0),
                        'executed_price': 1,
                        'executed_size': dividend,
                        'commission': 0,
                    }
                    analyzer.trade_order.append(info)
                self.logger.debug(f"[{data.datetime.date(0)}] {data._name} ref.{info['reference']} dividend cash: {dividend:.2f}")

            # hold the stock which has split
            if size and not np.isnan(splitfactor):
                splitsize = int(size * splitfactor)
                pos.update(size=splitsize, price=0)
                analyzer = getattr(self.cerebro.runningstrats[0].analyzers, self.params.split_div_recorder, None)
                if analyzer:
                    info = {
                        "notify_time": data.datetime.date(0),
                        "code": data._name,
                        'reference': next(bt.Order.refbasis),
                        'type': "Split",
                        'status': "Completed",
                        'created_time': data.datetime.date(0),
                        'created_price': 0,
                        'created_size': splitsize,
                        'executed_time': data.datetime.date(0),
                        'executed_price': 0,
                        'executed_size': splitsize,
                        'commission': 0,
                    }
                    analyzer.trade_order.append(info)
                self.logger.debug(f"[{data.datetime.date(0)}] {data._name} ref.{info['reference']} split size: {splitsize:.0f}")
    
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


class TradeOrderRecorder(bt.Analyzer):
    params = (
        ("with_order", True),
        ("only_not_alive", True),
        ("with_trade", True),
        ("only_trade_close", True),
    )

    def __init__(self):
        self.trade_order = []

    def notify_order(self, order):
        if self.params.with_order:
            if order.alive() and not self.params.only_not_alive:
                self.trade_order.append({
                    "notify_time": order.data.datetime.date(0),
                    "code": order.data._name,
                    'reference': order.ref,
                    'type': order.ordtypename(),
                    'status': order.getstatusname(),
                    'created_time': bt.num2date(order.created.dt),
                    'created_price': order.created.price,
                    'created_size': order.created.size,
                    'price_limit': order.pricelimit,
                    'trail_amount': order.trailamount,
                    'trail_percent': order.trailpercent,
                    'execute_type': order.getordername(),
                    'commission': order.executed.comm,
                    **order.info
                })
            
            if not order.alive():
                self.trade_order.append({
                    "notify_time": order.data.datetime.date(0),
                    "code": order.data._name,
                    'reference': order.ref,
                    'type': order.ordtypename(),
                    'status': order.getstatusname(),
                    'created_time': bt.num2date(order.created.dt),
                    'created_price': order.created.price,
                    'created_size': order.created.size,
                    'executed_time': bt.num2date(order.executed.dt),
                    'executed_price': order.executed.price,
                    'executed_size': order.executed.size,
                    'price_limit': order.pricelimit,
                    'trail_amount': order.trailamount,
                    'trail_percent': order.trailpercent,
                    'execute_type': order.getordername(),
                    'commission': order.executed.comm,
                    **order.info
                })
    
    def notify_trade(self, trade):
        if self.params.with_trade:
            if not trade.isclosed and not self.params.only_trade_close:
                self.trade_order.append({
                    "notify_time": trade.data.datetime.date(0),
                    "code": trade.data._name,
                    'reference': trade.ref,
                    'type': 'Trade',
                    'status': trade.status_names[trade.status],
                    'created_time': trade.open_datetime(),
                    'created_price': trade.price,
                    'created_size': trade.size,
                    'commission': trade.commission,
                })
            
            if trade.isclosed:
                self.trade_order.append({
                    "notify_time": trade.data.datetime.date(0),
                    "code": trade.data._name,
                    'reference': trade.ref,
                    'type': 'Trade',
                    'status': trade.status_names[trade.status],
                    'created_time': trade.open_datetime(),
                    'executed_time': trade.close_datetime(),
                    'profit': trade.pnl,
                    'net_profit': trade.pnlcomm,
                    'commission': trade.commission,
                })
        
    def get_analysis(self):
        self.rets = pd.DataFrame(self.trade_order)
        if not self.rets.empty:
            self.rets["notify_time"] = pd.to_datetime(self.rets["notify_time"])
            self.rets["created_time"] = pd.to_datetime(self.rets["created_time"])
            self.rets["executed_time"] = pd.to_datetime(self.rets["executed_time"])
            self.rets = self.rets.set_index(["reference"])
        return self.rets


class CashValueRecorder(bt.Analyzer):
    
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
        self.rets["datetime"] = pd.to_datetime(self.rets["datetime"])
        self.rets = self.rets.set_index("datetime")
        self.rets.index.name = "datetime"
        return self.rets


class Cerebro:

    def __init__(
        self, 
        data: pd.DataFrame | pd.Series,
        code_level: int | str = 0,
        date_level : int | str = 1,
    ):
        if data.index.nlevels == 1:
            self._date_level = date_level if isinstance(date_level, str) else data.index.names[date_level]
            self._date_level = self._date_level or "date"
        else:
            self._code_level = code_level if isinstance(code_level, str) else data.index.names[code_level]
            self._code_level = self._code_level or "code"
            self._date_level = date_level if isinstance(date_level, str) else data.index.names[date_level]
            self._date_level = self._date_level or "date"
        self._data = data
        self._check()
        self.logger = Logger("Cerebro", display_time=False)

    def _check(self):
        if len(self._data.shape) > 1 and not 'close' in self._data.columns.str.lower():
            raise ValueError('Your data should at least have a column named close')
        
        required_col = ['open', 'high', 'low']
        base_col = required_col + ['close', 'volume']
        if len(self._data.shape) > 1:
            # you should at least have a column named close
            for col in required_col:
                if col != 'volume' and not col in self._data.columns.str.lower():
                    # if open, high, low doesn't exist, default setting to close
                    self._data[col] = self._data['close']
            if not 'volume' in self._data.columns.str.lower():
                # volume default is 0
                self._data['volume'] = 0
        else:
            # just a series, all ohlc data will be the same, volume set to 0
            self._data = self._data.to_frame(name='close')
            for col in required_col:
                self._data[col] = col
            self._data['volume'] = 0
        
        if self._data.index.nlevels == 1:
            self._data.index = pd.MultiIndex.from_product(
                [['data'], self._data.index],
                names = [self._code_level, self._date_level], 
            )
        
        self._data = self._data.reindex(index=pd.MultiIndex.from_product(
            [self._data.index.get_level_values(self._code_level).unique(), 
             self._data.index.get_level_values(self._date_level).unique()],
            names = [self._code_level, self._date_level],
        ))
        self._data.loc[:, base_col] = self._data.groupby(
            level=self._code_level)[base_col].ffill().fillna(0)
        self._data = self._data.sort_index()
        
        return self

    def run(
        self, 
        strategy: Strategy | list[Strategy], 
        *,
        broker: bt.BrokerBase = None,
        cash: float = 1e6,
        commission: float = 0.00025,
        indicators: 'bt.Indicator | list' = None,
        analyzers: 'bt.Analyzer | list' = None,
        observers: 'bt.Observer | list' = None,
        coc: bool = False,
        n_jobs: int = None,
        preload: bool = True,
        runonce: bool = True,
        exactbars: bool = False,
        optdatas: bool = True,
        optreturn: bool = True,
        **kwargs
    ):
        cerebro = bt.Cerebro(stdstats=False)

        if broker is None:
            broker = CBroker()
        elif isinstance(broker, tuple):
            broker = broker[0](**broker[1])
        else:
            raise ValueError("Broker should be a tuple of (class, kwargs)")
        cerebro.setbroker(broker)
        
        cerebro.broker.set_cash(cash)
        if coc:
            cerebro.broker.set_coc(True)
        cerebro.broker.setcommission(commission=commission)
        start = self._data.index.get_level_values(self._date_level).min()
        stop = self._data.index.get_level_values(self._date_level).max()
        
        # add data
        more = set(self._data.columns.to_list()) - set(['open', 'high', 'low', 'close', 'volume'])
        PandasData = type("_PandasData", (bt.feeds.PandasData,), {"lines": tuple(more), "params": tuple(zip(more, [-1] * len(more)))})
        # without setting bt.metabase._PandasData, PandasData cannot be pickled
        bt.metabase._PandasData = PandasData

        datanames = self._data.index.get_level_values(self._code_level).unique().to_list()
        for dn in datanames:
            d = self._data.xs(dn, level=self._code_level)
            feed = PandasData(dataname=d, fromdate=start, todate=stop)
            cerebro.adddata(feed, name=dn)

        # add indicators
        indicators = indicators if isinstance(indicators, list) else [indicators]
        for indicator in indicators:
            if isinstance(indicator, tuple):
                cerebro.addindicator(indicator[0], **indicator[1])
            elif isinstance(indicator, bt.Indicator):
                cerebro.addindicator(indicator)
        
        # add analyzers
        analyzers = analyzers if isinstance(analyzers, list) else [analyzers]
        cerebro.addanalyzer(TradeOrderRecorder)
        cerebro.addanalyzer(CashValueRecorder)
        for analyzer in analyzers:
            if isinstance(analyzer, tuple):
                cerebro.addanalyzer(analyzer[0], **analyzer[1])
            elif isinstance(analyzer, bt.Analyzer):
                cerebro.addanalyzer(analyzer)

        # add observers
        observers = observers if isinstance(observers, list) else [observers]
        for observer in observers:
            if isinstance(observer, tuple):
                cerebro.addobserver(observer[0], **observer[1])
            elif isinstance(observer, bt.Observer):
                cerebro.addobserver(observer)

        # add strategies
        strategy = [strategy] if not isinstance(strategy, list) else strategy
        if n_jobs is None:
            for strat in strategy:
                cerebro.addstrategy(strat, **kwargs)
        else:
            if len(strategy) > 1:
                raise ValueError("multiple strategies are not supported in optstrats mode")
            cerebro.optstrategy(strategy[0], **kwargs)
        
        # run strategies
        strats = cerebro.run(
            maxcpus = n_jobs,
            preload = preload,
            runonce = runonce,
            exactbars = exactbars,
            optdatas = optdatas,
            optreturn = optreturn,
        )
        if n_jobs is not None:
            # here we only get first element because backtrader doesn't
            # support multiple strategies in optstrategy mode
            strats = [strat[0] for strat in strats]
        
        self.strats = strats
    
    def evaluate(
        self,
        *,
        verbose: bool = True,
        benchmark: pd.Series = None,
        param_format: str = None,
        result: str | Path = None,
    ):
        if param_format is None:
            param_format = f"_".join([f"{key}{{{key}}}" for key in self.strats.kwargs.keys()])
        params = [param_format.format(**strat.params._getkwargs()) for strat in self.strats]
        cashvalue = [strat.analyzers.cashvaluerecorder.get_analysis() for strat in self.strats]

        # add benchmark if available
        if benchmark is not None:
            if (not isinstance(benchmark, pd.Series) or 
                not isinstance(benchmark.index, pd.Index)):
                raise TypeError('benchmark must be a pandas Series with datetime index')
            
            benchmark = benchmark.copy().ffill()
            benchmark.name = benchmark.name or 'benchmark'
            benchmark = benchmark / benchmark.dropna().iloc[0]
            cashvalue_ = []
            for cv in cashvalue:
                cvb = pd.concat([cv, benchmark * cv.iloc[0, 0], (
                    ((cv["value"].pct_change() - benchmark.pct_change())
                    .fillna(0) + 1).cumprod() * cv.iloc[0, 0]).to_frame(benchmark.name)
                    .add_prefix("ex(").add_suffix(')')
                ], axis=1).ffill()
                cvb.index.name = 'datetime'
                cashvalue_.append(cvb)
            cashvalue = cashvalue_
            
        abstract = []
        for i, strat in enumerate(self.strats):
            # basic parameters
            _abstract = strat.params._getkwargs()
            # add return to abstract
            _abstract.update(evaluate(cashvalue[i]["value"], cashvalue[i]["cash"], benchmark=benchmark).to_dict())
            tor = strat.analyzers.tradeorderrecorder.get_analysis()
            trd = tor[(tor["type"] == 'Trade') & (tor["status"] == "Closed")]
            if not trd.empty:
                # add trade count
                _abstract.update({"trade_count": trd.shape[0]})
                # add trade win rate
                wr = trd[trd["net_profit"] > 0].shape[0] / trd.shape[0]
                _abstract.update({"winrate(%)": wr * 100})
                # add average holding days
                avhd = (trd["executed_time"] - trd["created_time"]).mean()
                _abstract.update({"average_holding_days": avhd})
                # add average return rate
                avrr = (trd["net_profit"].values / (cashvalue[i].loc[
                    trd["created_time"].values, "value"].values)).mean()
                _abstract.update({"average_return_per_trade(%)": avrr * 100})
            # other analyzers information
            for analyzer in strat.analyzers:
                ret = analyzer.get_analysis()
                if isinstance(ret, dict):
                    _abstract.update(ret)
            # append result
            abstract.append(_abstract)
        abstract = pd.DataFrame(abstract)

        if self.strats[0].params._getkwargs():
            abstract = abstract.set_index(keys=list(self.strats[0].params._getkwargs()))
        
        if verbose:
            self.logger.info(abstract)
        
        if result is not None:
            with pd.ExcelWriter(result) as writer:
                abstract.reset_index().to_excel(writer, sheet_name='ABSTRACT', index=False)
                for name, cv, strat in zip(params, cashvalue, self.strats):
                    cv.to_excel(writer, sheet_name='CV_' + name)
                    strat.analyzers.tradeorderrecorder.rets.reset_index().to_excel(
                        writer, sheet_name='TO_' + name, index=False)

        return abstract


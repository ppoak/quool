import numpy as np
import pandas as pd
import backtrader as bt
import matplotlib.pyplot as plt
from pathlib import Path
from .core.util import Logger
from .core.backtrade import Strategy, Analyzer
from .data import Dim3Frame


class RebalanceStrategy(Strategy):

    params = (("ratio", 0.95), )

    def __init__(self) -> None:
        self.holdings = pd.Series(
            np.zeros(len(self.datas)), 
            index=[d._name for d in self.datas], name='holdings'
        )

    def next(self):
        target = pd.Series(dict([(d._name, d.portfolio[0]) for d in self.datas]), name='target')
        dec = target[target < self.holdings]
        inc = target[target > self.holdings]
        # sell before buy
        for d in dec.index:
            self.order_target_percent(data=d, target=target.loc[d] * (1 - self.p.ratio))
        for i in inc.index:
            self.order_target_percent(data=i, target=target.loc[i] * (1 - self.p.ratio))
        if not dec.empty or not inc.empty:
            self.holdings = target


class TradeOrderRecorder(Analyzer):
    params = (
        ("with_order", True),
        ("only_not_alive", True),
        ("with_trade", True),
        ("only_trade_close", True),
        ("code_level", "code"),
        ("date_level", "notify_time")
    )

    def __init__(self):
        self.trade_order = []

    def notify_order(self, order):
        if order.alive() and not self.params.only_not_alive:
            self.trade_order.append({
                self.params.date_level: order.data.datetime.date(0),
                self.params.code_level: order.data._name,
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
            })
        
        if not order.alive():
            self.trade_order.append({
                self.params.date_level: order.data.datetime.date(0),
                self.params.code_level: order.data._name,
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
            })
    
    def notify_trade(self, trade):
        if not trade.isclosed and not self.params.only_trade_close:
            self.trade_order.append({
                self.params.date_level: trade.data.datetime.date(0),
                self.params.code_level: trade.data._name,
                'reference': trade.ref,
                'type': 'Trade',
                'status': trade.status_names[trade.status],
                'created_time': trade.open_datetime(),
                'created_price': trade.price,
                'created_size': trade.size,
            })
        
        if trade.isclosed:
            self.trade_order.append({
                self.params.date_level: trade.data.datetime.date(0),
                self.params.code_level: trade.data._name,
                'reference': trade.ref,
                'type': 'Trade',
                'status': trade.status_names[trade.status],
                'created_time': trade.open_datetime(),
                'executed_time': trade.close_datetime(),
                'profit': trade.pnl,
                'net_profit': trade.pnlcomm,
            })
        
    def get_analysis(self):
        self.rets = pd.DataFrame(self.trade_order)
        if not self.rets.empty:
            self.rets[self.params.date_level] = pd.to_datetime(self.rets[self.params.date_level])
            self.rets = self.rets.set_index([self.params.date_level, self.params.code_level])
        return self.rets


class CashValueRecorder(Analyzer):
    params = (
        ("date_level", "date"),
    )
    def __init__(self):
        self.cashvalue = []

    def next(self):
        self.cashvalue.append({
            self.params.date_level: self.data.datetime.date(0),
            'cash': self.strategy.broker.get_cash(),
            'value': self.strategy.broker.get_value(),
        })

    def get_analysis(self):
        self.rets = pd.DataFrame(self.cashvalue)
        self.rets[self.params.date_level] = pd.to_datetime(self.rets[self.params.date_level])
        self.rets = self.rets.set_index(self.params.date_level)
        self.rets.index.name = self.params.date_level
        return self.rets


class Cerebro(Dim3Frame):

    def __init__(
        self, 
        data: pd.DataFrame | pd.Series,
        code_level: int | str = 0,
        date_level : int | str = 1,
        price_level: int | str = 2,
    ):
        self._code_level = code_level
        self._date_level = date_level
        super().__init__(data, price_level)
        self._check()
        self.logger = Logger("Cerebro", display_time=False)

    def _check(self):
        if self.naxes > 1 and not 'close' in self.columns.str.lower():
            raise ValueError('Your data should at least have a column named close')
        
        required_col = ['open', 'high', 'low']
        base_col = required_col + ['close', 'volume']
        if self.naxes > 1:
            # you should at least have a column named close
            for col in required_col:
                if col != 'volume' and not col in self.columns.str.lower():
                    # if open, high, low doesn't exist, default setting to close
                    self._data[col] = self._data['close']
            if not 'volume' in self.columns.str.lower():
                # volume default is 0
                self._data['volume'] = 0
        else:
            # just a series, all ohlc data will be the same, volume set to 0
            self._data = self._data.to_frame(name='close')
            for col in required_col:
                self._data[col] = col
            self._data['volume'] = 0
        
        self.panelize()
        self._data.loc[:, base_col] = self._data.groupby(
            level=self._code_level)[base_col].ffill().fillna(0)
        
        return self

    def __call__(
        self, 
        strategy: bt.Strategy | list[bt.Strategy], 
        *,
        cash: float = 1e6,
        coc: bool = False,
        commission: float = 0.005,
        riskfreerate: float = 0.02,
        verbose: bool = True,
        benchmark: pd.Series = None,
        indicators: 'bt.Indicator | list' = None,
        analyzers: 'bt.Analyzer | list' = None,
        observers: 'bt.Observer | list' = None,
        n_jobs: int = None,
        param_format: str = None,
        oldimg: str | Path = None,
        image: str | Path = None,
        result: str | Path = None,
        preload: bool = True,
        runonce: bool = True,
        exactbars: bool = False,
        optdatas: bool = True,
        optreturn: bool = True,
        **kwargs
    ):
        cerebro = bt.Cerebro(stdstats=False)
        cerebro.broker.set_cash(cash)
        if coc:
            self.cerebro.broker.set_coc(True)
        cerebro.broker.setcommission(commission=commission)
        riskfreerate = riskfreerate
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
            if indicator is not None:
                cerebro.addindicator(indicator)
        
        # add analyzers
        analyzers = analyzers if isinstance(analyzers, list) else [analyzers]
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=riskfreerate)
        cerebro.addanalyzer(bt.analyzers.TimeDrawDown)
        cerebro.addanalyzer(TradeOrderRecorder, date_level=self._date_level, code_level=self._code_level)
        cerebro.addanalyzer(CashValueRecorder, date_level=self._date_level)
        for analyzer in analyzers:
            if analyzer is not None:
                cerebro.addanalyzer(analyzer)

        # add observers
        observers = observers if isinstance(observers, list) else [observers]
        cerebro.addobserver(bt.observers.DrawDown)
        for observer in observers:
            if observer is not None:
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
        
        if param_format is None:
            param_format = f"_".join([f"{key}{{{key}}}" for key in kwargs.keys()])
        params = [param_format.format(**strat.params._getkwargs()) for strat in strats]
        cashvalue = [strat.analyzers.cashvaluerecorder.get_analysis() for strat in strats]

        # add benchmark if available
        if benchmark is not None:
            if (not isinstance(benchmark, pd.Series) or 
                not isinstance(benchmark.index, pd.Index)):
                raise TypeError('benchmark must be a pandas Series with datetime index')
            
            benchmark = benchmark.copy().ffill()
            benchmark.name = benchmark.name or 'benchmark'
            benchmark = benchmark / benchmark.dropna().iloc[0] * cash
            cashvalue_ = []
            for cv in cashvalue:
                cvb = pd.concat([cv, benchmark, (
                    (cv["value"].pct_change() - benchmark.pct_change())
                    .fillna(0) + 1).cumprod().to_frame(benchmark.name)
                    .add_prefix("ex(").add_suffix(')') * cash
                ], axis=1).ffill()
                cvb.index.name = 'datetime'
                cashvalue_.append(cvb)
            cashvalue = cashvalue_
            
        abstract = []
        for i, strat in enumerate(strats):
            # basic parameters
            ab = strat.params._getkwargs()
            # add return to abstract
            ret = (cashvalue[i].dropna().iloc[-1] /
                cashvalue[i].dropna().iloc[0] - 1) * 100
            ret = ret.drop(index="cash").add_prefix("return(").add_suffix(")(%)")
            ab.update(ret.to_dict())
            tor = strat.analyzers.tradeorderrecorder.get_analysis()
            trd = tor[(tor["type"] == 'Trade') & (tor["status"] == "Closed")]
            if not trd.empty:
                # add trade count
                ab.update({"trade_count": trd.shape[0]})
                # add trade win rate
                wr = trd[trd["net_profit"] > 0].shape[0] / trd.shape[0]
                ab.update({"winrate(%)": wr * 100})
                # add average holding days
                avhd = (trd["executed_time"] - trd["created_time"]).mean()
                ab.update({"average_holding_days": avhd})
                # add average return rate
                avrr = (trd["net_profit"].values / (cashvalue[i].loc[
                    trd["created_time"].values, "value"].values)).mean()
                ab.update({"average_return(%)": avrr * 100})
            # return dicomposed to beta and alpha
            if benchmark is not None:
                sc = cashvalue[i].iloc[:, 1].pct_change().dropna()
                bc = cashvalue[i].iloc[:, 2].pct_change().dropna()
                beta = sc.corr(bc) * sc.std() / bc.std()
                alpha = ab["return(value)(%)"] - (riskfreerate * 100 + beta * 
                    (ab[f"return({benchmark.name})(%)"] - riskfreerate * 100))
                ab.update({"alpha(%)": alpha, "beta": beta})
            # other analyzers information
            for analyzer in strat.analyzers:
                ret = analyzer.get_analysis()
                if isinstance(ret, dict):
                    ab.update(ret)
            # append result
            abstract.append(ab)
        abstract = pd.DataFrame(abstract)
        abstract = abstract.set_index(keys=list(kwargs.keys()))
        
        if verbose:
            self.logger.info(abstract)
        
        if oldimg is not None and n_jobs is None:
            if len(datanames) > 3:
                self.logger.warning(f"There are {len(datanames)} stocks, the image "
                      "may be nested and takes a long time to draw")
            figs = self.cerebro.plot(style='candel')
            fig = figs[0][0]
            fig.set_size_inches(18, 3 + 6 * len(datanames))
            if not isinstance(oldimg, bool):
                fig.savefig(oldimg, dpi=300)

        if image is not None:
            fig, axes = plt.subplots(nrows=len(params), figsize=(20, 10 * len(params)))
            axes = [axes] if len(params) == 1 else axes
            for i, (name, cv) in enumerate(zip(params, cashvalue)):
                cv.plot(ax=axes[i], title=name)
            if not isinstance(image, bool):
                fig.tight_layout()
                fig.savefig(image)
            else:
                fig.show()
        
        if result is not None:
            with pd.ExcelWriter(result) as writer:
                abstract.reset_index().to_excel(writer, sheet_name='ABSTRACT', index=False)
                for name, cv, strat in zip(params, cashvalue, strats):
                    cv.to_excel(writer, sheet_name='CV_' + name)
                    strat.analyzers.tradeorderrecorder.rets.reset_index().to_excel(
                        writer, sheet_name='TO_' + name, index=False)
    
        return strats

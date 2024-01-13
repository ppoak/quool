import logging
import numpy as np
import pandas as pd
import backtrader as bt
import matplotlib.pyplot as plt
from pathlib import Path
from .tool import parse_date, Logger, DimFormatter
from .exception import NotRequiredDimError


class Strategy(bt.Strategy):
    """
    A pre-defined trading strategy class for backtesting using Backtrader.

    Attributes:
        params (tuple): Parameters for the strategy, e.g., minimum stake size.
        logger (Logger): Custom logger for logging strategy events.

    Methods:
        log(self, text, level, datetime): Logs messages with a timestamp.
        resize(self, size): Adjusts the order size to comply with minimum stake.
        buy(self, ...): Places a buy order with adjusted size.
        sell(self, ...): Places a sell order with adjusted size.
        notify_order(self, order): Handles notifications for order status.
        notify_trade(self, trade): Handles notifications for trade executions.

    Example:
        class MyStrategy(Strategy):
            def __init__(self):
                # Strategy initialization code here
    """

    params = (
        ("minstake", 1), 
        ("minshare", 100), 
        ("splitfactor", "splitfactor"),
        ("divfactor", "divfactor"),
    )
    logger = Logger("QuoolStrategy", level=logging.DEBUG, display_time=False, display_name=False)

    def log(self, text: str, level: int = logging.DEBUG, datetime: pd.Timestamp = None):
        """
        Logs messages for the strategy with a timestamp.

        Args:
            text (str): The message to log.
            level (int): The logging level (e.g., logging.DEBUG).
            datetime (pd.Timestamp, optional): The timestamp for the log message.
        """
        datetime = datetime or self.data.datetime.date(0)
        self.logger.log(level=level, msg=f'[{datetime}] {text}')

    def split_dividend(self):
        """Process split and dividend information on market."""
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
        """
        Adjusts the order size to comply with the strategy's minimum stake.

        Args:
            size (int): The intended size of the order.

        Returns:
            int: The adjusted size of the order.
        """
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
        """
        Places a buy order with adjusted size.

        This method adjusts the order size using `resize` method and then
        places a buy order using the parent class's `buy` method.

        Args:
            data, size, price, plimit, exectype, valid, tradeid, oco, 
            trailamount, trailpercent, parent, transmit, **kwargs: 
            Parameters for the buy order (see Backtrader documentation for details).

        Returns:
            The order object returned by the parent class's `buy` method.
        """
        size = self.resize(size)
        return super().buy(data, size, price, plimit, 
            exectype, valid, tradeid, oco, trailamount, 
            trailpercent, parent, transmit, **kwargs)
    
    def sell(
        self, data=None, size=None, price=None, plimit=None, 
        exectype=None, valid=None, tradeid=0, oco=None, trailamount=None, 
        trailpercent=None, parent=None, transmit=True, **kwargs
    ):
        """
        Places a sell order with adjusted size.

        This method adjusts the order size using `resize` method and then
        places a sell order using the parent class's `sell` method.

        Args:
            data, size, price, plimit, exectype, valid, tradeid, oco, 
            trailamount, trailpercent, parent, transmit, **kwargs: 
            Parameters for the sell order (see Backtrader documentation for details).

        Returns:
            The order object returned by the parent class's `sell` method.
        """
        size = self.resize(size)
        return super().sell(data, size, price, plimit, 
            exectype, valid, tradeid, oco, trailamount, 
            trailpercent, parent, transmit, **kwargs)

    def notify_order(self, order: bt.Order):
        """
        Handles notifications for order status.

        This method logs the status of orders, including completed, canceled, 
        margin calls, rejected, and expired orders.

        Args:
            order (bt.Order): The order for which the notification is received.
        """
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
        """
        Handles notifications for trade executions.

        This method logs the details of closed trades, including profit and loss.

        Args:
            trade (bt.Trade): The trade for which the notification is received.
        """
        if not trade.isclosed:
            # trade not closed, skip
            return
        # else, log it
        self.log(f'{trade.data._name} gross {trade.pnl:.2f}, net {trade.pnlcomm:.2f}')

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


class Cerebro:
    """
    A free-to-go trading environment class that integrates with Backtrader for strategy testing.

    Attributes:
        logger (Logger): An instance of a custom Logger class for logging.
        data (pd.DataFrame): The input data for backtesting.
        code_level (str): Level name in 'data' that represents the stock symbol.
        date_level (str): Level name in 'data' that represents the dates.

    Methods:
        __init__(self, data, code_level, date_level): Initializes the Cerebro environment.
        _valid(self, data): Validates and preprocesses the input data.
        run(self, strategy, ...): Executes the specified trading strategy.

    Example:
        cerebro = Cerebro(data=my_dataframe)
        cerebro.run(strategy=my_strategy)
    """

    def __init__(
        self, 
        cash: float = 1e6,
        coc: bool = False,
        commission: float = 0.005,
        riskfreerate: float = 0.02,
        verbose: bool = True,
        code_level: str | int = 0,
        date_level: str | int = 1,
    ):
        self.logger = Logger("Cerebro", display_time=False)
        self.cerebro = bt.Cerebro(stdstats=False)
        self.cerebro.broker.setcash(cash)
        if coc:
            self.cerebro.broker.set_coc(True)
        self.cerebro.broker.setcommission(commission=commission)
        self.cash = cash
        self.riskfreerate = riskfreerate
        self.verbose = verbose
        self.code_level = code_level
        self.date_level = date_level

    def _format_data(self, formatter: DimFormatter) -> DimFormatter:
        if formatter.naxes > 1 and not 'close' in formatter.data.columns.str.lower():
            raise ValueError('Your data should at least have a column named close')
        
        required_col = ['open', 'high', 'low']
        base_col = required_col + ['close', 'volume']
        if formatter.naxes > 1:
            # you should at least have a column named close
            for col in required_col:
                if col != 'volume' and not col in formatter.data.columns.str.lower():
                    # if open, high, low doesn't exist, default setting to close
                    formatter.data[col] = formatter.data['close']
            if not 'volume' in formatter.data.columns.str.lower():
                # volume default is 0
                formatter['volume'] = 0
        else:
            # just a series, all ohlc data will be the same, volume set to 0
            formatter.data = formatter.data.to_frame(name='close')
            for col in required_col:
                formatter[col] = col
            formatter['volume'] = 0
        
        formatter = formatter.panelize()
        formatter.data.loc[:, base_col] = formatter.data.groupby(
            level=self.code_level)[base_col].ffill().fillna(0)
        
        return formatter

    def run(
        self, 
        strategy: bt.Strategy | list[bt.Strategy], 
        data: pd.DataFrame,
        *,
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
        formatter = DimFormatter(data)
        if formatter.ndims > 3 or formatter.ndims < 2:
            raise NotRequiredDimError(2)
        
        data = self._format_data(formatter).data
        start = data.index.get_level_values(self.date_level).min()
        stop = data.index.get_level_values(self.date_level).max()
        
        # add data
        more = set(data.columns.to_list()) - set(['open', 'high', 'low', 'close', 'volume'])
        PandasData = type("_PandasData", (bt.feeds.PandasData,), {"lines": tuple(more), "params": tuple(zip(more, [-1] * len(more)))})
        # without setting bt.metabase._PandasData, PandasData cannot be pickled
        bt.metabase._PandasData = PandasData

        if formatter.rowdim == 2:
            datanames = data.index.get_level_values(self.code_level).unique().to_list()
        else:
            datanames = ['data']
        
        for dn in datanames:
            d = data.xs(dn, level=self.code_level) if formatter.rowdim == 2 else data
            feed = PandasData(dataname=d, fromdate=start, todate=stop)
            self.cerebro.adddata(feed, name=dn)

        # add indicators
        indicators = indicators if isinstance(indicators, list) else [indicators]
        for indicator in indicators:
            if indicator is not None:
                self.cerebro.addindicator(indicator)
        
        # add analyzers
        analyzers = analyzers if isinstance(analyzers, list) else [analyzers]
        self.cerebro.addanalyzer(bt.analyzers.SharpeRatio, riskfreerate=self.riskfreerate)
        self.cerebro.addanalyzer(bt.analyzers.TimeDrawDown)
        self.cerebro.addanalyzer(TradeOrderRecorder, date_level=self.date_level, code_level=self.code_level)
        self.cerebro.addanalyzer(CashValueRecorder, date_level=self.date_level)
        for analyzer in analyzers:
            if analyzer is not None:
                self.cerebro.addanalyzer(analyzer)

        # add observers
        observers = observers if isinstance(observers, list) else [observers]
        self.cerebro.addobserver(bt.observers.DrawDown)
        for observer in observers:
            if observer is not None:
                self.cerebro.addobserver(observer)

        # add strategies
        strategy = [strategy] if not isinstance(strategy, list) else strategy
        if n_jobs is None:
            for strat in strategy:
                self.cerebro.addstrategy(strat, **kwargs)
        else:
            if len(strategy) > 1:
                raise ValueError("multiple strategies are not supported in optstrats mode")
            self.cerebro.optstrategy(strategy[0], **kwargs)
        
        # run strategies
        strats = self.cerebro.run(
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
            benchmark = benchmark / benchmark.dropna().iloc[0] * self.cash
            cashvalue_ = []
            for cv in cashvalue:
                cvb = pd.concat([cv, benchmark, (
                    (cv["value"].pct_change() - benchmark.pct_change())
                    .fillna(0) + 1).cumprod().to_frame(benchmark.name)
                    .add_prefix("ex(").add_suffix(')') * self.cash
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
                alpha = ab["return(value)(%)"] - (self.riskfreerate * 100 + beta * 
                    (ab[f"return({benchmark.name})(%)"] - self.riskfreerate * 100))
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
        
        if self.verbose:
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

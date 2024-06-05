import logging
import numpy as np
import pandas as pd
import backtrader as bt
import matplotlib.pyplot as plt
from pathlib import Path


class __TimeFormatter(logging.Formatter):

    def __init__(
        self, 
        display_time: bool = True,
        display_name: str = True,
        fmt: str | None = None, 
        datefmt: str | None = None, 
        style: str = "%", 
        validate: bool = True, *, 
        defaults = None
    ) -> None:
        super().__init__(fmt, datefmt, style, validate, defaults=defaults)
        self.display_time = display_time
        self.display_name = display_name


class _StreamFormatter(__TimeFormatter):
    COLORS = {
        'DEBUG': '\033[94m',
        'INFO': '\033[92m',
        'WARNING': '\033[93m',
        'ERROR': '\033[95m',
        'CRITICAL': '\033[31m',
        'RESET': '\033[0m',
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.message = record.getMessage()
        record.asctime = self.formatTime(record, self.datefmt)
        formatted_record = f'{color}'
        if self.display_time:
            formatted_record += f'[{record.asctime}] '
        if self.display_name:
            formatted_record += f'<{record.name}> '
        formatted_record += f'{record.message}{self.COLORS["RESET"]}'
        return formatted_record


class _FileFormatter(__TimeFormatter):

    def format(self, record):
        record.message = record.getMessage()
        record.asctime = self.formatTime(record, self.datefmt)
        formatted_record = ''
        if self.display_time:
            formatted_record += f'[{record.asctime}] '
        if self.display_name:
            formatted_record += f'<{record.name}> '
        formatted_record += f'|{record.levelname}| {record.message}'
        return formatted_record


class Logger(logging.Logger):

    def __init__(
        self, 
        name: str = None, 
        level: int = logging.DEBUG, 
        stream: bool = True, 
        file: str = None,
        display_time: bool = True,
        display_name: bool = False,
    ):
        name = name or 'QuoolLogger'
        super().__init__(name, level)

        if stream:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(_StreamFormatter(
                display_time=display_time, display_name=display_name
            ))
            self.addHandler(stream_handler)

        if file:
            file_handler = logging.FileHandler(file)
            file_handler.setFormatter(_FileFormatter(
                display_time=display_time, display_name=display_name
            ))
            self.addHandler(file_handler)


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
        ("minstake", 1), 
        ("minshare", 100), 
        ("divfactor", "divfactor"), 
        ("splitfactor", "splitfactor"),
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
                self.logger.debug(f"[{data.datetime.date(0)}] {data._name} dividend cash: {dividend:.2f}")
                self.add_cash(dividend)
            # hold the stock which has split
            if size and not np.isnan(splitfactor):
                splitsize = int(size * splitfactor)
                self.logger.debug(f"[{data.datetime.date(0)}] {data._name} split size: {splitsize:.0f}")
                pos.update(size=splitsize, price=data.close[0])
    
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
            self.rets = self.rets.set_index(["notify_time", "code"])
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
        
        return self

    def run(
        self, 
        strategy: Strategy | list[Strategy], 
        *,
        broker: bt.BrokerBase = None,
        minstake: int = 1,
        minshare: int = 100,
        divfactor: str = "divfactor",
        splitfactor: str = "splitfactor",
        cash: float = 1e6,
        coc: bool = False,
        commission: float = 0.005,
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
        if broker is None:
            broker = CBroker(
                minstake=minstake, minshare=minshare,
                divfactor=divfactor, splitfactor=splitfactor,
            )
        cerebro.setbroker(broker)
        cerebro.broker.set_cash(cash)
        if coc:
            self.cerebro.broker.set_coc(True)
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
        if kwargs:
            abstract = abstract.set_index(keys=list(kwargs.keys()))
        
        if verbose:
            self.logger.info(abstract)
        
        if oldimg is not None and n_jobs is None:
            if len(datanames) > 3:
                self.logger.warning(f"There are {len(datanames)} stocks, the image "
                      "may be nested and takes a long time to draw")
            figs = cerebro.plot(style='candel')
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


def parse_commastr(
    commastr: 'str | list',
) -> pd.Index:
    if isinstance(commastr, str):
        commastr = commastr.split(',')
        return list(map(lambda x: x.strip(), commastr))
    elif commastr is None:
        return None
    else:
        return commastr

def reduce_mem_usage(df: pd.DataFrame):
    logger = Logger("QuoolReduceMemUsage")
    start_mem = df.memory_usage().sum()
    logger.info('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum()
    logger.info('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    logger.info('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

def evaluate(
        value: pd.Series, 
        cash: pd.Series = None,
        turnover: pd.Series = None,
        benchmark: pd.Series = None,
        image: str = None,
        result: str = None,
    ):
        cash = cash.squeeze() if isinstance(cash, (pd.Series, pd.DataFrame)) else \
            pd.Series(np.zeros(value.shape[0]), index=value.index)
        turnover = turnover.squeeze() if isinstance(turnover, (pd.Series, pd.DataFrame)) else \
            pd.Series(np.zeros(value.shape[0]), index=value.index)
        benchmark = benchmark if isinstance(benchmark, (pd.Series, pd.DataFrame)) else \
            pd.Series(np.zeros(value.shape[0]), index=value.index)
        benchmark = benchmark.loc[value.index.intersection(benchmark.index)]
        net_value = value / value.iloc[0]
        net_cash = cash / cash.iloc[0]
        returns = value.pct_change(fill_method=None).fillna(0)
        benchmark_returns = benchmark.pct_change(fill_method=None).fillna(0)
        benchmark_returns = benchmark_returns if not benchmark_returns.isna().all() else pd.Series(np.zeros(benchmark_returns.shape[0]), index=benchmark.index)
        drawdown = net_value / net_value.cummax() - 1

        # evaluation indicators
        evaluation = pd.Series(name='evaluation')
        evaluation['total_return(%)'] = (net_value.iloc[-1] / net_value.iloc[0] - 1) * 100
        evaluation['annual_return(%)'] = ((evaluation['total_return(%)'] / 100 + 1) ** (
            365 / (value.index.max() - value.index.min()).days) - 1) * 100
        evaluation['annual_volatility(%)'] = (returns.std() * np.sqrt(252)) * 100
        down_volatility = (returns[returns < 0].std() * np.sqrt(252)) * 100
        enddate = drawdown.idxmin()
        startdate = drawdown.loc[:enddate][drawdown.loc[:enddate] == 0].index[-1]
        evaluation['max_drawdown(%)'] = (-drawdown.min()) * 100
        evaluation['max_drawdown_period(days)'] = enddate - startdate
        evaluation['max_drawdown_start'] = startdate
        evaluation['max_drawdown_stop'] = enddate
        evaluation['daily_turnover(%)'] = turnover.mean() * 100
        evaluation['sharpe_ratio'] = evaluation['annual_return(%)'] / evaluation['annual_volatility(%)'] \
            if evaluation['annual_volatility(%)'] != 0 else np.nan
        evaluation['sortino_ratio'] = evaluation['annual_return(%)'] / down_volatility \
            if down_volatility != 0 else np.nan
        evaluation['calmar_ratio'] = evaluation['annual_return(%)'] / evaluation['max_drawdown(%)'] \
            if evaluation['max_drawdown(%)'] != 0 else np.nan

        if not (benchmark==0).all():
            exreturns = returns - benchmark_returns.loc[returns.index.intersection(benchmark_returns.index)]
            benchmark_volatility = (benchmark_returns.std() * np.sqrt(252)) * 100
            exvalue = (1 + exreturns).cumprod()
            cum_benchmark_return = (1 + benchmark_returns).cumprod()
            exdrawdown = exvalue / exvalue.cummax() - 1
            evaluation['total_exreturn(%)'] = (exvalue.iloc[-1] - exvalue.iloc[0]) * 100
            evaluation['annual_exreturn(%)'] = ((evaluation['total_exreturn(%)'] / 100 + 1
                ) ** (365 / (exvalue.index.max() - exvalue.index.min()).days) - 1) * 100
            evaluation['annual_exvolatility(%)'] = (exreturns.std() * np.sqrt(252)) * 100
            enddate = exdrawdown.idxmin()
            startdate = exdrawdown.loc[:enddate][exdrawdown.loc[:enddate] == 0].index[-1]
            evaluation['ext_max_drawdown(%)'] = (exdrawdown.min()) * 100
            evaluation['ext_max_drawdown_period(days)'] = enddate - startdate
            evaluation['ext_max_drawdown_start'] = startdate
            evaluation['ext_max_drawdown_stop'] = enddate
            evaluation['beta'] = returns.cov(benchmark_returns) / benchmark_returns.var()
            evaluation['alpha(%)'] = (returns.mean() - (evaluation['beta'] * (benchmark_returns.mean()))) * 100
            evaluation['treynor_ratio(%)'] = (evaluation['annual_exreturn(%)'] / evaluation['beta'])
            evaluation['information_ratio'] = evaluation['annual_exreturn(%)'] / benchmark_volatility \
                if benchmark_volatility != 0 else np.nan
        else:
            exvalue = net_value
            exdrawdown = drawdown
            cum_benchmark_return = pd.Series(np.ones(returns.shape[0]), index=returns.index)
            
        data = pd.concat([value, net_value, exvalue, net_cash, returns, cum_benchmark_return, drawdown, exdrawdown, turnover], 
                axis=1, keys=['value', 'net_value', 'exvalue', 'net_cash', 'returns', 'benchmark', 'drawdown', 'exdrawdown', 'turnover'])
        
        if result is not None:
            data.to_excel(result, sheet_name="performances")
        
        if image is not None:
            fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
            plt.subplots_adjust(wspace=0.3, hspace=0.5)

            ax00 = data["net_value"].plot(ax=ax[0,0], title="Fund Return", color=['#1C1C1C'], legend=True)
            ax00.legend(loc='lower left')
            ax00.set_ylabel("Cumulative Return")
            ax00_twi = ax[0,0].twinx()
            ax00_twi.fill_between(data.index, 0, data['drawdown'], color='#009100', alpha=0.3)
            ax00_twi.set_ylabel("Drawdown")

            if not (benchmark==0).all():
                year = (data[['net_value', 'exvalue', 'benchmark']].resample('YE').last() - data[['net_value', 'exvalue', 'benchmark']].resample('YE').first())
            else:
                year = (data['net_value'].resample('YE').last() - data['net_value'].resample('YE').first())
            month = (data['net_value'].resample('ME').last() - data['net_value'].resample('ME').first())
            year.index = year.index.year
            year.plot(ax=ax[0,1], kind='bar', title="Yearly Return", rot=45, colormap='Paired')
            ax[0, 2].bar(month.index, month.values, width=20)
            ax[0, 2].set_title("Monthly Return")

            ax10 = data['exvalue'].plot(ax=ax[1,0], title='Extra Return', legend=True)
            ax10.legend(loc='lower left')
            ax10.set_ylabel("Cumulative Return")
            ax10_twi = ax[1,0].twinx()
            ax10_twi.fill_between(data.index, 0, data['exdrawdown'], color='#009100', alpha=0.3)
            ax10_twi.set_ylabel("Drawdown")

            data[['net_value', 'benchmark']].plot(ax=ax[1,1], title="Fund Return")

            ax12 = data['net_cash'].plot(ax=ax[1,2], title="Turnover")
            ax12.set_ylabel('net_cash')
            ax12_twi = ax[1,2].twinx()
            ax12_twi.set_ylabel('turnover')
            ax12_twi.plot(data.index, data['turnover'], color='red')

            fig.tight_layout()
            if isinstance(image, (str, Path)):
                fig.savefig(image)
            else:
                fig.show()

        return evaluation

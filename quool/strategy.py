import logging
import pandas as pd
from .util import setup_logger
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.schedulers.background import BackgroundScheduler
from .order import Order
from .source import Source
from .broker import Broker
from .evaluator import Evaluator
from .friction import FixedRateCommission, FixedRateSlippage


class Strategy:
    """Base class for trading strategies orchestrating data, broker actions, and scheduling.

    Strategy coordinates the market data Source and Broker to implement trading
    logic. It provides lifecycle hooks (init, preupdate, update, stop), blocking
    and background schedulers to run in real time, a backtest loop, and helper
    methods to place orders by target value or percent.

    Attributes:
      source (Source): Market data source providing the current time and snapshot.
      broker (Broker): Brokerage interface handling orders, executions, and portfolio state.
      logger (logging.Logger): Logger used for strategy messages.
    """

    def __init__(
        self,
        source: Source,
        broker: Broker,
        logger: logging.Logger = None,
    ):
        """Initialize a strategy with a data source and broker.

        If no logger is provided, a default logger is created with DEBUG level.

        Args:
          source (Source): Market data provider.
          broker (Broker): Execution and portfolio accounting interface.
          logger (logging.Logger, optional): Logger for diagnostics. Defaults to a
            logger named after the class with DEBUG level.
        """
        self.source = source
        self.broker = broker
        self.logger = logger or setup_logger(
            f"{self.__class__.__name__}", level="DEBUG"
        )

    def init(self, **kwargs):
        """Initialize strategy-specific state before the first run.

        Override this method to set up indicators, parameters, or any required
        state prior to live or backtest execution.

        Args:
          **kwargs: Arbitrary keyword arguments for user-defined initialization.

        Returns:
          None
        """
        pass

    def preupdate(self, **kwargs):
        """Hook executed after each run iteration but before strategy update.

        Override to perform housekeeping tasks such as logging, caching, or
        risk checks right before the core update() logic.

        Args:
          **kwargs: Arbitrary keyword arguments.

        Returns:
          None
        """
        pass

    def update(self, **kwargs):
        """Core strategy logic executed on every iteration.

        Implement trading decisions (signals, order placement, risk management)
        using self.source and self.broker.

        Args:
          **kwargs: Arbitrary keyword arguments.

        Raises:
          NotImplementedError: Must be implemented in subclasses.
        """
        raise NotImplementedError("`update` method must be implemented")

    def stop(self, **kwargs):
        """Finalize strategy execution.

        Override to release resources, persist state, or emit final logs at the
        end of a backtest or upon stopping a live session.

        Args:
          **kwargs: Arbitrary keyword arguments.

        Returns:
          None
        """
        pass

    @property
    def time(self):
        """Current strategy timestamp.

        This proxies Source.time.

        Returns:
          pandas.Timestamp: The current timestamp from the source.
        """
        return self.source.time

    @property
    def data(self):
        """Current market snapshot.

        This proxies Source.data.

        Returns:
          pandas.DataFrame or None: The current market data snapshot.
        """
        return self.source.data

    def _run(self, store: str = None, history: bool = False, **kwargs):
        """Single execution step: advance data, update broker, then strategy.

        Internally:
          1. Calls source.update(); if None is returned, halts the loop.
          2. Calls broker.update(source) and handles order notifications via notify().
          3. Executes strategy update() logic.
          4. Optionally stores broker state.

        Args:
          store (str, optional): Path to store broker state as JSON. Defaults to None.
          history (bool, optional): Whether to include deliveries and processed orders
            when storing. Defaults to False.
          **kwargs: Extra arguments forwarded to update().

        Returns:
          bool: True if the loop should continue; False if halted (e.g., no new data).

        Raises:
          ValueError: If the broker update detects invalid time or data types.
          OSError: If storing to file fails.
        """
        if self.source.update() is None:
            return False
        for notif in self.broker.update(source=self.source):
            self.notify(notif)
        self.update(**kwargs)
        if store:
            self.broker.store(store, history)
        return True

    def run(
        self,
        store: str = None,
        history: bool = False,
        trigger: str = "interval",
        trigger_kwargs: dict = None,
        **kwargs,
    ):
        """Run the strategy with a blocking scheduler.

        Schedules _run() using apscheduler's BlockingScheduler. Suitable for
        foreground execution where the process remains active until stopped.

        Args:
          store (str, optional): Path to store broker state as JSON. Defaults to None.
          history (bool, optional): Whether to include deliveries and processed orders
            when storing. Defaults to False.
          trigger (str, optional): Scheduler trigger type (e.g., 'interval', 'cron').
            Defaults to 'interval'.
          trigger_kwargs (dict, optional): Arguments for the chosen trigger, such as
            {'seconds': 30}. Defaults to {'seconds': 30} if None.
          **kwargs: Extra arguments forwarded to _run().

        Returns:
          None

        Raises:
          Exception: Any exception raised by the scheduler or by _run() will propagate,
            causing the blocking scheduler to stop.
        """
        scheduler = BlockingScheduler()
        scheduler.add_job(
            self._run,
            trigger,
            kwargs={"store": store, "history": history, **kwargs},
            id=self.__class__.__name__,
            **(trigger_kwargs or {"seconds": 30}),
        )
        scheduler.start()

    def arun(
        self,
        store: str = None,
        history: bool = False,
        trigger: str = "interval",
        trigger_kwargs: dict = None,
        scheduler: BackgroundScheduler = None,
        **kwargs,
    ):
        """Run the strategy with a background scheduler.

        Schedules _run() using apscheduler's BackgroundScheduler, allowing the
        strategy to run in the background while the caller continues executing.
        If a job with the class name exists, it is resumed; otherwise a new job
        is created.

        Args:
          store (str, optional): Path to store broker state as JSON. Defaults to None.
          history (bool, optional): Whether to include deliveries and processed orders
            when storing. Defaults to False.
          trigger (str, optional): Scheduler trigger type. Defaults to 'interval'.
          trigger_kwargs (dict, optional): Trigger arguments. Defaults to {'seconds': 30} if None.
          scheduler (BackgroundScheduler, optional): Existing scheduler to use. Defaults to a new instance.
          **kwargs: Extra arguments forwarded to _run().

        Returns:
          BackgroundScheduler: The scheduler instance managing the job.

        Raises:
          Exception: Any exception raised by the scheduler or by _run() may propagate.
        """
        scheduler = scheduler or BackgroundScheduler()
        job = scheduler.get_job(self.__class__.__name__)
        if job is not None:
            job.resume()
        else:
            scheduler.add_job(
                self._run,
                trigger,
                kwargs={"store": store, "history": history, **kwargs},
                id=self.__class__.__name__,
                **(trigger_kwargs or {"seconds": 30}),
            )
        scheduler.start()
        return scheduler

    def backtest(self, benchmark: pd.Series = None, history: bool = False, **kwargs):
        """Run a backtest loop until the data source is exhausted.

        Workflow:
          - Calls init(**kwargs) once at the start.
          - Repeatedly calls _run(history=history, **kwargs); after each iteration,
            calls preupdate(**kwargs).
          - Calls stop(**kwargs) when the loop ends.
          - Returns the evaluation summary.

        Args:
          benchmark (pandas.Series, optional): Benchmark index or asset for comparison.
          history (bool, optional): Whether to store history during backtest. Defaults to False.
          **kwargs: Additional arguments forwarded to init, preupdate, and update.

        Returns:
          Any: Evaluation summary as produced by Evaluator.report().

        Raises:
          ValueError: If evaluation is attempted without delivery data.
        """
        self.init(**kwargs)
        while self._run(history=history, **kwargs):
            self.preupdate(**kwargs)
        self.stop(**kwargs)
        return self.evaluate(benchmark=benchmark)

    def __str__(self) -> str:
        return (
            f"{self.__class__}@{self.time}\n"
            f"Broker:\n{self.broker}\n"
            f"Source:\n{self.source}\n"
        )

    def __repr__(self):
        return self.__str__()

    def log(self, message: str, level: str = "DEBUG"):
        """Log a message with a given level, prefixed by the current time.

        Args:
          message (str): The message to log.
          level (str, optional): Logging level name (e.g., 'DEBUG', 'INFO', 'WARNING').
            Defaults to 'DEBUG'.

        Returns:
          None

        Raises:
          KeyError: If the level name is not recognized by logging.getLevelNamesMapping().
        """
        self.logger.log(
            logging.getLevelNamesMapping().get(level, 0),
            f"[{self.source.time}]: {message}",
        )

    def notify(self, order: Order):
        """Notification hook for order status changes.

        Default behavior logs the order. Override to handle events such as
        filled orders, rejections, or cancellations.

        Args:
          order (Order): The order that changed status.

        Returns:
          None
        """
        self.log(order)

    def get_value(self):
        """Get current portfolio value from the broker.

        Uses broker.get_value(source) with the source's close prices.

        Returns:
          float: Total portfolio value (positions mark-to-market + cash).
        """
        return self.broker.get_value(self.source)

    def get_positions(self):
        """Get current positions from the broker.

        Returns:
          pandas.Series: Series indexed by instrument code with quantities.
        """
        return self.broker.get_positions()

    def buy(
        self,
        code: str,
        quantity: int,
        exectype: str = Order.MARKET,
        limit: float = None,
        trigger: float = None,
        id: str = None,
        valid: str = None,
    ) -> Order:
        """Create and submit a BUY order via the broker.

        Args:
          code (str): Instrument code.
          quantity (int): Requested quantity.
          exectype (str, optional): Execution type. Defaults to Order.MARKET.
          limit (float, optional): Limit price for LIMIT/STOPLIMIT orders. Defaults to None.
          trigger (float, optional): Trigger price for STOP/STOPLIMIT orders. Defaults to None.
          id (str, optional): Order identifier. Defaults to None (auto-generated).
          valid (str, optional): Good-till timestamp as a pandas-parsable string. Defaults to None.

        Returns:
          Order: The submitted BUY order.

        Raises:
          ValueError: If broker time is not initialized.
        """
        return self.broker.create(
            type=self.broker.order_type.BUY,
            code=code,
            quantity=quantity,
            exectype=exectype,
            limit=limit,
            trigger=trigger,
            id=id,
            valid=valid,
        )

    def sell(
        self,
        code: str,
        quantity: int,
        exectype: str = Order.MARKET,
        limit: float = None,
        trigger: float = None,
        id: str = None,
        valid: str = None,
    ) -> Order:
        """Create and submit a SELL order via the broker.

        Args:
          code (str): Instrument code.
          quantity (int): Requested quantity.
          exectype (str, optional): Execution type. Defaults to Order.MARKET.
          limit (float, optional): Limit price for LIMIT/STOPLIMIT orders. Defaults to None.
          trigger (float, optional): Trigger price for STOP/STOPLIMIT orders. Defaults to None.
          id (str, optional): Order identifier. Defaults to None (auto-generated).
          valid (str, optional): Good-till timestamp as a pandas-parsable string. Defaults to None.

        Returns:
          Order: The submitted SELL order.

        Raises:
          ValueError: If broker time is not initialized.
        """
        return self.broker.create(
            type=self.broker.order_type.SELL,
            code=code,
            quantity=quantity,
            exectype=exectype,
            limit=limit,
            trigger=trigger,
            id=id,
            valid=valid,
        )

    def order_target_value(
        self,
        code: str,
        value: float,
        exectype: str = Order.MARKET,
        limit: float = None,
        trigger: float = None,
        id: str = None,
        valid: str = None,
    ) -> Order:
        """Order to adjust a position to a target notional value.

        Computes the difference between desired value and current mark-to-market
        value for the instrument, then submits a BUY or SELL order for the required
        quantity. If the instrument is not present in the source data index, no order
        is placed and None is returned.

        Args:
          code (str): Instrument code.
          value (float): Target position value in currency units.
          exectype (str, optional): Execution type. Defaults to Order.MARKET.
          limit (float, optional): Limit price for LIMIT/STOPLIMIT orders. Defaults to None.
          trigger (float, optional): Trigger price for STOP/STOPLIMIT orders. Defaults to None.
          id (str, optional): Order identifier. Defaults to None (auto-generated).
          valid (str, optional): Good-till timestamp as a pandas-parsable string. Defaults to None.

        Returns:
          Order or None: The submitted order if an adjustment is needed; otherwise None.
        """
        if code not in self.source.data.index:
            quantity = 0
            type = self.broker.order_type.BUY
        else:
            delta = (
                value
                - self.get_positions().get(code, 0)
                * self.source.data.loc[code, "close"]
            )
            quantity = delta / self.source.data.loc[code, "close"]
        if quantity > 0:
            type = self.broker.order_type.BUY
        elif quantity < 0:
            type = self.broker.order_type.SELL
        if abs(quantity) > 0:
            return self.broker.create(
                type=type,
                code=code,
                quantity=abs(quantity),
                exectype=exectype,
                limit=limit,
                trigger=trigger,
                id=id,
                valid=valid,
            )

    def order_target_percent(
        self,
        code: str,
        percent: float,
        exectype: str = Order.MARKET,
        limit: float = None,
        trigger: float = None,
        id: str = None,
        valid: str = None,
    ) -> Order:
        """Order to adjust a position to a target portfolio percentage.

        Converts the desired percentage of total portfolio value into a target
        value and delegates to order_target_value().

        Args:
          code (str): Instrument code.
          percent (float): Target fraction of portfolio value (e.g., 0.10 for 10%).
          exectype (str, optional): Execution type. Defaults to Order.MARKET.
          limit (float, optional): Limit price for LIMIT/STOPLIMIT orders. Defaults to None.
          trigger (float, optional): Trigger price for STOP/STOPLIMIT orders. Defaults to None.
          id (str, optional): Order identifier. Defaults to None (auto-generated).
          valid (str, optional): Good-till timestamp as a pandas-parsable string. Defaults to None.

        Returns:
          Order or None: The submitted order if an adjustment is needed; otherwise None.
        """
        value = self.get_value() * percent
        return self.order_target_value(
            code=code,
            value=value,
            exectype=exectype,
            limit=limit,
            trigger=trigger,
            id=id,
            valid=valid,
        )

    def close(
        self,
        code: str,
        exectype: str = Order.MARKET,
        limit: float = None,
        trigger: float = None,
        id: str = None,
        valid: str = None,
    ) -> Order:
        """Close the entire position in the given instrument.

        If a position exists for the specified code, submits a SELL order for
        the full position size. If no position exists, returns None.

        Args:
          code (str): Instrument code.
          exectype (str, optional): Execution type. Defaults to Order.MARKET.
          limit (float, optional): Limit price for LIMIT/STOPLIMIT orders. Defaults to None.
          trigger (float, optional): Trigger price for STOP/STOPLIMIT orders. Defaults to None.
          id (str, optional): Order identifier. Defaults to None (auto-generated).
          valid (str, optional): Good-till timestamp as a pandas-parsable string. Defaults to None.

        Returns:
          Order or None: The submitted close order if a position exists; otherwise None.
        """
        if code in self.broker.positions.keys():
            return self.broker.create(
                type=self.broker.order_type.SELL,
                code=code,
                quantity=self.broker.positions[code],
                exectype=exectype,
                limit=limit,
                trigger=trigger,
                id=id,
                valid=valid,
            )

    def evaluate(self, benchmark: pd.Series = None):
        """Evaluate strategy performance using broker deliveries.

        Requires non-empty delivery data; otherwise raises a ValueError.

        Args:
          benchmark (pandas.Series, optional): Benchmark series for comparison.

        Returns:
          Any: Evaluation summary from Evaluator.report().

        Raises:
          ValueError: If no delivery data is available.
        """
        if self.broker.get_delivery().empty:
            raise ValueError("No delivery data available")
        summary = Evaluator(self.broker, self.source).report(benchmark)
        return summary

    def cancel(self, order_or_id: str | Order) -> Order:
        """Cancel an existing order by object or identifier.

        Delegates to broker.cancel().

        Args:
          order_or_id (str | Order): The order object or its unique identifier.

        Returns:
          Order: The canceled order (status set to CANCELED if it was open).

        Raises:
          KeyError: If an order id is provided but cannot be found.
        """
        return self.broker.cancel(order_or_id)

    def dump(self, history: bool = True):
        """Serialize broker state for persistence or inspection.

        Delegates to broker.dump().

        Args:
          history (bool, optional): Include deliveries and processed orders. Defaults to True.

        Returns:
          dict: Serialized broker state.
        """
        return self.broker.dump(history=history)

    @classmethod
    def load(
        cls,
        data: dict,
        commssion: FixedRateCommission,
        slippage: FixedRateSlippage,
        source: Source,
        logger: logging.Logger = None,
    ):
        """Reconstruct a Strategy from serialized broker state and runtime components.

        Uses Broker.load(data, commission, slippage) to reconstruct the broker and
        returns a Strategy bound to the provided source and logger.

        Args:
          data (dict): Serialized broker state as produced by broker.dump().
          commssion (FixedRateCommission): Commission model to use.
          slippage (FixedRateSlippage): Slippage model to use.
          source (Source): Market data source to attach to the strategy.
          logger (logging.Logger, optional): Logger for diagnostics. Defaults to None.

        Returns:
          Strategy: The reconstructed strategy instance.

        Raises:
          KeyError: If required fields in data are missing.
          ValueError: If date-time fields cannot be parsed.
        """
        broker = Broker.load(data, commssion, slippage)
        return cls(broker, source, logger)

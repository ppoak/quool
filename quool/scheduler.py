import logging
import pandas as pd
from .util import setup_logger


class Scheduler:
    """Multi-strategy orchestrator coordinating backtest across multiple strategies.

    Scheduler aggregates multiple Strategy instances, forwarding each one by one
    at every time step. It provides lifecycle hooks (init, preupdate, update, stop),
    a backtest loop that advances the shared source and broker, and logging utilities.

    Attributes:
      strategies (tuple[Strategy]): Strategies managed by this scheduler.
      source (Source): Market data source (set dynamically before running).
      broker (Broker): Broker for all strategies (set dynamically before running).
      logger (logging.Logger): Logger used for diagnostic messages.

    Example:
        >>> scheduler = Scheduler(strategy1, strategy2)
        >>> scheduler.source = some_source
        >>> scheduler.broker = some_broker
        >>> results = scheduler.backtest(benchmark=benchmark_series)
    """

    def __init__(self, *strategies, logger: logging.Logger = None):
        self.strategies = strategies
        for strategy in strategies:
            setattr(self, strategy.__name__.lower(), strategy)
        self.logger = logger or setup_logger(self.__class__.__name__, level="DEBUG")

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
        """Core scheduler logic executed on every iteration.

        Implement multi-strategy coordination logic (e.g., risk checks across
        strategies) using self.source and self.broker.

        Args:
          **kwargs: Arbitrary keyword arguments.

        Raises:
          NotImplementedError: Must be implemented in subclasses.
        """
        raise NotImplementedError("`update` method must be implemented")

    def stop(self, **kwargs):
        """Finalize scheduler execution.

        Override to release resources, persist state, or emit final logs at the
        end of a backtest or upon stopping a live session.

        Args:
          **kwargs: Arbitrary keyword arguments.

        Returns:
          None
        """
        pass

    def _run(self, store: str = None, history: bool = False, **kwargs):
        """Single execution step: advance data, update broker, then scheduler.

        Internally:
          1. Calls source.update(); if None is returned, halts the loop.
          2. Calls broker.update(source) and handles order notifications via notify().
          3. Executes scheduler update() logic.
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
        for notif in self.broker.update(time=self.source.time, data=self.data):
            self.notify(notif)
        self.update(**kwargs)
        if store:
            self.broker.store(store, history)
        return True

    def backtest(self, benchmark: pd.Series, history: bool = False, **kwargs):
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
        return f"{self.__class__}(\n" + "\n".join(self.strategies) + "\n)"

    def __repr__(self):
        return self.__str__()

    def log(self, message: str, level: str = "DEBUG"):
        self.logger.log(
            logging.getLevelNamesMapping().get(level, 0),
            f"[{self.source.time}]: {message}",
        )

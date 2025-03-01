import logging
import pandas as pd
from .order import Order
from .source import Source
from .broker import Broker
from .util import setup_logger
from .evaluator import report, evaluate
from joblib import Parallel, delayed


class Strategy:

    def __init__(
        self,
        source: Source,
        broker: Broker,
        logger: logging.Logger = None,
    ):
        self.source = source
        self.broker = broker
        self.logger = logger or setup_logger(
            f"{self.__class__.__name__}", level="DEBUG"
        )
        self.data = None

    def init(self, **kwargs):
        pass

    def preupdate(self, **kwargs):
        pass

    def update(self, **kwargs):
        raise NotImplementedError("`update` method must be implemented")

    def stop(self, **kwargs):
        pass

    @property
    def time(self):
        return self.source.time

    def run(self, **kwargs):
        self.data = self.source.update()
        if self.data is None:
            return False
        for notif in self.broker.update(time=self.source.time, data=self.data):
            self.notify(notif)
        self.update(**kwargs)
        return True

    def backtest(self, benchmark: pd.Series = None, **kwargs):
        self.init(**kwargs)
        while self.run(**kwargs):
            self.preupdate(**kwargs)
        self.stop(**kwargs)
        self.evaluate(benchmark=benchmark)

    def __call__(self, params: dict, since: pd.Timestamp = None, n_jobs: int = -1):
        grid_params = pd.MultiIndex.from_product(
            [param for param in params.values()],
            names=[name for name in params.keys()],
        )
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(self.backtest)(param, path, since) for path, param in params.items()
        )
        evaluations = pd.concat([result["evaluation"] for result in results])
        evaluations.index = grid_params
        return evaluations

    def __str__(self) -> str:
        return (
            f"{self.__class__}({self.id})@{self.status}\n"
            f"Broker:\n{self.broker}\n"
            f"Source:\n{self.source}\n"
        )

    def __repr__(self):
        return self.__str__()

    def log(self, message: str, level: str = "DEBUG"):
        self.logger.log(
            logging.getLevelNamesMapping().get(level, 0),
            f"[{self.source.time}]: {message}",
        )

    def notify(self, notif: Order):
        self.log(notif)

    def get_value(self):
        return self.broker.get_value(self.source.data)

    def get_positions(self):
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
        delta = (
            value
            - self.get_positions().get(code, 0) * self.source.data.loc[code, "close"]
        )
        quantity = (delta / self.source.data.loc[code, "close"] // 100) * 100
        if quantity > 0:
            type = self.broker.order_type.BUY
        elif quantity < 0:
            type = self.broker.order_type.SELL
        if abs(quantity):
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
        return self.broker.create(
            type=self.broker.order_type.SELL,
            code=code,
            quantity=self.broker.positions.get(code),
            exectype=exectype,
            limit=limit,
            trigger=trigger,
            id=id,
            valid=valid,
        )

    def evaluate(self, benchmark: pd.Series = None):
        if self.broker.get_delivery().empty:
            raise ValueError("No delivery data available")
        summary = report(self.broker.get_delivery(), self.source.datas)
        evaluation = evaluate(
            summary["values"]["total"],
            benchmark=benchmark,
            turnover=summary["values"]["turnover"],
            trades=summary["trades"],
        )
        summary["evaluation"] = evaluation
        return summary

    def cancel(self, order_or_id: str | Order) -> Order:
        return self.broker.cancel(order_or_id)

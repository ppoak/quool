import logging
from .util import setup_logger


class Scheduler:

    def __init__(*strategies, logger: logging.Logger = None):
        self.strategies = strategies
        for strategy in strategies:
            setattr(self, strategy.__name__.lower(), strategy)
        self.logger = logger or setup_logger(self.__class__.__name__, level="DEBUG")

    def init(self, **kwargs):
        pass

    def preupdate(self, **kwargs):
        pass

    def update(self, **kwargs):
        raise NotImplementedError("`update` method must be implemented")

    def stop(self, **kwargs):
        pass

    def _run(self, store: str = None, history: bool = False, **kwargs):
        if self.source.update() is None:
            return False
        for notif in self.broker.update(time=self.source.time, data=self.data):
            self.notify(notif)
        self.update(**kwargs)
        if store:
            self.broker.store(store, history)
        return True

    def backtest(self, **kwargs):
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

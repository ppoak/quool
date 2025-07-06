import pandas as pd


class Source:

    def __init__(
        self,
        time: pd.Timestamp,
        data: pd.DataFrame = None,
        open: str = "open",
        high: str = "high",
        low: str = "low",
        close: str = "close",
        volume: str = "volume",
    ):
        self._time = time
        self._data = data
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

    @property
    def time(self):
        return self._time

    @property
    def data(self):
        return self._data

    def update(self) -> pd.DataFrame:
        raise NotImplementedError

    def __str__(self):
        return f"{self.__class__.__name__}@[{self.time}]"

    def __repr__(self):
        return super().__repr__()

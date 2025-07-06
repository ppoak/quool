import pandas as pd
from quool import Source


class DataFrameSource(Source):

    def __init__(
        self,
        data: pd.DataFrame,
        open: str = "open",
        high: str = "high",
        low: str = "low",
        close: str = "close",
        volume: str = "volume",
    ):
        self._times = data.index.get_level_values(0).unique().sort_values()
        super().__init__(self._times.min(), data, open, high, low, close, volume)

    @property
    def times(self):
        return self._times[self._times <= self.time]

    @property
    def datas(self):
        return self._data.loc[: self.time]

    @property
    def data(self):
        return self._data.loc[self.time]

    def update(self) -> pd.DataFrame:
        future = self._times[self._times > self.time]
        if future.empty:
            return None  #
        self._time = future.min()
        data = self._data.loc[self._time]
        return data

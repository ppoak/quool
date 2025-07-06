import pandas as pd
from quool import Source
from pathlib import Path
from .util import DuckDBManager
from .dataframe import DataFrameSource


class DuckPreloadSource(DataFrameSource):

    def __init__(
        self,
        path: str | Path,
        table: str,
        begin: str,
        end: str,
        time_col: str = "time",
        code_col: str = "code",
        open: str = "open",
        high: str = "high",
        low: str = "low",
        close: str = "close",
        volume: str = "volume",
    ):
        self.table = table
        begin = pd.to_datetime(begin)
        end = pd.to_datetime(end)
        self.manager = DuckDBManager(path, read_only=True)
        super().__init__(
            self.manager.read(
                table,
                columns=[
                    f"{time_col} AS time",
                    f"{code_col} AS code",
                    f"{open} AS open",
                    f"{high} AS high",
                    f"{low} AS low",
                    f"{close} AS close",
                    f"{volume} AS volume",
                ],
                filters={f"{time_col}__ge": begin, f"{time_col}__le": end},
            ).set_index(["time", "code"]),
        )


class DuckSource(Source):

    def __init__(
        self,
        path: str | Path,
        table: str,
        begin: pd.Timestamp | str,
        end: pd.Timestamp | str,
        time_col: str = "time",
        code_col: str = "code",
        open: str = "open",
        high: str = "high",
        low: str = "low",
        close: str = "close",
        volume: str = "volume",
    ):
        self.table = table
        begin = pd.to_datetime(begin)
        end = pd.to_datetime(end)
        self.time_col = time_col
        self.code_col = code_col
        self.manager = DuckDBManager(path, read_only=True)
        self._times = self.manager.read(
            table,
            columns=[time_col],
            distinct=True,
            filters={f"{time_col}__ge": begin, f"{time_col}__le": end},
        ).squeeze()
        super().__init__(self._times.min(), None, open, high, low, close, volume)

    @property
    def times(self):
        return self._times

    @property
    def time(self):
        return self._time

    @property
    def data(self):
        return self.manager.read(
            self.table,
            columns=[
                f"{self.code_col} AS code",
                f"{self.open} AS open",
                f"{self.high} AS high",
                f"{self.low} AS low",
                f"{self.close} AS close",
                f"{self.volume} AS volume",
            ],
            filters={f"{self.time_col}": self._time},
        ).set_index(["code"])

    @property
    def datas(self):
        return self.manager.read(
            self.table,
            columns=[
                f"{self.time_col} AS time",
                f"{self.code_col} AS code",
                f"{self.open} AS open",
                f"{self.high} AS high",
                f"{self.low} AS low",
                f"{self.close} AS close",
                f"{self.volume} AS volume",
            ],
            filters={
                f"{self.time_col}__ge": self.begin,
                f"{self.time_col}__le": self.end,
            },
        ).set_index(["time", "code"])

    def update(self):
        future = self._times[self._times > self._time]
        if future.empty:
            return None
        self._time = future.min()
        return self.data

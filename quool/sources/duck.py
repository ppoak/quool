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
        columns: list[str] = None,
        time_col: str = "time",
        code_col: str = "code",
        open: str = "open",
        high: str = "high",
        low: str = "low",
        close: str = "close",
        volume: str = "volume",
    ):
        self.table = table
        self.begin = pd.to_datetime(begin)
        self.end = pd.to_datetime(end)
        self.manager = DuckDBManager(path, read_only=True)
        super().__init__(
            self.manager.select(
                table,
                columns=[
                    f"{time_col} AS time",
                    f"{code_col} AS code",
                    f"{open} AS open",
                    f"{high} AS high",
                    f"{low} AS low",
                    f"{close} AS close",
                    f"{volume} AS volume",
                ]
                + (columns or []),
                ands=[f"{time_col} >= ?", f"{time_col} <= ?"],
                params=(self.begin, self.end),
            ).set_index(["time", "code"]).sort_index(),
        )


class DuckSource(Source):

    def __init__(
        self,
        path: str | Path,
        table: str,
        begin: pd.Timestamp | str,
        end: pd.Timestamp | str,
        columns: list[str] = None,
        time_col: str = "time",
        code_col: str = "code",
        open: str = "open",
        high: str = "high",
        low: str = "low",
        close: str = "close",
        volume: str = "volume",
    ):
        self.table = table
        self.begin = pd.to_datetime(begin)
        self.end = pd.to_datetime(end)
        self.columns = columns
        self.time_col = time_col
        self.code_col = code_col
        self.manager = DuckDBManager(path, read_only=True)
        self._times = pd.to_datetime(
            self.manager.select(
                table,
                columns=[time_col],
                distinct=True,
                orderby=[time_col],
                ands=[f"{time_col} >= ?", f"{time_col} <= ?"],
                params=(self.begin, self.end),
            )
            .squeeze()
            .to_list()
        )
        super().__init__(self._times.min(), None, open, high, low, close, volume)

    @property
    def times(self):
        return self._times

    @property
    def time(self):
        return self._time

    @property
    def data(self):
        return self.manager.select(
            self.table,
            columns=[
                f"{self.code_col} AS code",
                f"{self.open} AS open",
                f"{self.high} AS high",
                f"{self.low} AS low",
                f"{self.close} AS close",
                f"{self.volume} AS volume",
            ]
            + (self.columns or []),
            ands=[f"{self.time_col} = ?"],
            params=(self._time,),
        ).set_index(["code"])

    @property
    def datas(self):
        return self.manager.select(
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
            ands=[f"{self.time_col} >= ?", f"{self.time_col} <= ?"],
            params=(self.begin, self.end),
        ).set_index(["time", "code"])

    def update(self):
        future = self._times[self._times > self._time]
        if future.empty:
            return None
        self._time = future.min()
        return self.data

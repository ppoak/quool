from quool import Source
from parquool import DuckParquet


class DuckParquetSource(Source):

    def __init__(
        self,
        manager: DuckParquet,
        begin: str,
        end: str,
        extra: list = None,
        date_col: str = "date",
        code_col: str = "code",
        adj_col: str = "adjfactor",
    ):
        self.manager = manager
        self._times = self.manager.select(
            self,
            columns=[date_col],
            where=f"{date_col} >= ? and {date_col} <= ?",
            params=[begin, end],
        ).index.unique()
        self._time = self._times.min()
        self.extra = extra or []
        self.fields = ["open", "high", "low", "close", "volume"] + self.extra
        self.date_col = date_col
        self.code_col = code_col
        self.adj_col = adj_col

    @property
    def data(self):
        if self.adj_col:
            data = self.manager.select(
                columns=[self.code_col] + self.fields + [self.adj_col],
                where=f"{self.date_col} = ?",
                params=[self._time],
            )
            multipler = ["open", "high", "low", "close"]
            data[multipler] = data[multipler].mul(data[self.adj_col], axis=0)
            data = data.drop(columns=self.adj_col)
        else:
            data = self.manager.select(
                index=self.code_col,
                columns=self.fields,
                **{self.date_col: self._time},
            )
        return data

    @property
    def datas(self):
        if self.adj_col:
            data = self.manager.select(
                columns=[self.code_col] + self.fields + [self.adj_col],
                where=f"{self.date_col} = ?",
                params=[self._time],
            )
            multipler = ["open", "high", "low", "close"]
            data[multipler] = data[multipler].mul(data[self.adj_col], axis=0)
            data = data.drop(columns=self.adj_col)
        else:
            data = self.manager.select(
                columns=[self.code_col] + self.fields + [self.adj_col],
                where=f"{self.date_col} = ?",
                params=[self._time],
            )
        return data

    def update(self):
        future = self._timepoint[self._timepoint > self.time]
        if future.empty:
            return None
        self._time = future.min()
        return self.data

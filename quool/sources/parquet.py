from quool import Source
from .util import ParquetManager


class ParquetSource(Source):

    def __init__(
        self,
        manager: ParquetManager,
        begin: str,
        end: str,
        extra: list = None,
        date_col: str = "date",
        code_col: str = "code",
        adj_col: str = "adjfactor",
    ):
        self.manager = manager
        self._times = self.manager.read(
            self, index=date_col, **{f"{date_col}__ge": begin, f"{date_col}__le": end}
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
            data = ParquetManager.read(
                self,
                index=self.code_col,
                columns=self.fields + [self.adj_col],
                **{self.date_col: self._time},
            )
            multipler = ["open", "high", "low", "close"]
            data[multipler] = data[multipler].mul(data[self.adj_col], axis=0)
            data = data.drop(columns=self.adj_col)
        else:
            data = ParquetManager.read(
                self,
                index=self.code_col,
                columns=self.fields,
                **{self.date_col: self._time},
            )
        return data

    @property
    def datas(self):
        if self.adj_col:
            data = ParquetManager.read(
                self,
                index=self.code_col,
                columns=self.fields + [self.adj_col],
                **{f"{self.date_col}__ge": self._time},
            )
            multipler = ["open", "high", "low", "close"]
            data[multipler] = data[multipler].mul(data[self.adj_col], axis=0)
            data = data.drop(columns=self.adj_col)
        else:
            data = ParquetManager.read(
                self,
                index=self.code_col,
                columns=self.fields,
                **{f"{self.date_col}__ge": self._time},
            )
        return data

    def update(self):
        future = self._timepoint[self._timepoint > self.time]
        if future.empty:
            return None
        self._time = future.min()
        return self.data

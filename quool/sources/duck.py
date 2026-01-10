from typing import Optional, List, Dict, Tuple

import pandas as pd

from quool import Source
from parquool import DuckPQ


def parse_factor_path(path: str, sep: str = "/") -> Tuple[str, str]:
    if not isinstance(path, str):
        raise TypeError(f"factor path must be str, got {type(path)}: {path}")
    s = path.strip()
    if not s:
        raise ValueError("empty factor path")
    parts = s.split(sep)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"invalid factor path: {path!r}, expected 'table{sep}factor'")
    return parts[0], parts[1]


class DuckPQSource(Source):

    def __init__(
        self,
        source: DuckPQ,
        begin: str,
        end: str,
        datetime_col: str = "date",
        code_col: str = "code",
        bar: Optional[Dict[str, str]] = None,
        extra: Optional[Dict[str, str]] = None,
        sep: str = "/",
    ):
        self.source = source
        self.sep = sep
        self.datetime_col = datetime_col
        self.code_col = code_col
        if bar is None:
            self.bar = {
                "open": f"target{self.sep}open_post",
                "high": f"target{self.sep}high_post",
                "low": f"target{self.sep}low_post",
                "close": f"target{self.sep}close_post",
                "volume": f"target{self.sep}volume",
            }
        else:
            self.bar = bar
        if extra is None:
            self.extra = {}
        else:
            self.extra = extra

        # Parse data paths and group requested columns by table.
        self._by_table: Dict[str, List[str]] = {}
        for k, p in self.bar.items():
            t, c = parse_factor_path(p, sep=self.sep)
            self._by_table.setdefault(t, [])
            if c not in self._by_table[t]:
                self._by_table[t].append(f"{c} AS {k}")
            if len(self._by_table) > 1:
                raise ValueError("Bar data should be in same table!")
        self._base_table = list(self._by_table.keys())[0]
        for k, p in self.extra.items():
            t, c = parse_factor_path(p, sep=self.sep)
            self._by_table.setdefault(t, [])
            if c not in self._by_table[t]:
                self._by_table[t].append(f"{c} AS {k}")

        tables = list(self._by_table.keys())
        for table in tables:
            self.source.register(table)

        begin = pd.to_datetime(begin)
        end = pd.to_datetime(end)
        self._times = self.source.query(
            f"""
            SELECT DISTINCT {self.datetime_col} AS datetime
            FROM {self._base_table} 
            WHERE {self.datetime_col} >= '{begin.date()}'
                AND {self.datetime_col} <= '{end.date()}'
        """.strip()
        ).iloc[:, 0]
        self._datas = []
        self._data = None
        self._time = self._times.min() - pd.Timedelta(days=1)
        self.update()

    @property
    def data(self):
        """Return the current market snapshot.

        Returns:
          pandas.DataFrame or None: The current data frame of market prices and
          volume. May be None if no snapshot is available.
        """
        return self._data

    @property
    def datas(self):
        """Return the data at the current time with the requested fields.

        Returns:
          pandas.DataFrame: Data snapshot at the current time with requested fields.
        """
        return pd.concat(self._datas, axis=0)

    def update(self):
        """Advance to the next available timestamp and return the data snapshot.

        Moves the current time forward to the next timestamp in the available series.

        Returns:
          pandas.DataFrame or None: Data at the new current time. Returns None if there
          are no future timestamps to advance to.
        """
        future = self._times[self._times > self.time]
        if future.empty:
            return None
        self._time = future.min()

        # Build a single SQL statement joining per-table subqueries.
        def subquery(table: str) -> str:
            cols = ", ".join(self._by_table[table])
            return f"""
                SELECT
                    CAST({self.datetime_col} AS TIMESTAMP) AS datetime,
                    {self.code_col} AS code,
                    {cols}
                FROM {table}
                WHERE datetime >= '{self._time.date()}' AND datetime <= '{self._time.date()}'
            """.strip()

        tables = list(self._by_table.keys())
        base_alias = "b"
        sql_from = f"FROM ({subquery(self._base_table)}) AS {base_alias}\n"
        key_time = f"{base_alias}.datetime"
        key_code = f"{base_alias}.code"

        i = 0
        for t in tables:
            if t == self._base_table:
                continue
            i += 1
            a = f"t{i}"
            sql_from += (
                f"LEFT JOIN ({subquery(t)}) AS {a}\n"
                f"ON {a}.datetime = {key_time} AND {a}.code = {key_code}\n"
            )

        select_cols: List[str] = [f"{key_time} AS datetime", f"{key_code} AS code"]
        for c in self._by_table[self._base_table]:
            select_cols.append(c.split(" AS ")[-1])
        i = 0
        for t in tables:
            if t == self._base_table:
                continue
            i += 1
            a = f"t{i}"
            for c in self._by_table[t]:
                select_cols.append(c.split(" AS ")[-1])

        sql = "SELECT\n    " + ",\n    ".join(select_cols) + "\n" + sql_from
        df = self.source.query(sql)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.set_index(["datetime", "code"]).sort_index()
        self._data = df
        self._datas.append(self._data)
        return df

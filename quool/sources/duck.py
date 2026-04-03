from typing import Optional, List, Dict, Tuple

import pandas as pd

from quool import Source
from quool import DuckPQ


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
    """Market data source backed by DuckDB/Parquet storage.

    DuckPQSource queries a DuckPQ store for OHLCV bars and optional extra fields
    within a specified date range. It advances time through available timestamps
    and returns per-timestamp snapshots suitable for broker-driven backtesting.

    Attributes:
      source (DuckPQ): handle to the DuckDB/Parquet store.
      sep (str): Path separator used in factor paths.
      datetime_col (str): Column name for the datetime field in the database.
      code_col (str): Column name for the instrument code field in the database.
      bar (dict[str, str]): Mapping from OHLCV field name to factor path.
      extra (dict[str, str]): Mapping from extra field name to factor path.
      times (pandas.Index): Available timestamps within the configured date range.
      limit (int or None): Maximum number of historical time snapshots visible in
        `datas`. None or -1 means unlimited. When set, `datas` only queries the
        most recent `limit` timestamps, bounding memory usage.
      data (pandas.DataFrame or None): Current snapshot at the current time.
    """

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
        limit: Optional[int] = None,
    ):
        """Initialize a DuckPQ-backed source and preload available timestamps.

        Registers required tables with the DuckPQ store, queries distinct timestamps
        in the [begin, end] window, sets the initial current time to one day before
        begin, and performs an initial update to populate the data snapshot.

        Args:
          source (DuckPQ): DuckPQ instance connected to the data store.
          begin (str): Start date/time for the data window (pandas-parsable).
          end (str): End date/time for the data window (pandas-parsable).
          datetime_col (str, optional): Name of the datetime column in the DB.
            Defaults to "date".
          code_col (str, optional): Name of the instrument code column in the DB.
            Defaults to "code".
          bar (dict[str, str], optional): Mapping of OHLCV field names to factor paths
            in the form "table/column". Defaults to target/open_post, target/high_post,
            target/low_post, target/close_post, target/volume.
          extra (dict[str, str], optional): Additional fields to fetch, mapping name
            to factor path. Defaults to {}.
          sep (str, optional): Separator used in factor paths. Defaults to "/".
          limit (int, optional): Maximum number of historical time snapshots visible in
            `datas`. Defaults to None (unlimited). -1 also means unlimited.

        Raises:
          ValueError: If bar data spans multiple tables (all bar fields must be in one table).
          Exception: Any errors raised by the DuckPQ store during registration or querying.
        """
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
            SELECT DISTINCT CAST({self.datetime_col} AS DATE) AS datetime
            FROM {self._base_table}
            WHERE CAST({self.datetime_col} AS DATE) >= '{begin.date()}'
                AND CAST({self.datetime_col} AS DATE) <= '{end.date()}'
        """.strip()
        ).iloc[:, 0]
        super().__init__(
            self._times.min() - pd.Timedelta(days=1),
            None,
            "open",
            "high",
            "low",
            "close",
            "volume",
        )
        self.limit = limit if limit != -1 else None
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
        """Return historical data up to the current time.

        Queries DuckPQ directly for a bounded time window, avoiding in-memory
        accumulation of snapshots. When ``limit`` is set, only the most recent
        ``limit`` timestamps are included; -1 or None means unlimited.

        Returns:
          pandas.DataFrame: Data indexed by (datetime, code).
        """
        # Determine the query window based on limit
        visible = self._times[self._times <= self.time]
        if self.limit is not None:
            visible = visible[-self.limit:]

        if visible.empty:
            start = end = self.time
        else:
            start, end = visible.min(), visible.max()

        # Build column specs in "table/field AS alias" format from _by_table
        col_specs: List[str] = []
        for table, col_aliases in self._by_table.items():
            for ca in col_aliases:
                col_specs.append(f"{table}{self.sep}{ca}")

        where = (
            f"CAST({self.datetime_col} AS DATE) >= '{start.date()}' AND CAST({self.datetime_col} AS DATE) <= '{end.date()}'"
        )
        return (
            self.source.load(
                columns=col_specs,
                where=where,
                sep=self.sep,
            )
            .set_index(["datetime", "code"])
            .sort_index()
        )

    def update(self):
        """Advance to the next available timestamp and return the data snapshot.

        Moves the current time forward to the next timestamp in the available series,
        queries the DuckDB store for all instruments at that date, and stores the result.

        Returns:
          pandas.DataFrame or None: Data at the new current time indexed by code.
            Returns None if there are no future timestamps to advance to.
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
                    CAST({self.datetime_col} AS DATE) AS datetime,
                    {self.code_col} AS code,
                    {cols}
                FROM {table}
                WHERE CAST({self.datetime_col} AS DATE) >= '{self._time.date()}'
                    AND CAST({self.datetime_col} AS DATE) <= '{self._time.date()}'
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

        select_cols: List[str] = [f"{key_code} AS code"]
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
        df = df.set_index(["code"]).sort_index()
        self._data = df
        return df

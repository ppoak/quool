from quool import Source
from parquool import DuckPQ


class DuckPQSource(Source):
    """Market data source powered by DuckDB parquet manager with optional price adjustment.

    DuckPQSource queries data from a DuckPQ manager within a specified date
    range and exposes data at the current time. If an adjustment column is provided,
    OHLC prices are adjusted by multiplying with the adjustment factor.

    Attributes:
      manager (DuckPQ): DuckDB parquet manager used for queries.
      times (pandas.Index): Available timestamps within the [begin, end] range.
      time (pandas.Timestamp): Current timestamp.
      fields (list[str]): Requested base fields ['open', 'high', 'low', 'close', 'volume'] plus extras.
      date_col (str): Column name for date.
      code_col (str): Column name for instrument code.
      adj_col (str or None): Column name for adjustment factor; if provided, OHLC fields are adjusted.
    """

    def __init__(
        self,
        manager: DuckPQ,
        begin: str,
        end: str,
        extra: list = None,
        date_col: str = "date",
        code_col: str = "code",
        adj_col: str = "adjfactor",
    ):
        """Initialize the DuckPQ-backed source and set the initial time range.

        Queries available timestamps from the manager within [begin, end] and sets the
        current time to the earliest timestamp. Base OHLCV fields can be extended with
        extras, and an adjustment factor column can be specified to adjust OHLC prices.

        Args:
          manager (DuckPQ): Parquet manager that provides select queries.
          begin (str): Start date (inclusive), parsable by DuckDB (e.g., 'YYYY-MM-DD').
          end (str): End date (inclusive), parsable by DuckDB.
          extra (list, optional): Additional field names to include beyond OHLCV. Defaults to None.
          date_col (str, optional): Name of the date column in the underlying storage. Defaults to "date".
          code_col (str, optional): Name of the instrument code column. Defaults to "code".
          adj_col (str, optional): Name of the adjustment factor column used to adjust OHLC.
            If None, no adjustment is applied. Defaults to "adjfactor".

        Raises:
          ValueError: If no timestamps are found within the specified range.
        """
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
        """Return the data at the current time, optionally adjusted by the adjfactor.

        If adj_col is set, OHLC fields ['open', 'high', 'low', 'close'] are multiplied
        by the adjustment factor before returning. The adjustment column is dropped
        from the returned DataFrame.

        Returns:
          pandas.DataFrame: Rows for the current time, indexed by code_col, containing
          requested fields (OHLCV plus extras), with OHLC adjusted if adj_col is provided.

        Raises:
          KeyError: If expected columns (e.g., adjfactor or OHLC fields) are missing.
        """
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
        """Return the data at the current time with the requested fields.

        Note:
          This property returns the same-time snapshot (not cumulative history). If adj_col
          is provided, it returns the adjusted OHLCV similarly to data; otherwise returns
          raw fields for the current time.

        Returns:
          pandas.DataFrame: Data snapshot at the current time with requested fields.
        """
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
        """Advance to the next available timestamp and return the data snapshot.

        Moves the current time forward to the next timestamp in the available series.

        Returns:
          pandas.DataFrame or None: Data at the new current time. Returns None if there
          are no future timestamps to advance to.
        """
        future = self._timepoint[self._timepoint > self.time]
        if future.empty:
            return None
        self._time = future.min()
        return self.data

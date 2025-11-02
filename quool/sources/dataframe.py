import pandas as pd
from quool import Source


class DataFrameSource(Source):
    """Market data source backed by a pandas DataFrame with a time-based MultiIndex.

    DataFrameSource wraps a DataFrame where the first level of the index represents
    timestamps and subsequent levels represent instruments (e.g., codes). It exposes
    the latest snapshot at the current time and an accumulated view up to the current
    time, and advances time through the available index values.

    Attributes:
      times (pandas.Index): All timestamps up to and including the current time.
      datas (pandas.DataFrame): Historical data up to the current time.
      data (pandas.DataFrame): Data slice at the current time.
      open (str): Column name for open price.
      high (str): Column name for high price.
      low (str): Column name for low price.
      close (str): Column name for close price.
      volume (str): Column name for traded volume.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        open: str = "open",
        high: str = "high",
        low: str = "low",
        close: str = "close",
        volume: str = "volume",
    ):
        """Initialize the source with a time-indexed DataFrame.

        The first level of the index is used as the timeline. The initial current time
        is set to the minimum timestamp from the index.

        Args:
          data (pandas.DataFrame): DataFrame with a MultiIndex where level 0 is time
            and level 1 (or more) identifies instruments (e.g., codes). Must include
            OHLCV columns consistent with the provided names.
          open (str, optional): Column name for open price. Defaults to "open".
          high (str, optional): Column name for high price. Defaults to "high".
          low (str, optional): Column name for low price. Defaults to "low".
          close (str, optional): Column name for close price. Defaults to "close".
          volume (str, optional): Column name for volume. Defaults to "volume".

        Raises:
          ValueError: If the DataFrame has no time level in its index.
        """
        self._times = data.index.get_level_values(0).unique().sort_values()
        super().__init__(self._times.min(), data, open, high, low, close, volume)

    @property
    def times(self):
        """Return all timestamps up to and including the current time.

        Returns:
          pandas.Index: Sorted index of timestamps less than or equal to the current time.
        """
        return self._times[self._times <= self.time]

    @property
    def datas(self):
        """Return historical data up to the current time.

        Returns:
          pandas.DataFrame: Subset of the original DataFrame with all records up to
          and including the current time across instruments.
        """
        return self._data.loc[: self.time]

    @property
    def data(self):
        """Return the data slice at the current time.

        Returns:
          pandas.DataFrame: Rows corresponding to the current time across instruments.
        """
        return self._data.loc[self.time]

    def update(self) -> pd.DataFrame:
        """Advance to the next available timestamp and return its data slice.

        Moves the current time to the next timestamp in the index, if any.

        Returns:
          pandas.DataFrame or None: Data slice at the new current time. Returns None
          if there are no future timestamps to advance to.
        """
        future = self._times[self._times > self.time]
        if future.empty:
            return None  #
        self._time = future.min()
        data = self._data.loc[self._time]
        return data

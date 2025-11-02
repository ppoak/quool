import pandas as pd


class Source:
    """Abstract market data source carrying time and an optional OHLCV snapshot.

    Source provides a standardized interface for accessing market data (e.g., OHLCV)
    at a given timestamp and for updating to new data. It stores the current
    timestamp, an optional pandas DataFrame snapshot, and the column names used
    to reference open, high, low, close, and volume.

    This class is intended to be subclassed. Subclasses should implement update()
    to retrieve or compute the next snapshot and advance time.

    Attributes:
      time (pandas.Timestamp): Current market timestamp.
      data (pandas.DataFrame or None): Current market snapshot, indexed by instrument code.
      open (str): Column name for the open price.
      high (str): Column name for the high price.
      low (str): Column name for the low price.
      close (str): Column name for the close price.
      volume (str): Column name for the traded volume.
    """

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
        """Initialize a Source with a timestamp, optional data, and column name mapping.

        Args:
          time (pandas.Timestamp): Current market timestamp.
          data (pandas.DataFrame, optional): Market snapshot with instrument codes
            as index and OHLCV columns. Defaults to None.
          open (str, optional): Column name for open price. Defaults to "open".
          high (str, optional): Column name for high price. Defaults to "high".
          low (str, optional): Column name for low price. Defaults to "low".
          close (str, optional): Column name for close price. Defaults to "close".
          volume (str, optional): Column name for volume. Defaults to "volume".

        Raises:
          ValueError: Not raised here by default; subclasses may validate inputs.
        """
        self._time = time
        self._data = data
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

    @property
    def time(self):
        """Return the current market timestamp.

        Returns:
          pandas.Timestamp: The current time associated with the source.
        """
        return self._time

    @property
    def data(self):
        """Return the current market snapshot.

        Returns:
          pandas.DataFrame or None: The current data frame of market prices and
          volume. May be None if no snapshot is available.
        """
        return self._data

    def update(self) -> pd.DataFrame:
        """Advance the source to the next snapshot and update time.

        Subclasses must implement this method to fetch or compute new market data
        and update the timestamp accordingly.

        Returns:
          pandas.DataFrame: The updated market snapshot.

        Raises:
          NotImplementedError: Always, in the base class. Must be overridden in subclasses.
        """
        raise NotImplementedError

    def __str__(self):
        return f"{self.__class__.__name__}@[{self.time}]"

    def __repr__(self):
        return super().__repr__()

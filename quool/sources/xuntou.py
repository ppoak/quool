import pandas as pd
from .dataframe import DataFrameSource


class XtDataPreloadSource(DataFrameSource):
    """Preloaded market data source using xtquant xtdata into a DataFrame.

    XtDataPreloadSource pulls historical data for a sector and period from xtquant,
    converts it into a time-code MultiIndex DataFrame, and exposes the interface
    of DataFrameSource. This is suitable for backtesting or offline analysis.

    Attributes:
      begin (str): Start timestamp used for xtdata queries (formatted as '%Y%m%d%H%M%S').
      end (str): End timestamp used for xtdata queries (formatted as '%Y%m%d%H%M%S').
      _stock_list (list[str]): List of stock codes in the specified sector.
      times (pandas.Index): Available timestamps up to the current time (inherited).
      datas (pandas.DataFrame): Historical data up to the current time (inherited).
      data (pandas.DataFrame): Data slice at the current time (inherited).
    """

    def __init__(
        self,
        path: str,
        begin: str,
        end: str,
        period: str = "1d",
        sector: str = "沪深A股",
    ):
        """Initialize the xtquant-backed source and preload historical data.

        Loads market data for the given stock sector and period from xtdata within
        the [begin, end] window, constructs a MultiIndex DataFrame with index levels:
        'datetime' (pandas.Timestamp) and 'code' (stock code). The data is sorted
        by the MultiIndex and passed to the DataFrameSource initializer.

        Args:
          path (str): xtquant data directory path.
          begin (str): Start time (parsable by pandas) for the data window.
          end (str): End time (parsable by pandas) for the data window.
          period (str, optional): Sampling period (e.g., '1d', '1m'). Defaults to '1d'.
          sector (str, optional): Stock sector name recognized by xtdata. Defaults to '沪深A股'.

        Raises:
          ImportError: If xtquant is not installed or cannot be imported.
          ValueError: If no data is returned for the given parameters.
        """
        from xtquant import xtdata

        xtdata.data_dir = path
        self.begin = pd.to_datetime(begin).strftime(r"%Y%m%d%H%M%S")
        self.end = pd.to_datetime(end).strftime(r"%Y%m%d%H%M%S")
        self._stock_list = xtdata.get_stock_list_in_sector(sector)
        data = xtdata.get_market_data_ex(
            stock_list=self._stock_list,
            period=period,
            start_time=self.begin,
            end_time=self.end,
        )

        data = pd.concat(data.values(), keys=data.keys())
        data.index = pd.MultiIndex.from_arrays(
            [
                pd.to_datetime(data.index.get_level_values(1)),
                data.index.get_level_values(0),
            ],
            names=["datetime", "code"],
        )
        data = data.sort_index()
        super().__init__(data=data)

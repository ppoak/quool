import pandas as pd
from .dataframe import DataFrameSource


class XtDataPreloadSource(DataFrameSource):

    def __init__(
        self,
        path: str,
        begin: str,
        end: str,
        period: str = "1d",
        sector: str = "沪深A股",
    ):
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

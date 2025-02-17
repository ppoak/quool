import pandas as pd
from quool.source import SourceBase


class DataFrameSource(SourceBase):

    def __init__(self, data: pd.DataFrame):
        self._data = data
        self._times = data.index.get_level_values(0).unique().sort_values()
        self._time = self._times.min()
    
    @property
    def times(self):
        return self._times[self._times <= self.time]
    
    @property
    def datas(self):
        return self._data.loc[:self.time]
    
    @property
    def data(self):
        return self._data.loc[self.time]
    
    def update(self):
        future = self._timepoint[self._timepoint > self.time]
        if future.empty:
            return None
        self._time = future.min()
        return self._data.loc[self._time]

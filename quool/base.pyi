import quool
import datetime
import numpy as np
import pandas as pd
import backtrader as bt
from typing import overload


class FrameWorkError(Exception):
    def __init__(self, func: str, hint: str) -> None:...
    def __str__(self) -> str: ...


class Worker(object):
    TSFR = 1
    CSFR = 2
    PNFR = 3
    TSSR = 4
    CSSR = 5
    PNSR = 6
    MISR = 8
    MIFR = 9
    MCFR = 10
    MIMC = 11

    @staticmethod
    def series2frame(data: quool.Series, name: str = None) -> quool.DataFrame: ...
    @staticmethod
    def frame2series(data: quool.DataFrame, name: str = None) -> quool.Series: ...
    @staticmethod
    def ists(data: 'quool.DataFrame | quool.Series') -> bool: ...
    @staticmethod
    def iscs(data: 'quool.DataFrame | quool.Series') -> bool: ...
    @staticmethod
    def ispanel(data: 'quool.DataFrame | quool.Series') -> bool: ...
    @staticmethod
    def isframe(data: 'quool.DataFrame | quool.Series') -> bool: ...
    @staticmethod
    def isseries(data: 'quool.DataFrame | quool.Series') -> bool: ...
    @staticmethod
    def ismi(index: pd.Index) -> bool: ...
    def _validate(self) -> None: ...
    def _flat(self, datetime, asset, indicator) -> quool.DataFrame: ...
    def _to_array(self, *axes) -> np.ndarray: ...

class Strategy(bt.Strategy):
    def log(self, text: str, datetime: datetime.datetime = None, hint: str = 'INFO') -> None: ...
    def notify_order(self, order: bt.Order) -> None: ...
    def notify_trade(self, trade) -> None: ...

class Indicator(bt.Indicator):
    def log(self, text: str, datetime: datetime.datetime = None, hint: str = 'INFO') -> None: ...

class Analyzer(bt.Analyzer):
    def log(self, text: str, datetime: datetime.datetime = None, hint: str = 'INFO') -> None: ...

class Observer(bt.Observer):
    def log(self, text: str, datetime: datetime.datetime = None, hint: str = 'INFO') -> None: ...

class OrderTable(Analyzer):
    def notify_order(self, order) -> None: ...
    def get_analysis(self) -> quool.DataFrame: ...

def from_array(
    arr: np.ndarray,
    index: pd.Index = None, 
    columns: pd.Index = None, 
    index_axis: 'int | list | tuple' = None,
    columns_axis: 'int | list | tuple' = None,
) -> 'quool.DataFrame | quool.Series': 
    """Create a DataFrame from multi-dimensional array
    ---------------------------------------------------

    arr: np.ndarray, a multi-dimensional array
    index: pd.Index, the index used as in row
    columns: pd.Index, the index used as in column
    index_axis: int, list or tuple, the sequence of axes used to transpose from original to result
    columns_axis, int, list or tuple, the sequence of axes used to transpose from original to result 
    """

def concat(objs, colnames: list = None, **kwargs) -> quool.DataFrame: ...
def read_csv(
    path_or_buffer,
    perspective: str = None,
    name_pattern: str = None,      
    **kwargs
) -> quool.DataFrame:
    '''A enhanced function for reading files in a directory to a panel DataFrame
    ----------------------------------------------------------------------------

    path: path to the directory
    perspective: 'datetime', 'asset', 'indicator'
    name_pattern: pattern to match the file name, which will be extracted as index
    kwargs: other arguments for pd.read_csv

    **note: the name of the file in the directory will be interpreted as the 
    sign(column or index) to the data, so set it to the brief one
    '''
def read_excel(
    path_or_buffer,
    perspective: str = None,
    name_pattern: str = None,
    **kwargs,
) -> quool.DataFrame:
    '''A enhanced function for reading files in a directory to a panel DataFrame
    ----------------------------------------------------------------------------

    path: path to the directory
    perspective: 'datetime', 'asset', 'indicator'
    kwargs: other arguments for pd.read_excel

    **note: the name of the file in the directory will be interpreted as the 
    sign(column or index) to the data, so set it to the brief one
    '''
def read_parquet(path, **kwargs) -> quool.DataFrame: ...
def read_sql(query, con, **kwargs) -> quool.DataFrame: ...
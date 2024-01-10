import re
import logging
import numpy as np
import pandas as pd
from typing import Any
from collections.abc import Mapping


class __TimeFormatter(logging.Formatter):

    def __init__(
        self, 
        display_time: bool = True,
        display_name: str = True,
        fmt: str | None = None, 
        datefmt: str | None = None, 
        style: Any = "%", 
        validate: bool = True, *, 
        defaults: Mapping[str, Any] | None = None
    ) -> None:
        super().__init__(fmt, datefmt, style, validate, defaults=defaults)
        self.display_time = display_time
        self.display_name = display_name


class _StreamFormatter(__TimeFormatter):
    COLORS = {
        'DEBUG': '\033[94m',
        'INFO': '\033[92m',
        'WARNING': '\033[93m',
        'ERROR': '\033[95m',
        'CRITICAL': '\033[31m',
        'RESET': '\033[0m',
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.message = record.getMessage()
        record.asctime = self.formatTime(record, self.datefmt)
        formatted_record = f'{color}'
        if self.display_time:
            formatted_record += f'[{record.asctime}] '
        if self.display_name:
            formatted_record += f'<{record.name}> '
        formatted_record += f'{record.message}{self.COLORS["RESET"]}'
        return formatted_record


class _FileFormatter(__TimeFormatter):

    def format(self, record):
        record.message = record.getMessage()
        record.asctime = self.formatTime(record, self.datefmt)
        formatted_record = ''
        if self.display_time:
            formatted_record += f'[{record.asctime}] '
        if self.display_name:
            formatted_record += f'<{record.name}> '
        formatted_record += f'|{record.levelname}| {record.message}'
        return formatted_record


class Logger(logging.Logger):
    """
    A custom Logger class that extends Python's standard logging.Logger.

    This Logger allows output to both console and file with configurable display options.

    Attributes:
        name (str): Name of the logger.
        level (int): Logging level (e.g., logging.DEBUG, logging.INFO).
        stream (bool): Flag to enable or disable console logging.
        file (str): Path to a log file for file logging.
        display_time (bool): Flag to display timestamps in log messages.
        display_name (bool): Flag to display the logger's name in log messages.

    Methods:
        __init__(self, name, level, stream, file, display_time, display_name): Initializes the Logger object.

    Example:
        logger = Logger(name="MyLogger", level=logging.INFO, file="log.txt")
        logger.info("This is an info message")
    """

    def __init__(
        self, 
        name: str = None, 
        level: int = logging.DEBUG, 
        stream: bool = True, 
        file: str = None,
        display_time: bool = True,
        display_name: bool = False,
    ):
        """
        Initializes the Logger object.

        Args:
            name (str, optional): Name of the logger. Defaults to 'QuoolLogger'.
            level (int, optional): Logging level. Defaults to logging.DEBUG.
            stream (bool, optional): Whether to log to console. Defaults to True.
            file (str, optional): File path to log to. Defaults to None (no file logging).
            display_time (bool, optional): Whether to display timestamps. Defaults to True.
            display_name (bool, optional): Whether to display the logger's name. Defaults to False.
        """
        name = name or 'QuoolLogger'
        super().__init__(name, level)

        if stream:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(_StreamFormatter(
                display_time=display_time, display_name=display_name
            ))
            self.addHandler(stream_handler)

        if file:
            file_handler = logging.FileHandler(file)
            file_handler.setFormatter(_FileFormatter(
                display_time=display_time, display_name=display_name
            ))
            self.addHandler(file_handler)


def parse_date(
    date: str | list = None,
    default: str = '20000104',
    format_: str = None,
    errors: str = 'ignore'
) -> tuple:
    """
    Parses a date string or a list of date strings into pandas.Timestamp format.

    Args:
        date (str | list, optional): The date string or list of date strings to parse. Defaults to None.
        default (str, optional): Default date string to use if 'date' is None. Defaults to '20000104'.
        format_ (str, optional): The format string to use for parsing. If None, pandas will infer the format. Defaults to None.
        errors (str, optional): Specifies how to handle errors. 
                                'ignore' returns the original input if parsing fails,
                                'raise' will raise an error, and 
                                'coerce' will set invalid parsing as NaT (Not a Time). Defaults to 'ignore'.

    Returns:
        pandas.Timestamp | list[pandas.Timestamp]: The parsed date(s). If 'date' is a single string, returns a single Timestamp. 
                                                  If 'date' is a list, returns a list of Timestamps. 
                                                  Returns the input as is if parsing fails and errors is set to 'ignore'.

    Example:
        single_date = parse_date("2021-01-01")
        multiple_dates = parse_date(["2021-01-01", "2021-02-01"])
    """
    if not isinstance(date, list):
        try:
            date = (
                pd.to_datetime(date, errors=errors, format=format_) if date is not None 
                else pd.to_datetime(default, errors=errors, format=format_)
            )
            return date
        except Exception as _:
            return date
            
    else:
        return pd.to_datetime(date, errors=errors, format=format_)

def parse_commastr(
    commastr: 'str | list',
) -> pd.Index:
    """
    Parses a comma-separated string into a list of strings, or returns the list if the input is already a list.

    Args:
        commastr (str | list): A comma-separated string or a list of strings.

    Returns:
        list: A list of strings derived from the comma-separated input string. 
              If the input is already a list, it is returned as is. 
              If the input is None, None is returned.

    Example:
        result = parse_commastr("apple, banana, cherry")
        # result will be ['apple', 'banana', 'cherry']
    """
    if isinstance(commastr, str):
        commastr = commastr.split(',')
        return list(map(lambda x: x.strip(), commastr))
    elif commastr is None:
        return None
    else:
        return commastr

def panelize(data: pd.DataFrame | pd.Series) -> pd.Series | pd.DataFrame:
    """
    Ensures that a DataFrame or Series with a MultiIndex covers all combinations of index levels.

    If the data's MultiIndex does not cover all combinations of its levels, the data is reindexed 
    to include all possible combinations, filling missing combinations with NaN.

    Args:
        data (pd.DataFrame | pd.Series): The DataFrame or Series to be reindexed.

    Returns:
        pd.DataFrame | pd.Series: The reindexed DataFrame or Series with a complete MultiIndex.

    Example:
        df = pd.DataFrame({
            'value': [1, 2, 3]
        }, index=pd.MultiIndex.from_tuples([
            ('A', 'x'), ('B', 'y'), ('C', 'z')
        ], names=['first', 'second']))
        panelized_df = panelize(df)
        # panelized_df will have a complete MultiIndex with all combinations of 'first' and 'second' levels.
    """
    if isinstance(data, (pd.DataFrame, pd.Series))\
        and isinstance(data.index, pd.MultiIndex):
        levels = [data.index.get_level_values(i).unique() 
            for i in range(data.index.nlevels)]
        if data.shape[0] < np.prod([level.size for level in levels]):
            data = data.reindex(pd.MultiIndex.from_product(levels))
        return data
    
    else:
        raise ValueError("only series and dataframe with multiindex is available")

def reduce_mem_usage(df: pd.DataFrame):
    """
    Reduces the memory usage of a pandas DataFrame by downcasting data types.

    Iterates through all columns of the DataFrame and modifies the data type
    to the most memory-efficient type that still supports the range of values
    in each column.

    Args:
        df (pd.DataFrame): The DataFrame whose memory usage is to be reduced.

    Returns:
        pd.DataFrame: The DataFrame with optimized memory usage.

    Example:
        optimized_df = reduce_mem_usage(original_df)
        # optimized_df will have the same values as original_df but with reduced memory usage.
    """
    logger = Logger("QuoolReduceMemUsage")
    start_mem = df.memory_usage().sum()
    logger.info('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum()
    logger.info('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    logger.info('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

def format_code(code, format_str = '{market}.{code}', upper: bool = True):
    """
    Formats a stock code according to a specified pattern.

    Args:
        code (str): The stock code to format.
        format_str (str, optional): The format string to use. Defaults to '{market}.{code}'.
        upper (bool, optional): Flag to convert the market code to uppercase. Defaults to True.

    Returns:
        str: The formatted stock code.

    Raises:
        ValueError: If the input code format is not understood.

    Example:
        formatted_code = format_code('600000', upper=True)
        # formatted_code will be 'SH.600000'
    """
    if len(c := code.split('.')) == 2:
        dig_code = c.pop(0 if c[0].isdigit() else 1)
        market_code = c[0]
        if upper:
            market_code = market_code.upper()
        return format_str.format(market=market_code, code=dig_code)
    elif len(code.split('.')) == 1:
        sh_code_pat = '6\d{5}|9\d{5}'
        sz_code_pat = '0\d{5}|2\d{5}|3\d{5}'
        bj_code_pat = '8\d{5}|4\d{5}'
        if re.match(sh_code_pat, code):
            return format_str.format(code=code, market='sh' if not upper else 'SH')
        if re.match(sz_code_pat, code):
            return format_str.format(code=code, market='sz' if not upper else 'SZ')
        if re.match(bj_code_pat, code):
            return format_str.format(code=code, market='bj' if not upper else 'BJ')
    else:
        raise ValueError("Your input code is not unstood")

def strip_stock_code(code: str):
    """
    Strips the market prefix from a stock code.

    Args:
        code (str): The stock code with potential market prefix.

    Returns:
        str: The stock code without the market prefix.

    Example:
        stripped_code = strip_stock_code('SZ.000001')
        # stripped_code will be '000001'
    """
    code_pattern = r'\.?[Ss][Zz]\.?|\.?[Ss][Hh]\.?|\.?[Bb][Jj]\.?'\
        '|\.?[Oo][Ff]\.?'
    return re.sub(code_pattern, '', code)

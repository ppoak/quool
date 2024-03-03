import logging
import numpy as np
import pandas as pd


class __TimeFormatter(logging.Formatter):

    def __init__(
        self, 
        display_time: bool = True,
        display_name: str = True,
        fmt: str | None = None, 
        datefmt: str | None = None, 
        style: str = "%", 
        validate: bool = True, *, 
        defaults = None
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

    def __init__(
        self, 
        name: str = None, 
        level: int = logging.DEBUG, 
        stream: bool = True, 
        file: str = None,
        display_time: bool = True,
        display_name: bool = False,
    ):
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


def parse_commastr(
    commastr: 'str | list',
) -> pd.Index:
    if isinstance(commastr, str):
        commastr = commastr.split(',')
        return list(map(lambda x: x.strip(), commastr))
    elif commastr is None:
        return None
    else:
        return commastr

def reduce_mem_usage(df: pd.DataFrame):
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

"""Some common functions to be used in quool
============================================

Like convert some type of data into another type of data,
or some common tools to get basic trading date infomation

Examples:
-----------

>>> import bearalpha as ba
>>> ba.last_report_period('2015-01-01')

You can get access to this module to prettify your output backend
by rich, or you can just register some fonts just by matplotlib

Examples:
----------

>>> import bearalpha as ba
>>> ba.reg_font('/some/path/to/your/font', 'the_name_you_give')
"""

import re
import rich
import time
import numpy
import pandas
import datetime
import matplotlib
import pandas as pd
from functools import wraps
from six import with_metaclass
from rich.console import Console as RichConsole
from rich.progress import track
from rich.progress import Progress as RichProgress
from rich.traceback import install
from rich.table import Table
from rich.progress import (
    BarColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
    SpinnerColumn,
)

def time2str(date: 'str | datetime.datetime | int | datetime.date', formatstr: str = r'%Y-%m-%d') -> str:
    """convert a datetime class to time-like string"""
    if isinstance(date, int):
        date = str(date)
    date = pd.to_datetime(date)
    if isinstance(date, datetime.datetime):
        date = date.strftime(formatstr)
    return date

def str2time(date: 'str | datetime.datetime') -> datetime.datetime:
    """convert a time-like string to datetime class"""
    if isinstance(date, (str, datetime.date)):
        date = pd.to_datetime(date)
    elif isinstance(date, (float, int)):
        date = pd.to_datetime(str(int(date)))
    return date

def item2list(item) -> list:
    """convert a non list item to a list"""
    if item is None:
        return []
    elif not isinstance(item, list):
        return [item]
    else:
        return item

def item2tuple(item) -> list:
    """convert a non tuple item to a tuple"""
    if not isinstance(item, tuple):
        return (item, )
    else:
        return item
        
def hump2snake(hump: str) -> str:
    """convert hump name to snake name"""
    snake = re.sub(r'([a-z]|\d)([A-Z])', r'\1_\2', hump).lower()
    return snake

def strip_stock_code(code: str):
    code_pattern = r'\.?[Ss][Zz]\.?|\.?[Ss][Hh]\.?|\.?[Bb][Jj]\.?'\
        '|\.?[Oo][Ff]\.?'
    return re.sub(code_pattern, '', code)
    
def wrap_stock_code(code: str, formatstr: str = '{code}.{market}', to_lower: bool = False):
    if len(code.split('.')) != 1:
        raise ValueError('It seems your code is already wrapped')
    sh_code_pat = r'6\d{5}|9\d{5}'
    sz_code_pat = r'0\d{5}|2\d{5}|3\d{5}'
    bj_code_pat = r'4\d{5}|8\d{5}'
    if re.match(sh_code_pat, code):
        return formatstr.format(code=code, market='sh' if to_lower else 'SH')
    elif re.match(sz_code_pat, code):
        return formatstr.format(code=code, market='sz' if to_lower else 'SZ')
    elif re.match(bj_code_pat, code):
        return formatstr.format(code=code, market='bj' if to_lower else 'BJ')
    else:
        raise ValueError('No pattern can match your code, please check it')

def latest_report_period(date: 'str | datetime.datetime | datetime.date',
    n: int = 1) -> 'str | list[str]':
    """Get the nearest n report period
    ----------------------------------

    date: str, datetime or date, the given date
    n: int, the number of report periods before the given date,
    """
    date = str2time(date)
    this_year = date.year
    last_year = this_year - 1
    nearest_report_date = {
        "01-01": str(last_year) + "-09-30",
        "04-30": str(this_year) + "-03-31",
        "08-31": str(this_year) + "-06-30",
        "10-31": str(this_year) + "-09-30"
    }
    report_date = list(filter(lambda x: x <= date.strftime(r'%Y-%m-%d')[-5:], 
        nearest_report_date.keys()))[-1]
    report_date = nearest_report_date[report_date]
    fundmental_dates = pd.date_range(end=report_date, periods=n, freq='q')
    fundmental_dates = list(map(lambda x: x.strftime(r'%Y-%m-%d'), fundmental_dates))
    return fundmental_dates

def timeit(func):
    wraps(func)
    def decorated(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        spent = end - start

        if spent <= 60:
            timestr = f'{spent: .4f}s'
        
        elif spent <= 3600:
            minute = spent // 60
            second = spent - minute * 60
            timestr = f'{minute}m {second: .4f}s'
            
        else:
            hour = spent // 3600
            minute = (spent - hour * 3600) // 60
            second = spent - hour * 3600 - minute * 60
            timestr = f'{hour}h {minute}m {second:.4f}s'
        
        print(f'{func.__name__} Time Spent: {timestr}')
        return result
    return decorated

def varexist(varname):
    try:
        eval(varname)
        return True
    except:
        return False


class Singleton(type):
    _instance = None

    def __call__(cls, *args, **kwargs):
        if cls is not cls._instance:
            cls._instance = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instance


class Console(with_metaclass(Singleton, RichConsole)):
    ...


def progressor(
    *columns: str,
    console = None,
    auto_refresh: bool = True,
    refresh_per_second: float = 10,
    speed_estimate_period: float = 30.0,
    transient: bool = False,
    redirect_stdout: bool = True,
    redirect_stderr: bool = True,
    get_time = None,
    disable: bool = False,
    expand: bool = False,
):
    return RichProgress(
        SpinnerColumn(spinner_name='monkey'), 
        BarColumn(), 
        MofNCompleteColumn(), 
        TimeRemainingColumn(),
        *columns,
        console = console,
        auto_refresh = auto_refresh,
        refresh_per_second = refresh_per_second,
        speed_estimate_period = speed_estimate_period,
        transient = transient,
        redirect_stdout = redirect_stdout,
        redirect_stderr = redirect_stderr,
        get_time = get_time,
        disable = disable,
        expand = expand,
    )
    
def beautify_traceback(
    *,
    console = None,
    width: int = 100,
    extra_lines: int = 3,
    theme: str = None,
    word_wrap: bool = False,
    show_locals: bool = False,
    indent_guides: bool = True,
    suppress: 'str | list' = None,
    max_frames: int = 100
):
    """Enable traceback beautifier backend by rich"""
    import backtrader
    install(
        console = console,
        suppress = [rich, pandas, numpy, matplotlib, backtrader] or suppress, 
        width = width,
        extra_lines = extra_lines,
        theme = theme,
        word_wrap = word_wrap,
        show_locals = show_locals,
        indent_guides = indent_guides,
        max_frames = max_frames,
    )

def reg_font(fontpath: str, fontname: str):
    """Register a font in matplotlib and use it

    fontpath: str, the path of the font
    fontname: str, the name of the font    
    """
    
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    font_manager.fontManager.addfont(fontpath)
    plt.rcParams['font.sans-serif'] = fontname


if __name__ == '__main__':
    pass
import re
import datetime
import pandas as pd


def time2str(date: 'str | datetime.datetime | int | datetime.date') -> str:
    if isinstance(date, int):
        date = str(date)
    date = pd.to_datetime(date)
    date = date.strftime(r'%Y-%m-%d')
    return date

def str2time(date: 'str | datetime.datetime') -> datetime.datetime:
    if isinstance(date, (str, datetime.date)):
        date = pd.to_datetime(date)
    elif isinstance(date, (float, int)):
        date = pd.to_datetime(str(int(date)))
    return date

def item2list(item) -> list:
    if item is None:
        return []
    elif not isinstance(item, (list, tuple, set, dict)):
        return [item]
    else:
        return item

def item2tuple(item) -> list:
    if item is None:
        return ()
    elif not isinstance(item, (list, tuple, set, dict)):
        return (item, )
    else:
        return item
        
def hump2snake(hump: str) -> str:
    snake = re.sub(r'([a-z]|\d)([A-Z])', r'\1_\2', hump).lower()
    return snake

def nearest_report_period(date: 'str | datetime.datetime | datetime.date',
    n: int = 1) -> 'str | list[str]':
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

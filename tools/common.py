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
    if not isinstance(item, (list, tuple, set, dict)):
        return [item]
    else:
        return item

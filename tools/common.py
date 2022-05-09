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
    if item == None:
        return item
    elif not isinstance(item, (list, tuple, set, dict)):
        return [item]
    else:
        return item

def hump2snake(hump: str) -> str:
    snake = re.sub(r'([a-z]|\d)([A-Z])', r'\1_\2', hump).lower()
    return snake

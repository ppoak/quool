import datetime
import pandas as pd


def time2str(date: 'str | datetime.date | datetime.datetime') -> str:
    if isinstance(date, (datetime.datetime, datetime.date)):
        date = date.strftime(r'%Y-%m-%d')
    return date

def str2time(date: 'str | datetime.date | datetime.datetime') -> datetime.datetime:
    if isinstance(date, (str, datetime.date)):
        date = pd.to_datetime(date)
    return date
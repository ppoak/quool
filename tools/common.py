import re
import redis
import pickle
import hashlib
import datetime
import pandas as pd
from functools import wraps


try:
    REDISCON = redis.Redis(host='localhost', port=6379)
    REDISCON.get(name='test')
except:
    REDISCON = None
DEBUG = False
REDIS_TIME = 3600 * 2

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

def get_cache(db, key):
    try:
        r_key = db + ":" + key
        v = REDISCON.get(name=r_key)
        if v:
            return pickle.loads(v)
        else:
            return None
    except Exception as e:
        print(str(e))
        return None

def get_raw_cache(db, key):
    try:
        r_key = db + ":" + key
        v = REDISCON.get(name=r_key)
        if v:
            return v
        else:
            return None
    except Exception as e:
        print(str(e))
        return None

def to_cache(db, key, data, expire=REDIS_TIME):
    try:
        r_key = db + ":" + key
        p_data = pickle.dumps(data)
        REDISCON.set(name=r_key, value=p_data)
        REDISCON.expire(name=r_key, time=expire)
        return True
    except Exception as e:
        print(str(e))
        return False

def str_to_cache(db, key, dataStr, expire=REDIS_TIME):
    try:
        r_key = db + ":" + key
        REDISCON.set(name=r_key, value=dataStr)
        REDISCON.expire(name=r_key, time=expire)
        return True
    except Exception as e:
        print(str(e))
        return False

def redis_cache(db="", ex=REDIS_TIME):
    def _cache(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            hash_key = hashlib.md5(pickle.dumps((func.__name__, args, kwargs))).hexdigest()
            cache = get_raw_cache(db=db, key=hash_key)
            if cache:
                # get cache successful
                if DEBUG:
                    print('get cache')
                return pickle.loads(cache)
            else:
                # not fund cache,return data will be cache
                if DEBUG:
                    print('cache data')
                d = func(*args, **kwargs)
                to_cache(db=db, key=hash_key, data=d, expire=ex)
                return d

        return wrapper

    return _cache

def delete_cache(db='', key=""):
    try:
        db_key = db + ":" + key
        REDISCON.delete(db_key)
        return True
    except Exception as e:
        print(str(e))
        return False

@redis_cache(db='test', ex=60)
def cache_test(a, b):
    return a + b

if __name__ == '__main__':
    import time
    s = time.time()
    print(cache_test(6, 8))
    e = time.time()
    print(e - s)
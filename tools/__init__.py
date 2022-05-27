
from .core import (
    FrameWorkError,
    Worker,
    Request,
    ProxyRequest
    )

from .common import (
    time2str,
    str2time,
    item2list,
    hump2snake,
    nearest_report_period,
)

from .common import REDISCON
if REDISCON is not None:
    from .common import redis_cache as cache
else:
    from functools import lru_cache as cache
    print('[!] Your redis server is not started or installed,'
          'some crawler will not response as quick as they can')

__all__ = [
    'FrameWorkError',
    'Worker',
    'Request',
    'ProxyRequest',
    'time2str',
    'str2time',
    'item2list',
    'hump2snake',
    'nearest_report_period',
    'cache',
    ]
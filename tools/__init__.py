from .core import (
    FrameWorkError,
    Worker,
    Request,
    ProxyRequest,
    REDIS,
    RedisCache,
    get_proxy,
    CBD,
    )

from .common import (
    time2str,
    str2time,
    item2list,
    hump2snake,
    latest_report_period,
    MICROSECOND,
    SECOND,
    MINUTE,
    HOUR,
    DAY,
    WEEK,
    MONTH,
    YEAR,
)


try:
    REDIS.ping()
except:
    print('[!] Your redis server is not started or installed, '
          'some crawler will not response as quick as they can')

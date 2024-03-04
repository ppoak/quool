from .core import Table

from .table import (
    ItemTable,
    DatetimeTable,
    PanelTable,
)

from .spider import (
    ProxyManager,
    get_spot_data,
    wechat_login,
    ewechat_notify,
)

from .trade import (
    TradeRecorder
)

from .factor import (
    Factor
)

from .util import (
    Logger,
    parse_commastr,
    reduce_mem_usage,
)


DEBUG = 10
INFO = 20
WARNING = 30
CRITICAL = 40


__version__ = "5.0.0"

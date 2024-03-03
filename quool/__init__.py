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


DEBUG = 10
INFO = 20
WARNING = 30
CRITICAL = 40


__version__ = "4.4.8"

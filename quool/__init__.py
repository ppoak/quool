from .database import (
    Table,
    AssetTable,
    FrameTable,
    DiffTable,
)

try:
    import akshare
    from .collector import AkShare
except:
    print("akshare is not installed, install it by `pip install quool[crawler]`")

from .collector import (
    Request,
    Em,
    StockUS,
    Cnki,
    WeiboSearch,
    HotTopic,
    KaiXin,
    KuaiDaili,
    Ip3366,
    Ip98,
)

from .tools import (
    Logger,
    parse_commastr,
    parse_date,
    reduce_mem_usage,
    format_code,
    strip_stock_code,
)

try:
    import backtrader as bt
    from .backtest import (
        BackTrader,
        Relocator,
        Strategy,
        Indicator,
        Analyzer,
        Observer,
        OrderTable,
        CashValueRecorder,
    )
except:
    print("backtrader is not installed, install it by `pip install quool[backtest]`")

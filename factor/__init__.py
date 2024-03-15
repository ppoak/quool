from .base import (
    BaseFactor, fqtm, fqtd, fcon,
    zscore, minmax,
    madoutlier, stdoutlier, iqroutlier,
    fillna, log, tsmean,
)

from .marketsize import MarketSizeFactor
from .retdist import RetDistFactor
from .deraprice import DeraPriceFactor
from .voldist import VolDistFactor
from .volatile import VolatileFactor
from .capflow import CapFlowFactor

class Factor(MarketSizeFactor, RetDistFactor, DeraPriceFactor, VolDistFactor, VolatileFactor, CapFlowFactor):

    pass
    
factor = Factor("./data/factor", code_level="order_book_id", date_level="date")

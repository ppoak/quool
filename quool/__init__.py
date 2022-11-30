from pandas import (
    DataFrame as PDDataFrame,
    Series as PDSeries,
)

from .base import (
    Strategy,
    Indicator,
    Analyzer,
    Observer,
    OrderTable,
    from_array,
    concat,
    read_excel, read_csv,
)

from .artist import (
    Drawer,
    Printer,
)

from .analyst import (
    Regressor,
    Describer,
    Decompositer,
    SigTester,
)

from .fetcher import (
    Filer,
    Sqliter,
    Mysqler,
)

from .calculator import (
    Calculator
)
    
from .processor import (
    PreProcessor,
    Converter,
)

from .backtester import (
    Relocator,
    BackTrader,
    Factester,
)

from .evaluator import (
    Evaluator,
)


class DataFrame(PDDataFrame):
    drawer: Drawer
    printer: Printer
    regressor: Regressor
    describer: Describer
    decompositer: Decompositer
    tester: SigTester
    filer: Filer
    sqliter: Sqliter
    mysqler: Mysqler
    calculator: Calculator
    converter: Converter
    preprocessor: PreProcessor
    backtrader: BackTrader
    relocator: Relocator
    factester: Factester
    evaluator: Evaluator


class Series(PDSeries):
    drawer: Drawer
    printer: Printer
    regressor: Regressor
    describer: Describer
    decompositer: Decompositer
    tester: SigTester
    filer: Filer
    sqliter: Sqliter
    mysqler: Mysqler
    calculator: Calculator
    converter: Converter
    preprocessor: PreProcessor
    backtrader: BackTrader
    relocator: Relocator
    factester: Factester
    evaluator: Evaluator


__version__ = '0.0.1'

del PDDataFrame, PDSeries

import datetime
import pandas as pd
from pathlib import Path
from .core.table import Table
from .core.util import parse_commastr


class FrameTable(Table):

    @property
    def spliter(self):
        return super().spliter
    
    @property
    def namer(self):
        return super().namer
    
    def read(
        self,
        column: str | list = None,
        start: str | list = None,
        stop: str = None
    ):
        filters = None
        if isinstance(start, list):
            filters = [(self.get_levelname(0), "in", start)]
        elif isinstance(start, str):
            filters = [(self.get_levelname(0), ">=", start)]
            if isinstance(stop, str):
                filters.append((self.get_levelname(0), "<=", stop))
        return super().read(parse_commastr(column), filters)


class PanelTable(Table):
    
    def __init__(
        self,
        uri: str | Path,
        code_level: str | int = 0,
        date_level: str | int = 1,
        freq: str = "M",
        format: str = r"%Y%m",
        create: bool = False,
    ):
        self._code_level = code_level
        self._date_level = date_level
        self._freq = freq
        self._format = format
        super().__init__(uri, create)
    
    @property
    def spliter(self):
        return pd.Grouper(level=self.get_levelname(self._date_level), freq=self._freq, sort=True)
    
    @property
    def namer(self):
        return lambda x: x.index.get_level_values(self.get_levelname(self._date_level))[0].strftime(self._format)
        
    def read(
        self, 
        field: str | list = None,
        code: str | list = None,
        start: str | list = None,
        stop: str = None,
        filters: list[list[tuple]] = None,
    ) -> pd.Series | pd.DataFrame:
        date_level = self.get_levelname(self._date_level)
        code_level = self.get_levelname(self._code_level)
        date_level = f'__index_level_{date_level}__' if isinstance(date_level, int) else date_level
        code_level = f'__index_level_{code_level}__' if isinstance(code_level, int) else code_level
        
        code = parse_commastr(code)
        field = parse_commastr(field)
        filters = filters or []
        start = pd.to_datetime(start or "20000104")
        stop = pd.to_datetime(stop or datetime.datetime.today().strftime(r'%Y%m%d %H%M%S'))

        if not isinstance(start, pd.DatetimeIndex):
            filters += [
                (date_level, ">=", start),
                (date_level, "<=", stop), 
            ]
            if code is not None:
                filters.append((code_level, "in", code))
            return super().read(field, filters)
        
        else:
            filters += [(date_level, "in", start)]
            if code is not None:
                filters.append((code_level, "in", code))
            return super().read(field, filters)
        
    def __str__(self) -> str:
        return super().__str__() + (f'\tindex: '
            f'<code {self.get_levelname(self._code_level)}> <date {self.get_levelname(self._date_level)}>')


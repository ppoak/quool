import datetime
import pandas as pd
from pathlib import Path
from .core.table import Table
from .core.backtrade import evaluate
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


class TradeTable(Table):

    def __init__(
        self, 
        uri: str | Path, 
        principle: float = None, 
        start_date: str | pd.Timestamp = None,
    ):
        if not Path(uri).exists():
            if principle is None or start_date is None:
                raise ValueError("principle and start_date must be specified when initiating")
            pd.DataFrame([{
                "datetime": pd.to_datetime(start_date), 
                "code": "cash", "size": principle, 
                "price": 1, "amount": principle, "commission": 0
            }]).to_parquet(uri)
        super().__init__(uri, False)
        if not self.columns.isin(["datetime", "code", "size", "price", "amount", "commission"]).all():
            raise ValueError("The table must have columns datetime, code, size, price, amount, commission")
            
    @property
    def fragments(self):
        return [self.path]
    
    @property
    def minfrag(self):
        return self.path
    
    @property
    def spliter(self):
        return super().spliter

    @property
    def namer(self):
        return super().namer
    
    def __str__(self):
        return str(self.read())
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def read(self, date: str | pd.Timestamp = None):
        if date is not None:
            date = [("datetime", "<=", pd.to_datetime(date))]
        return super().read(None, date)
    
    def update(
        self, 
        date: str | pd.Timestamp,
        code: str,
        size: int = None,
        price: float = None,
        amount: float = None,
        commission: float = 0,
        **kwargs,
    ):
        pd.concat([self.read(), pd.DataFrame([{
            "datetime": pd.to_datetime(date),
            "code": code, "size": size,
            "price": price, "amount": price * size,
            "commission": commission, **kwargs
        }])] + ([] if code == "cash" else [pd.DataFrame([{
            "datetime": pd.to_datetime(date),
            "code": "cash", "size": -size * price - commission,
            "price": 1, "commission": 0,
            "amount": -size * price - commission, **kwargs
        }])]), axis=0, ignore_index=True).sort_index().to_parquet(self.path)

    def peek(self, date: str | pd.Timestamp = None) -> pd.Series:
        df = self.read(date)
        return df.groupby("code")[["size", "amount", "commission"]].sum()
        
    def report(
        self, 
        price: pd.DataFrame, 
        benchmark: pd.Series = None
    ) -> pd.DataFrame:
        values = []
        cashes = []
        for date in price.index:
            pr = price.loc[date]
            ps = self.peek(date)
            val = pr.loc[pr.index.intersection(ps.index)].mul(ps["size"], axis=0)
            val.loc["cash"] = ps.loc["cash", "size"]
            values.append(val.sum())
            cashes.append(val.loc["cash"])
        
        value = pd.Series(values, index=price.index)
        cash = pd.Series(cashes, index=price.index)
        drawdown = value / value.cummax() - 1
        turnover_rate = (value.diff() / value.shift(1)).abs()
        info = pd.concat(
            [value, cash, drawdown, turnover_rate], axis=1, 
            keys=["value", "cash", "drawdown", "turnover_rate"]
        )
        evaluation = evaluate(value, turnover_rate, benchmark=benchmark)
        return info, evaluation
            

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


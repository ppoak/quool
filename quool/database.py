import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable
from .tools import parse_commastr, parse_date


class Table:

    def __init__(
        self,
        uri: str | Path,
        spliter: str | list | dict | pd.Series | Callable | None = None,
        namer: str | list | dict | pd.Series | Callable | None = None,
    ):
        """Create Table Class
        ======================
        uri: str or Path, the target path to the database
        spliter: str, list, dict, Series or Callable, the split function to divide 
            a dataframe into several partitions
        namer: str, list, dict, Seris, Callable, the naming function to name a
            specific dataframe partition
        """
        self.path = Path(uri).expanduser().resolve()
        self.name = self.path.stem
        if not ((spliter is None and namer is None) or 
                (spliter is not None and namer is not None)):
            raise ValueError('spliter and namer must be both None or both not None')
        self.spliter = spliter
        self.namer = namer
    
    @property
    def fragments(self):
        return sorted([f.stem for f in list(self.path.glob('**/*.parquet'))])
    
    @property
    def columns(self):
        return self._read_fragment(self.fragments[-1]).columns

    def create(self):
        self.path.mkdir(parents=True, exist_ok=True)
    
    def read(
        self,
        columns: str | list[str] | None = None,
        filters: list[list[tuple]] = None,
    ):
        """Reading Data
        ===============
        columns: str, list[str], the columns to be read
        filters: filter format can be referred to `pd.read_parquet`
        """
        df = pd.read_parquet(
            self.path, 
            engine = 'pyarrow', 
            columns = parse_commastr(columns),
            filters = filters,
        )
        return df
    
    def _read_fragment(
        self,
        fragment: list | str = None,
    ):
        """Read a given fragment
        ========================
        fragment: str or list, fragment to be provided
        """
        fragment = fragment or self.fragments
        fragment = [fragment] if not isinstance(fragment, list) else fragment
        fragment = [(self.path / frag).with_suffix('.parquet') for frag in fragment]
        return pd.read_parquet(fragment, engine='pyarrow')
    
    def _write_fragment(
        self,
        df: pd.DataFrame,
        fragment: str = None,
    ):
        """Writing data
        ================
        df: DataFrame, dataframe to be written,
        fragment: str, the specific fragment to be written
        """
        fragment = fragment or self.name
        df = df.loc[~df.index.duplicated(keep='last')].sort_index()
        if not isinstance(fragment, str):
            raise ValueError("fragment should be in string format")
        if self.spliter:
            df.groupby(self.spliter).apply(
                lambda x: x.to_parquet(
                    f"{self.path / self.namer(x)}.parquet"
            ))
        else:
            df.to_parquet((self.path / fragment).with_suffix('.parquet'))
    
    def update(
        self,
        df: pd.DataFrame,
        fragment: str = None,
    ):
        """Update the database
        =======================
        df: DataFrame, the dataframe to be saved into the database, 
            note the columns should be aligned,
        fragment: str, to specify which fragment to save the df
        """
        fragment = fragment or self.name
        df = pd.concat([self._read_fragment(fragment), df], axis=0)
        self._write_fragment(df, fragment)
    
    def remove(
        self,
        fragment: str | list = None,
    ):
        """Remove fragment
        ===================
        fragment: str or list, specify which fragment(s) to be removed
        """
        fragment = fragment or self.fragments
        fragment = [fragment] if not isinstance(fragment, list) else fragment
        for frag in fragment:
            ((self.path / frag).with_suffix('.parquet')).unlink()

    def add(
        self,
        df: pd.Series | pd.DataFrame
    ):
        """Add (a) column(s)
        ================
        df: DataFrame, data in extra column
        """
        if self.spliter:
            df.groupby(self.spliter).apply(
                lambda x: pd.concat([self._read_fragment(self.namer(x)), x], axis=1).to_parquet(
                    (self.path / self.namer(x)).with_suffix('.parquet')
                ) if self.namer(x) in self.fragments else x.reindex(self.columns, axis=1).to_parquet(
                    (self.path / self.namer(x)).with_suffix('.parquet')
            ))
            related_fragment = df.groupby(self.spliter).apply(lambda x: self.namer(x))
            columns = df.columns if isinstance(df, pd.DataFrame) else [df.name]
            for frag in set(related_fragment.to_list()) - set(self.fragments):
                d = self._read_fragment(frag)
                d[columns] = np.nan
                d.to_parquet((self.path / frag).with_suffix('.parquet'))
        else:
            for frag in self.fragments:
                pd.concat([pd._read_fragment(frag), df], axis=1).to_parquet(
                    (self.path / frag).with_suffix('.parquet')
                )
    
    def sub(
        self,
        column: str | list
    ):
        """Delet (a) column(s)
        ================
        column: str or list, column name(s)
        """
        column = parse_commastr(column)
        for frag in self.fragments:
            df = self._read_fragment(frag)
            df = df.drop(column, axis=1)
            self._write_fragment(df, frag)

    def rename(
        self,
        column: dict
    ):
        """Rename (a) column(s)
        ================
        column: dict, `{<old column name(s)>: <new column name(s)>}`
        """
        column = parse_commastr(column)
        for frag in self.fragments:
            df = self._read_fragment(frag)
            df = df.rename(columns=column)
            self._write_fragment(df, frag)
    
    def __str__(self) -> str:
        return (f'Table at <{self.path.absolute()}>\n' + 
                (f'\tfragments: <{self.fragments[0]}> - <{self.fragments[-1]}>\n' 
                 if self.fragments else '\tfragments: EMPTY\n') + 
                (f'\tcolumns: <{self.columns.to_list()}>' 
                 if self.fragments else '\tcolumns: EMPTY\n'))
    
    def __repr__(self) -> str:
        return self.__str__()


class AssetTable(Table):

    def __init__(
        self,
        uri: str | Path,
        spliter: str | list | dict | pd.Series | Callable | None = None,
        namer: str | list | dict | pd.Series | Callable | None = None,
        date_index: str = 'date',
        code_index: str = 'order_book_id',
    ):
        spliter = spliter or (lambda x: x[1].year * 100 + x[1].month)
        namer = namer or (lambda x: x.index.get_level_values(1)[0].strftime(r'%Y%m'))
        super().__init__(uri, spliter, namer)
        self.date_index = date_index
        self.code_index = code_index
    
    def read(
        self, 
        field: str | list = None,
        code: str | list = None,
        start: str | list = None,
        stop: str = None,
        filters: list[list[tuple]] = None,
    ) -> pd.Series | pd.DataFrame:
        code = parse_commastr(code)
        field = parse_commastr(field)
        filters = filters or []
        start = parse_date(start or "20000104")
        stop = parse_date(stop or datetime.datetime.today().strftime(r'%Y%m%d'))

        if isinstance(start, list) and stop is not None:
            raise ValueError("If start is list, stop should be None")
                
        elif not isinstance(start, list):
            filters += [
                (self.date_index, ">=", parse_date(start)), 
                (self.date_index, "<=", parse_date(stop)), 
            ]
            if code is not None:
                filters.append((self.code_index, "in", code))
            return super().read(field, filters)
        
        elif isinstance(start, list) and stop is None:
            filters += [(self.date_index, "in", parse_date(start))]
            if code is not None:
                filters.append((self.code_index, "in", code))
            return super().read(field, filters)
        
        else:
            raise ValueError("Invalid start, stop or field values")
    
    def update(
        self, df: pd.DataFrame, 
    ):
        fragment = self.fragments[-1]
        super().update(df, fragment)
        

class FrameTable(Table):

    def read(
        self,
        column: str | list = None,
        index: str | list = None,
        index_name: str = 'order_book_id',
    ):
        filters = None
        if index is not None:
            filters = [(index_name, "in", parse_commastr(index))]
        return super().read(parse_commastr(column), filters)


class DiffTable(AssetTable):

    def __init__(
        self,
        uri: str | Path,
        spliter: str | list | dict | pd.Series | Callable | None = None,
        namer: str | list | dict | pd.Series | Callable | None = None,
        date_index: str = 'date',
        code_index: str = 'order_book_id',
    ):
        spliter = spliter or (lambda x: x[1].year)
        namer = namer or (lambda x: x.index.get_level_values(1)[0].strftime(r'%Y'))
        super().__init__(uri, spliter, namer, date_index, code_index)

    def _diff(self, df: pd.DataFrame):
        df = df.copy()
        diff = df.groupby(self.code_index).apply(lambda x: x.diff())
        df = df.loc[(diff != 0).any(axis=1).values]
        df = df.loc[~df.index.duplicated(keep='last')].sort_index()
        return df
    
    def update(
        self, df: pd.DataFrame,
    ):
        fragment = self.fragments[-1]
        df = pd.concat([self._read_fragment(fragment), df], axis=0)
        df = self._diff(df)
        self._write_fragment(df, fragment)
    
    def add(self, df: pd.DataFrame):
        df = self._diff(df)
        super().add(df)
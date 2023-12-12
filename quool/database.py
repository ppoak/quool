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
        spliter: pd.Grouper | Callable | None = None,
        namer: pd.Grouper | Callable | None = None,
        create: bool = True,
    ):
        """Create Table Class
        ======================
        uri: str or Path, the target path to the database
        spliter: pd.Grouper or Callable, the split function to divide 
            a dataframe into several partitions
        namer: pd.Grouper or Callable, the naming function to name a
            specific dataframe partition
        """
        self.path = Path(uri).expanduser().resolve()
        self.name = self.path.stem
        self.spliter = spliter or (lambda x: 1)
        self.namer = namer or (lambda x: self.name)
        if create:
            self.path.mkdir(parents=True, exist_ok=True)
    
    @property
    def fragments(self):
        return sorted([f.stem for f in list(self.path.glob('**/*.parquet'))])
    
    @property
    def columns(self):
        if self.fragments:
            return self._read_fragment(self.fragments[-1]).columns
        else:
            return pd.Index([])

    def __fragment_path(self, fragment: str):
        return (self.path / fragment).with_suffix('.parquet')
    
    def __related_frag(self, df: pd.DataFrame | pd.Series):
        frags = df.groupby(self.spliter).apply(self.namer)
        if frags.empty:
            return []
        return frags.unique().tolist()
    
    def __update_frag(self, frag: pd.DataFrame):
        name = self.namer(frag)
        if name in self.fragments:
            frag_dat = self._read_fragment(name)
            common_idx = frag.index.intersection(frag_dat.index)
            new_idx = frag.index.difference(frag_dat.index)
            frag_dat.loc[common_idx, frag.columns] = frag.loc[common_idx, frag.columns]
            frag_dat = pd.concat([frag_dat, frag.loc[new_idx].reindex(columns=frag_dat.columns)], axis=0)
            frag_dat.to_parquet(self.__fragment_path(name))
        else:
            frag.reindex(columns=self.columns).to_parquet(self.__fragment_path(name))
    
    def __add_frag(self, frag: pd.DataFrame):
        name = self.namer(frag)
        if name in self.fragments:
            frag_dat = self._read_fragment(name)
            frag_dat = pd.concat([frag_dat, frag], axis=1, join='inner')
            frag_dat.to_parquet(self.__fragment_path(name))
        else:
            frag.reindex(columns=self.columns.tolist() 
                + frag.columns.tolist()).to_parquet(self.__fragment_path(name))
    
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
    
    def update(
        self,
        df: pd.DataFrame | pd.Series,
    ):
        """Update the database
        =======================
        df: DataFrame, the dataframe to be saved into the database, 
            note the columns should be aligned,
        fragment: str, to specify which fragment to save the df
        """        
        if isinstance(df, pd.Series):
            df = df.to_frame()
        
        if not df.columns.difference(self.columns).empty:
            raise ValueError("new field found, please add first")

        df.groupby(self.spliter).apply(self.__update_frag)

    def add(
        self,
        df: pd.Series | pd.DataFrame
    ):
        """Add (a) column(s)
        ================
        df: DataFrame, data in extra column
        """
        if isinstance(df, pd.Series):
            df = df.to_frame()
        
        if not df.columns.intersection(self.columns).empty:
            raise ValueError("existing field found, please update it")
        
        df.groupby(self.spliter).apply(self.__add_frag)
        related_fragment = self.__related_frag(df)
        columns = df.columns if isinstance(df, pd.DataFrame) else [df.name]
        for frag in set(related_fragment) - set(self.fragments):
            d = self._read_fragment(frag)
            d[columns] = np.nan
            d.to_parquet(self.__fragment_path(frag))
    
    def delete(
        self,
        index: pd.Index,
    ):
        related_fragment = self.__related_frag(pd.DataFrame(index=index))
        for frag in related_fragment:
            df = self._read_fragment(frag)
            df = df.drop(index=index.intersection(df.index))
            df.to_parquet(self.__fragment_path(frag))

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
            df.to_parquet(self.__fragment_path(frag))

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
        

class FrameTable(Table):

    def __init__(
        self, 
        uri: str | Path, 
        spliter: pd.Grouper | Callable | None = None, 
        namer: pd.Grouper | Callable | None = None,
        index_name: str = '__index_level_0__', 
        create: bool = True
    ):
        self.spliter = spliter or (lambda x: 1)
        self.namer = namer or (lambda x: self.name)
        self.index_name = index_name
        super().__init__(uri, spliter, namer, create)

    def read(
        self,
        column: str | list = None,
        index: str | list = None,
    ):
        filters = None
        if index is not None:
            filters = [(self.index_name, "in", parse_commastr(index))]
        return super().read(parse_commastr(column), filters)


class PanelTable(Table):

    def __init__(
        self,
        uri: str | Path,
        spliter: str | list | dict | pd.Series | Callable | None = None,
        namer: str | list | dict | pd.Series | Callable | None = None,
        date_index: str = '__index_level_0__',
        code_index: str = '__index_level_1__',
    ):
        spliter = spliter or pd.Grouper(level=date_index, freq='M', sort=True)
        namer = namer or (lambda x: x.index.get_level_values(date_index)[0].strftime(r'%Y%m'))
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


class DiffTable(PanelTable):

    def __init__(
        self,
        uri: str | Path,
        spliter: pd.Grouper | Callable | None = None,
        namer: pd.Grouper | Callable | None = None,
        date_index: str = '__index_level_0__',
        code_index: str = '__index_level_1__',
    ):
        spliter = spliter or pd.Grouper(level=date_index, freq='Y', sort=True)
        namer = namer or (lambda x: x.index.get_level_values(1)[0].strftime(r'%Y'))
        super().__init__(uri, spliter, namer, date_index, code_index)

    def _diff(self, df: pd.DataFrame):
        df = df.copy()
        diff = df.groupby(self.code_index).apply(lambda x: x.diff())
        df = df.loc[(diff != 0).any(axis=1).values]
        df = df.loc[~df.index.duplicated(keep='last')].sort_index()
        return df
    
    def __update_frag(self, frag: pd.DataFrame):
        name = self.namer(frag)
        if name in self.fragments:
            frag_dat = self._read_fragment(name)
            common_idx = frag.index.intersection(frag_dat.index)
            new_idx = frag.index.difference(frag_dat.index)
            frag_dat.loc[common_idx, frag.columns] = frag.loc[common_idx, frag.columns]
            frag_dat = pd.concat([frag_dat, frag.loc[new_idx].reindex(columns=frag_dat.columns)], axis=0)
            frag_dat = self._diff(frag_dat)
            frag_dat.to_parquet(self.__fragment_path(name))
        else:
            frag.reindex(columns=self.columns).to_parquet(self.__fragment_path(name))
    
    def __add_frag(self, frag: pd.DataFrame):
        name = self.namer(frag)
        if name in self.fragments:
            frag_dat = self._read_fragment(name)
            frag_dat = pd.concat([frag_dat, self._diff(frag)], axis=1, join='outer')
            frag_dat.to_parquet(self.__fragment_path(name))
        else:
            frag.reindex(columns=self.columns.tolist() 
                + frag.columns.tolist()).to_parquet(self.__fragment_path(name))    

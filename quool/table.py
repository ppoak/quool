import abc
import numpy as np
import pandas as pd
from pathlib import Path
from .util import parse_commastr


class Table(abc.ABC):

    def __init__(
        self,
        uri: str | Path,
        create: bool = False,
    ):
        self.path = Path(uri).expanduser().resolve()
        self.name = self.path.stem
        if create:
            self.path.mkdir(parents=True, exist_ok=True)
        elif not self.path.exists():
            raise NotADirectoryError(f"{self.path} does not exist.")
    
    @property
    def minfrag(self):
        if self.fragments:
            return sorted(list(self.path.glob('**/*.parquet')), key=lambda f: f.stat().st_size)[0]
    
    @property
    def spliter(self):
        return lambda _: 1
    
    @property
    def namer(self):
        return lambda _: self.name
    
    @property
    def fragments(self):
        return sorted([f.stem for f in list(self.path.glob('**/*.parquet'))])
    
    @property
    def columns(self):
        if self.fragments:
            return self._read_fragment(self.minfrag).columns
        else:
            return pd.Index([])
    
    @property
    def dtypes(self):
        if self.fragments:
            return self._read_fragment(self.minfrag).dtypes
        else:
            return pd.Series([])
    
    @property
    def dimshape(self):
        if self.fragments:
            return (self._read_fragment(self.minfrag)).dimshape
        return None
    
    def get_levelname(self, level: int | str) -> int | str:
        minfrag = self.minfrag
        if isinstance(level, int) and minfrag:
            return self._read_fragment(minfrag).index.names[level] or level
        return level

    def _fragment_path(self, fragment: str):
        return (self.path / fragment).with_suffix('.parquet')
    
    def _related_frag(self, df: pd.DataFrame | pd.Series):
        # in case of empty dataframe
        frags = df.groupby(self.spliter).apply(
            lambda x: np.nan if x.empty else self.namer(x)
        ).dropna()
        if frags.empty:
            return []
        return frags.unique().tolist()
    
    def _update_frag(self, frag: pd.DataFrame):
        # in case of empty dataframe
        if frag.empty:
            return
        
        name = self.namer(frag)
        if self.fragments and name in self.fragments:
            frag_dat = self._read_fragment(name)
            common_idx = frag.index.intersection(frag_dat.index)
            new_idx = frag.index.difference(frag_dat.index)
            upd_dat = frag.loc[common_idx].astype(self.dtypes.loc[frag.columns])
            new_dat = frag.loc[new_idx].astype(self.dtypes.loc[frag.columns])
            if not upd_dat.empty:
                frag_dat.loc[common_idx, frag.columns] = upd_dat
            if not (new_dat.empty or new_dat.isna().all().all()):
                frag_dat = pd.concat([frag_dat, 
                    new_dat.reindex(columns=frag_dat.columns).astype(self.dtypes)
                ], axis=0)
            frag_dat.sort_index().to_parquet(self._fragment_path(name))
        elif not self.fragments:
            frag.sort_index().to_parquet(self._fragment_path(name))
        else:
            frag.reindex(columns=self.columns).astype(
                self.dtypes).sort_index().to_parquet(self._fragment_path(name))
    
    def _read_fragment(
        self,
        fragment: list | str = None,
    ):
        fragment = fragment or self.fragments
        fragment = [fragment] if not isinstance(fragment, list) else fragment
        fragment = [self._fragment_path(frag) for frag in fragment]
        return pd.read_parquet(fragment, engine='pyarrow')
    
    def read(
        self,
        columns: str | list[str] | None = None,
        filters: list[list[tuple]] = None,
    ) -> pd.DataFrame | pd.Series:
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
        if isinstance(df, pd.Series):
            df = df.to_frame()
        
        if not df.columns.difference(self.columns).empty and self.fragments:
            raise ValueError("new field found, please add first")

        df.groupby(self.spliter).apply(self._update_frag)

    def add(self, column: dict):
        if not self.columns.intersection(column.keys()).empty:
            raise ValueError("existing field found, please update it")
        
        for frag in self.fragments:
            d = self._read_fragment(frag)
            d[list(column.keys())] = np.nan
            d = d.astype(column)
            d.sort_index().to_parquet(self._fragment_path(frag))
    
    def delete(self, index: pd.Index):
        related_fragment = self._related_frag(pd.Series(np.ones(len(index)), index=index))
        for frag in related_fragment:
            df = self._read_fragment(frag)
            df = df.drop(index=index.intersection(df.index))
            df.sort_index().to_parquet(self._fragment_path(frag))

    def remove(self, fragment: str | list = None):
        fragment = fragment or self.fragments
        fragment = [fragment] if not isinstance(fragment, list) else fragment
        for frag in fragment:
            ((self.path / frag).with_suffix('.parquet')).unlink()
    
    def sub(self, column: str | list):
        column = parse_commastr(column)
        for frag in self.fragments:
            df = self._read_fragment(frag)
            df = df.drop(column, axis=1)
            df.sort_index().to_parquet(self._fragment_path(frag))

    def rename(self, column: dict):
        column = parse_commastr(column)
        for frag in self.fragments:
            df = self._read_fragment(frag)
            df = df.rename(columns=column)
            df.sort_index().to_parquet(self._fragment_path(frag))
    
    def __str__(self) -> str:
        return (f'Table at <{self.path.absolute()}>\n' + 
                (f'\tfragments: <{self.fragments[0]}> - <{self.fragments[-1]}>\n' 
                 if self.fragments else '\tfragments: EMPTY\n') + 
                (f'\tcolumns: <{self.columns.to_list()}>\n' 
                 if self.fragments else '\tcolumns: EMPTY\n'))
    
    def __repr__(self) -> str:
        return self.__str__()


class ItemTable(Table):

    def read(
        self,
        column: str | list = None,
        start: str | list = None,
        stop: str = None,
        datecol: str = None,
        filters: list[list[tuple]] = None,
    ):
        filters = filters or []
        if datecol is None:
            datecol = self.get_levelname(0)
            datecol = '__index_level_0__' if isinstance(datecol, int) else datecol

        if isinstance(start, (list, pd.DatetimeIndex)):
            filters.append((datecol, "in", pd.to_datetime(start)))
        elif isinstance(start, (str, pd.Timestamp)):
            filters.append((datecol, ">=", pd.to_datetime(start)))
        if isinstance(stop, str) and not isinstance(start, list):
            filters.append((datecol, "<=", pd.to_datetime(stop)))
        
        if not len(filters):
            filters = None
        
        return super().read(parse_commastr(column), filters)


class DatetimeTable(Table):

    def __init__(
        self, 
        uri: str | Path,
        freq: str = 'M',
        format: str = '%Y%m',
        create: bool = False,
    ):
        self._freq = freq
        self._format = format
        super().__init__(uri, create)
    
    @property
    def spliter(self):
        return pd.Grouper(level=0, freq=self._freq, sort=True)
    
    @property
    def namer(self):
        return lambda x: x.index[0].strftime(self._format)
    
    def get_levelname(self) -> int | str:
        return super().get_levelname(0)
    
    def read(
        self, 
        field: str | list = None,
        start: str | list = None,
        stop: str = None,
        filters: list[list[tuple]] = None,
    ) -> pd.Series | pd.DataFrame:
        index_name = self.get_levelname()
        field = parse_commastr(field)
        filters = filters or []
        start = pd.to_datetime(start or "20000104")
        stop = pd.to_datetime(stop or 'now')

        if not isinstance(start, pd.DatetimeIndex):
            filters += [
                (index_name, ">=", start),
                (index_name, "<=", stop), 
            ]
            return super().read(field, filters)
        
        else:
            filters += [(index_name, "in", start)]
            return super().read(field, filters)


class PanelTable(Table):
    
    def __init__(
        self,
        uri: str | Path,
        code_level: str | int = 0,
        date_level: str | int = 1,
        freq: str = "ME",
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
        start = pd.to_datetime(start if start is not None else "20000104")
        stop = pd.to_datetime(stop or 'now')

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


import abc
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from .tool import parse_commastr, parse_date
from .exception import NotRequiredDimError


class DataWrapper:

    def __init__(self, datatype):
        self.datatype = datatype
    
    def __call__(self, func):
        def wrapped(*args, **kwargs):
            result = func(*args, **kwargs)

            if not isinstance(result, (pd.DataFrame, pd.Series)):
                raise ValueError("returned value is not Data type")
            return self.datatype(result)

        return wrapped


class Data(abc.ABC):

    def __init__(self, data: pd.DataFrame | pd.Series) -> None:
        if not isinstance(data, (pd.DataFrame, pd.Series)):
            raise TypeError('data must be a pandas DataFrame or Series')
        else:
            if isinstance(data, pd.DataFrame) and data.shape[1] == 1:
                data = data.squeeze()
            self.data = data
    
    def __str__(self) -> str:
        return str(self.data)
    
    def __repr__(self) -> str:
        return self.__str__()

    def __getitem__(self, key):
        return self.data.__getitem__(key)

    def __getattr__(self, name):
        return getattr(self.data, name)

    @property
    def dimshape(self):
        if isinstance(self.data, pd.Series):
            return (self.data.index.nlevels, )
        return (self.data.index.nlevels, self.data.columns.nlevels)
    
    @property
    def naxes(self):
        return len(self.data.shape)
    
    @property
    def rowdim(self):
        return self.dimshape[0]
    
    @property
    def coldim(self):
        if isinstance(self.data, pd.Series):
            return None
        return self.dimshape[1]
    
    @property
    def dimnames(self):
        if isinstance(self.data, pd.Series):
            return self.data.index.names
        return self.data.index.names + self.data.columns.names

    @property
    def rowname(self):
        return self.data.index.names

    @property
    def colname(self):
        if isinstance(self.data, pd.Series):
            return None
        return self.data.columns.names

    def swapdim(self, fromdim: int | str, todim: int | str):
        rowdim = self.rowdim
        if self.naxes > 1:
            fromdim = self.dimnames.index(fromdim) if isinstance(fromdim, str) else fromdim
            fromdim = fromdim + self.ndim if fromdim < 0 else fromdim
            todim = self.dimnames.index(todim) if isinstance(todim, str) else todim
            todim = todim + self.ndim if todim < 0 else todim
            if fromdim < rowdim and todim < rowdim:
                # this is on axis 0
                self.data = self.data.swaplevel(i=fromdim, j=todim, axis=0)
            elif fromdim >= rowdim and todim >= rowdim:
                # this is on axis 1
                self.data = self.data.swaplevel(i=fromdim, j=todim, axis=1)
            elif fromdim < rowdim and todim >= rowdim:
                # this is from axis 0 to axis 1
                self.data = self.data.unstack(level=fromdim)
                self.data = self.data.swaplevel(i=-1, j=todim - rowdim, axis=1)
            elif fromdim >= rowdim and todim < rowdim:
                # this is from axis 1 to axis 0
                self.data = self.data.stack(level=int(fromdim - rowdim))
                if self.naxes > 1:
                    self.data = self.data.swaplevel(i=-1, j=todim, axis=0)
                else:
                    self.data = self.data.swaplevel(i=-1, j=todim)
        else:
            if todim < 0:
                # when todim < 0, meaning data needs to be unstacked to extend axes
                self.data = self.data.unstack(level=fromdim)
            else:
                # when todim > 0 or todim is string type (changed to int), naively reorder
                self.data = self.data.swaplevel(i=fromdim, j=todim)

        return self

    def panelize(self):
        if self.rowdim > 1:
            levels = [self.data.index.get_level_values(i).unique() for i in range(self.rowdim)]
            if self.data.shape[0] < np.prod([level.size for level in levels]):
                self.data = self.data.reindex(pd.MultiIndex.from_product(levels), axis=0)
        if self.coldim > 1:
            levels = [self.data.columns.get_level_values(i).unique() for i in range(self.coldim)]
            if self.data.shape[1] < np.prod([level.size for level in levels]):
                self.data = self.data.reindex(pd.MultiIndex.from_product(levels), axis=1)
        return self


class Dim1Data(Data):

    def __init__(self, data: pd.Series | pd.DataFrame):
        super().__init__(data)
        if self.ndim != 1:
            raise NotRequiredDimError(1)


class Dim2Data(Data):

    def __init__(self, data: pd.Series | pd.DataFrame):
        super().__init__(data)
        if self.ndim != 2:
            raise NotRequiredDimError(2)


class Dim2Frame(Dim2Data):

    def __init__(
        self, 
        data: pd.Series | pd.DataFrame, 
        level: int | str = 0
    ):
        super().__init__(data)
        if self.rowdim == 2:
            self.swapdim(level, -1)


class Dim2Series(Dim2Data):

    def __init__(
        self, 
        data: pd.Series | pd.DataFrame, 
        level: int | str = 0
    ):
        super().__init__(data)
        if self.rowdim == 1:
            self.swapdim(-1, level)


class Dim3Data(Data):

    def __init__(self, data: pd.Series | pd.DataFrame):
        super().__init__(data)
        if self.ndim != 3:
            raise NotRequiredDimError(3)


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
    @abc.abstractmethod
    def spliter(self):
        return lambda _: 1
    
    @property
    @abc.abstractmethod
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
            return Data(self._read_fragment(self.minfrag)).dimshape
        return None
    
    @property
    def rowname(self):
        if self.fragments:
            return Data(self._read_fragment(self.minfrag)).rowname
        return None
    
    def get_levelname(self, level: int | str) -> int | str:
        minfrag = self.minfrag
        if isinstance(level, int) and minfrag:
            return Data(self._read_fragment(minfrag)).rowname[level] or level
        return level

    def __fragment_path(self, fragment: str):
        return (self.path / fragment).with_suffix('.parquet')
    
    def __related_frag(self, df: pd.DataFrame | pd.Series):
        # in case of empty dataframe
        frags = df.groupby(self.spliter).apply(
            lambda x: np.nan if x.empty else self.namer(x)
        ).dropna()
        if frags.empty:
            return []
        return frags.unique().tolist()
    
    def __update_frag(self, frag: pd.DataFrame):
        # in case of empty dataframe
        if frag.empty:
            return
        
        name = self.namer(frag)
        if name in self.fragments:
            frag_dat = self._read_fragment(name)
            common_idx = frag.index.intersection(frag_dat.index)
            new_idx = frag.index.difference(frag_dat.index)
            frag_dat.loc[common_idx, frag.columns] = frag.loc[common_idx, frag.columns]
            new_dat = frag.loc[new_idx]
            if not new_dat.empty:
                frag_dat = pd.concat([frag_dat, new_dat.reindex(columns=frag_dat.columns)], axis=0)
            elif frag_dat.empty:
                frag_dat = new_dat
            frag_dat.to_parquet(self.__fragment_path(name))
        else:
            frag.reindex(columns=self.columns).to_parquet(self.__fragment_path(name))
    
    def __add_frag(self, frag: pd.DataFrame):
        # in case of empty dataframe
        if frag.empty:
            return
        
        name = self.namer(frag)
        if name in self.fragments:
            frag_dat = self._read_fragment(name)
            # here we need to use outer join for inner join may delete data
            frag_dat = pd.concat([frag_dat, frag], axis=1, join='outer')
            frag_dat = frag_dat.astype(frag.dtypes)
            frag_dat.to_parquet(self.__fragment_path(name))
        else:
            frag.reindex(columns=self.columns.union(frag.columns)
                ).to_parquet(self.__fragment_path(name))
    
    def _read_fragment(
        self,
        fragment: list | str = None,
    ):
        fragment = fragment or self.fragments
        fragment = [fragment] if not isinstance(fragment, list) else fragment
        fragment = [self.__fragment_path(frag) for frag in fragment]
        return pd.read_parquet(fragment, engine='pyarrow')
    
    @DataWrapper(Data)
    def read(
        self,
        columns: str | list[str] | None = None,
        filters: list[list[tuple]] = None,
    ):
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
        
        if not df.columns.difference(self.columns).empty:
            raise ValueError("new field found, please add first")

        df.groupby(self.spliter).apply(self.__update_frag)

    def add(self, df: pd.Series | pd.DataFrame):
        if isinstance(df, pd.Series):
            df = df.to_frame()
        
        if not (df.columns == df.columns.astype('str')).all():
            raise TypeError("all columns or name should be string type")
        
        if not df.columns.intersection(self.columns).empty:
            raise ValueError("existing field found, please update it")
        
        df.groupby(self.spliter).apply(self.__add_frag)
        related_fragment = self.__related_frag(df)
        dtypes = self._read_fragment(related_fragment[0]).dtypes
        columns = df.columns if isinstance(df, pd.DataFrame) else [df.name]
        for frag in set(self.fragments) - set(related_fragment):
            d = self._read_fragment(frag)
            d[columns] = np.nan
            d = d.astype(dtypes)
            d.to_parquet(self.__fragment_path(frag))
    
    def delete(self, index: pd.Index):
        related_fragment = self.__related_frag(pd.DataFrame(index=index))
        for frag in related_fragment:
            df = self._read_fragment(frag)
            df = df.drop(index=index.intersection(df.index))
            df.to_parquet(self.__fragment_path(frag))

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
            df.to_parquet(self.__fragment_path(frag))

    def rename(self, column: dict):
        column = parse_commastr(column)
        for frag in self.fragments:
            df = self._read_fragment(frag)
            df = df.rename(columns=column)
            df.to_parquet(self.__fragment_path(frag))
    
    def __str__(self) -> str:
        return (f'Table at <{self.path.absolute()}>\n' + 
                (f'\tfragments: <{self.fragments[0]}> - <{self.fragments[-1]}>\n' 
                 if self.fragments else '\tfragments: EMPTY\n') + 
                (f'\tcolumns: <{self.columns.to_list()}>\n' 
                 if self.fragments else '\tcolumns: EMPTY\n'))
    
    def __repr__(self) -> str:
        return self.__str__()


class Dim2Table(Table):

    @property
    def spliter(self):
        return super().spliter
    
    @property
    def namer(self):
        return super().namer
    
    @DataWrapper(Dim2Data)
    def read(
        self,
        column: str | list = None,
        index: str | list = None,
    ):
        filters = None
        if index is not None:
            filters = [(self.get_levelname(0), "in", parse_commastr(index))]
        return super().read(parse_commastr(column), filters)


class Dim3Table(Table):
    
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
        
    @DataWrapper(Dim3Data)
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
        start = parse_date(start or "20000104")
        stop = parse_date(stop or datetime.datetime.today().strftime(r'%Y%m%d'))

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


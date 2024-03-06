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
        if name in self.fragments:
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
            frag_dat.to_parquet(self._fragment_path(name))
        else:
            frag.reindex(columns=self.columns).astype(
                self.dtypes).to_parquet(self._fragment_path(name))
    
    def _add_frag(self, frag: pd.DataFrame):
        # in case of empty dataframe
        if frag.empty:
            return
        
        name = self.namer(frag)
        if name in self.fragments:
            frag_dat = self._read_fragment(name)
            # here we need to use outer join for inner join may delete data
            frag_dat = pd.concat([frag_dat, frag], axis=1, join='outer')
            frag_dat = frag_dat.astype(frag.dtypes)
            frag_dat.to_parquet(self._fragment_path(name))
        else:
            frag.reindex(columns=self.columns.union(frag.columns)
                ).to_parquet(self._fragment_path(name))
    
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
        
        if not df.columns.difference(self.columns).empty:
            raise ValueError("new field found, please add first")

        df.groupby(self.spliter).apply(self._update_frag)

    def add(self, df: pd.Series | pd.DataFrame):
        if isinstance(df, pd.Series):
            df = df.to_frame()
        
        if not (df.columns == df.columns.astype('str')).all():
            raise TypeError("all columns or name should be string type")
        
        if not df.columns.intersection(self.columns).empty:
            raise ValueError("existing field found, please update it")
        
        df.groupby(self.spliter).apply(self._add_frag)
        related_fragment = self._related_frag(df)
        dtypes = self._read_fragment(related_fragment[0]).dtypes
        columns = df.columns if isinstance(df, pd.DataFrame) else [df.name]
        for frag in set(self.fragments) - set(related_fragment):
            d = self._read_fragment(frag)
            d[columns] = np.nan
            d = d.astype(dtypes)
            d.to_parquet(self._fragment_path(frag))
    
    def delete(self, index: pd.Index):
        related_fragment = self._related_frag(pd.DataFrame(index=index))
        for frag in related_fragment:
            df = self._read_fragment(frag)
            df = df.drop(index=index.intersection(df.index))
            df.to_parquet(self._fragment_path(frag))

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
            df.to_parquet(self._fragment_path(frag))

    def rename(self, column: dict):
        column = parse_commastr(column)
        for frag in self.fragments:
            df = self._read_fragment(frag)
            df = df.rename(columns=column)
            df.to_parquet(self._fragment_path(frag))
    
    def __str__(self) -> str:
        return (f'Table at <{self.path.absolute()}>\n' + 
                (f'\tfragments: <{self.fragments[0]}> - <{self.fragments[-1]}>\n' 
                 if self.fragments else '\tfragments: EMPTY\n') + 
                (f'\tcolumns: <{self.columns.to_list()}>\n' 
                 if self.fragments else '\tcolumns: EMPTY\n'))
    
    def __repr__(self) -> str:
        return self.__str__()

import abc
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from .tool import parse_commastr, parse_date, DimFormatter


class Table(abc.ABC):
    """
    A class for managing large datasets in a fragmented, file-based structure using Parquet files.

    Attributes:
        path (Path): The file path to the database or data storage directory.
        name (str): The base name of the data storage directory.
        spliter (Callable): Function to divide a DataFrame into several partitions.
        namer (Callable): Function to name a specific DataFrame partition.

    Methods:
        __init__(self, uri, spliter, namer, create): Initializes the Table object.
        fragments (property): Lists all data fragments in the storage directory.
        columns (property): Lists column names in the last fragment.
        _read_fragment(self, fragment): Reads a specified fragment from storage.
        read(self, columns, filters): Reads data with optional filtering.
        update(self, df): Updates the database with new data.
        add(self, df): Adds new columns to the database.
        delete(self, index): Deletes rows from the database.
        remove(self, fragment): Removes specified fragments from the database.
        sub(self, column): Deletes specified columns from the database.
        rename(self, column): Renames columns in the database.

    Example:
        table = Table(uri="path/to/data", create=True)
        table.add(df=my_dataframe)
    """

    def __init__(
        self,
        uri: str | Path,
        create: bool = False,
    ):
        """
        Initializes the Table object.

        Args:
            uri (str | Path): Path to the database or data storage directory.
            spliter (pd.Grouper | Callable, optional): Function to divide a DataFrame into partitions.
            namer (pd.Grouper | Callable, optional): Function to name DataFrame partitions.
            create (bool, optional): Whether to create the directory if it doesn't exist.
        """
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
        """
        Lists the names of all data fragments in the storage directory.

        Returns:
            List[str]: Names of the data fragments.
        """
        return sorted([f.stem for f in list(self.path.glob('**/*.parquet'))])
    
    @property
    def columns(self):
        """
        Lists the column names in the last data fragment.

        Returns:
            pd.Index: Column names.
        """
        if self.fragments:
            return self._read_fragment(self.minfrag).columns
        else:
            return pd.Index([])
    
    @property
    def dtypes(self):
        """
        Lists the dtypes of columns in the last data fragment.

        Returns:
            pd.Series: Column dtypes.
        """
        if self.fragments:
            return self._read_fragment(self.minfrag).dtypes
        else:
            return pd.Series([])
    
    @property
    def ndims(self):
        if self.fragments:
            return DimFormatter(self._read_fragment(self.minfrag)).ndims
        return None
    
    @property
    def dimshape(self):
        if self.fragments:
            return DimFormatter(self._read_fragment(self.minfrag)).dimshape
        return None
    
    @property
    def rowname(self):
        if self.fragments:
            return DimFormatter(self._read_fragment(self.minfrag)).rowname
        return None
    
    def get_levelname(self, level: int | str) -> int | str:
        minfrag = self.minfrag
        if isinstance(level, int) and minfrag:
            return DimFormatter(self._read_fragment(minfrag)).rowname[level] or level
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
        """
        Reads a specified fragment from storage.

        Args:
            fragment (list | str, optional): The name(s) of the fragment(s) to read.

        Returns:
            pd.DataFrame: The data from the specified fragment(s).
        """
        fragment = fragment or self.fragments
        fragment = [fragment] if not isinstance(fragment, list) else fragment
        fragment = [self.__fragment_path(frag) for frag in fragment]
        return pd.read_parquet(fragment, engine='pyarrow')
    
    def read(
        self,
        columns: str | list[str] | None = None,
        filters: list[list[tuple]] = None,
    ):
        """
        Reads data from storage with optional filtering.

        Args:
            columns (str | list[str], optional): The columns to read.
            filters (list[list[tuple]], optional): Filters for reading the data.

        Returns:
            pd.DataFrame: The read data.
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
        """
        Updates the database with new data.

        Args:
            df (pd.DataFrame | pd.Series): The DataFrame or Series to update the database with.
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
        """
        Adds new columns to the database.

        Args:
            df (pd.Series | pd.DataFrame): The data to add to the database.
        """
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
    
    def delete(
        self,
        index: pd.Index,
    ):
        """
        Deletes rows from the database.

        Args:
            index (pd.Index): Index of the rows to delete.
        """
        related_fragment = self.__related_frag(pd.DataFrame(index=index))
        for frag in related_fragment:
            df = self._read_fragment(frag)
            df = df.drop(index=index.intersection(df.index))
            df.to_parquet(self.__fragment_path(frag))

    def remove(
        self,
        fragment: str | list = None,
    ):
        """
        Removes specified fragments from the database.

        Args:
            fragment (str | list, optional): The fragment(s) to remove.
        """
        fragment = fragment or self.fragments
        fragment = [fragment] if not isinstance(fragment, list) else fragment
        for frag in fragment:
            ((self.path / frag).with_suffix('.parquet')).unlink()
    
    def sub(
        self,
        column: str | list
    ):
        """
        Deletes specified columns from the database.

        Args:
            column (str | list): The column(s) to delete.
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
        """
        Renames columns in the database.

        Args:
            column (dict): A mapping of old column names to new column names.
        """
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
    """
    A subclass of Table designed for handling data frames with enhanced index management.

    Attributes:
        spliter (Callable): Function to divide a DataFrame into several partitions.
        namer (Callable): Function to name a specific DataFrame partition.
        index_name (str): Name of the index column used in the stored data frames.

    Methods:
        __init__(self, uri, spliter, namer, index_name, create): Initializes the FrameTable object.
        read(self, column, index): Reads data with optional column and index filtering.

    Example:
        frame_table = FrameTable(uri="path/to/data", create=True)
        data = frame_table.read(column="my_column", index=["index1", "index2"])
    """

    def __init__(
        self, 
        uri: str | Path, 
        create: bool = True
    ):
        """
        Initializes the FrameTable object.

        Args:
            uri (str | Path): Path to the database or data storage directory.
            spliter (pd.Grouper | Callable, optional): Function to divide a DataFrame into partitions.
            namer (pd.Grouper | Callable, optional): Function to name DataFrame partitions.
            index_name (str, optional): Name of the index column in stored data frames.
            create (bool, optional): Whether to create the directory if it doesn't exist.
        """
        super().__init__(uri, create)

    @property
    def spliter(self):
        return super().spliter
    
    @property
    def namer(self):
        return super().namer
    
    def read(
        self,
        column: str | list = None,
        index: str | list = None,
    ):
        """
        Reads data from the storage with optional column and index filtering.

        Args:
            column (str | list, optional): The column(s) to read from the data.
            index (str | list, optional): The index value(s) to filter the data.

        Returns:
            pd.DataFrame: The filtered data frame.
        """
        filters = None
        if index is not None:
            filters = [(self.get_levelname(0), "in", parse_commastr(index))]
        return super().read(parse_commastr(column), filters)


class Dim3Table(Table):
    """
    A specialized subclass of Table for handling panel data with time and categorical indexing.

    Attributes:
        date_level (str): The name of the index column representing dates.
        code_level (str): The name of the index column representing categorical dimensions, like stock codes.

    Methods:
        __init__(self, uri, spliter, namer, date_level, code_level): Initializes the PanelTable object.
        read(self, field, code, start, stop, filters): Reads data with optional field, code, time, and other filters.

    Example:
        panel_table = PanelTable(uri="path/to/panel/data")
        data = panel_table.read(field="price", code="AAPL", start="2020-01-01", stop="2020-12-31")
    """

    def __init__(
        self,
        uri: str | Path,
        code_level: str | int = 0,
        date_level: str | int = 1,
        freq: str = "M",
        format: str = r"%Y%m",
        create: bool = False,
    ):
        """
        Initializes the PanelTable object.

        Args:
            uri (str | Path): Path to the database or data storage directory.
            spliter (str | list | dict | pd.Series | Callable, optional): Function or parameter to divide the DataFrame into partitions based on time.
            namer (str | list | dict | pd.Series | Callable, optional): Function or parameter to name DataFrame partitions.
            date_level (str, optional): Name of the index column for dates.
            code_level (str, optional): Name of the index column for categorical dimensions.
        """
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
        """
        Reads data from the storage with optional filtering by field, code, time range, and other conditions.

        Args:
            field (str | list, optional): The field(s) to read from the data.
            code (str | list, optional): The code(s) to filter the data.
            start (str | list, optional): The start date(s) for the time range filter.
            stop (str, optional): The end date for the time range filter.
            filters (list[list[tuple]], optional): Additional filters for reading the data.

        Returns:
            pd.Series | pd.DataFrame: The filtered data.
        """
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

import time
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy
from joblib import Parallel, delayed


class ParquetManager:
    """Class to manage Parquet database operations, including data insertions, updates, and reading.

    Attributes:
        path (Path): Base path for storing parquet files.
        partitions (list): List of existing partitions.
        name (str): Name of the database.
        partition (str): Column used for partitioning the data.
        index (list): Column(s) used for indexing to avoid duplicates.
    """

    def __init__(
        self, 
        path: str | Path, 
        index: str | list = None, 
        partition: str = None,
    ):
        """
        Initializes the ParquetManager with a base path, partition column(s), and index column(s).

        Args:
            path (str or Path): The base directory where parquet files will be stored.
            index (str, optional): Column(s) used for indexing to avoid duplicates.
            partition (str, optional): Column used for partitioning the data. If None, it will be automatically detected for existing data.
        """
        if not isinstance(partition, (str, type(None))):
            raise ValueError("partition must be a string.")
        
        self.path = Path(path).expanduser().resolve()
        self.path.mkdir(parents=True, exist_ok=True)
        self.name = self.path.name
        self.partition = self._auto_detect_partition_col() if partition is None else partition

        if self.partitions and partition and partition != self.partition:
            raise ValueError("Partition conflicts with existing data.")
        elif not self.partitions and not partition:
            raise ValueError("Partition must be specified when no existing data is present.")
        
        self.index = [index] if isinstance(index, str) else index
    
    @property
    def required_cols(self):
        """Returns the required columns for the data."""
        return [self.index] if isinstance(self.index, str) else self.index + [self.partition]
    
    @property
    def partitions(self):
        """Returns the list of existing partitions."""
        return sorted(list(self.path.iterdir()))

    def _auto_detect_partition_col(self):
        """Automatically detects the partition column based on file naming convention."""
        for file in self.partitions:
            parts = file.name.split("__", maxsplit=1)
            if len(parts) > 1:
                return parts[1].split("=")[0]
            return None

    def _generate_partition(self, data, partition=None):
        """Generates partition values for the data."""
        data = data.copy()
        if self.partition:
            if self.partition not in data.columns and partition is None:
                raise ValueError("partition must be provided when partition_col is specified.")
            
            # Check for column name conflict
            if self.partition in data.columns and partition is not None:
                raise ValueError(f"Column name '{self.partition}' already exists, set partition to None.")

            if isinstance(partition, str):
                # Map partition values from an existing column
                if partition not in data.columns:
                    raise ValueError(f"The specified partition column '{partition}' does not exist in new_data.")
                data[self.partition] = data[partition].astype(str)
            elif isinstance(partition, (pd.Series, list, tuple, pd.Index)):
                # Use provided partition values
                data[self.partition] = pd.Series(partition, index=data.index).astype(str)
            elif callable(partition):
                # Generate partition values using a callable function
                data[self.partition] = partition(data).astype(str)
            elif partition is not None:
                raise ValueError("Invalid partition input. It should be either a string representing a column name, a pd.Series, array-like, or a callable.")
        else:
            if partition is not None:
                raise ValueError("partition must be None when partition_col is not specified.")
        return data

    def _generate_filters(self, kwargs: dict):
        """Generates filters for the data based on keyword arguments."""
        filters = []
        operator_dict = {
            "eq": "==", "ne": "!=",
            "gt": ">", "ge": ">=", 
            "lt": "<", "le": "<=",
            "in": "in", "notin": "not in"
        }
        for key, value in kwargs.items():
            key = key.split("__")
            column = key[0]
            operator = operator_dict[key[1]] if len(key) > 1 else "=="

            if ("date" in column) or ("time" in column):
                value = pd.to_datetime(value)

            filters.append((column, operator, value))
        
        return None if not filters else filters

    def _get_partition_path(self, partition_value: str):
        """Returns the path from the partition value."""
        return self.path / f"{self.name}__{self.partition}={partition_value}.parquet"
    
    def _get_partition_value(self, partition_path: str | Path):
        """Returns the value from the partition path."""
        parts = Path(partition_path).stem.split("__", maxsplit=1)
        return parts[1].split("=")[1] if len(parts) > 1 else None
    
    def dropcol(self, columns: list | str, njobs: int = 4):
        """
        Drops specified columns or rows from the Parquet database.

        Args:
            columns (str or list) List of column names to drop.
            njobs(int, optional): Number of jobs to run in parallel. Defaults to 4.

        Raises:
            ValueError: If the current parquet database is readonly or if the specified columns are required.

        Returns:
            None
        """
        if not self.index:
            raise ValueError("Current parquet database is readonly! Enable write by setting index")

        columns = [columns] if isinstance(columns, str) else columns
        if pd.Index(columns).isin(pd.Index(self.required_cols)).any():
            raise ValueError("Cannot drop required columns.")

        def process_partition(partition):
            df = pd.read_parquet(partition)
            df.drop(columns=columns, inplace=True)
            df.to_parquet(partition)

        Parallel(n_jobs=njobs, backend="threading")(
            delayed(process_partition)(partition) for partition in self.partitions
        )
    
    def droprow(self, njobs: int = 4, **kwargs):
        """
        Drops specified rows from the Parquet database.

        Args:
            kwargs: Keyword arguments to filter rows to be dropped.
            njobs(int, optional): Number of jobs to run in parallel. Defaults to 4.

        Returns:
            None
        """
        if not self.index:
            raise ValueError("Current parquet database is readonly! Enable write by setting index")

        def process_partition(partition):
            if not pd.read_parquet(partition, filters=filters, columns=self.index).empty:
                df = pd.read_parquet(partition)
                conditions = pd.Series(np.ones_like(df.index), index=df.index, dtype="bool")
                for key, value in kwargs.items():
                    keyops = key.split("__")
                    if len(keyops) == 1:
                        key, ops = keyops[0], "eq"
                    else:
                        key, ops = keyops
                    ops = "in" if ops == "notin" else ops
                    rev = True if ops == "notin" else False
                    conditions = conditions & (getattr(df[key], ops)(value) if not rev else ~getattr(df[key], ops)(value))
                df[~conditions].to_parquet(partition, index=False)

        filters = self._generate_filters(kwargs)
        Parallel(n_jobs=njobs, backend="threading")(
            delayed(process_partition)(partition) for partition in self.partitions
        )

    def renamecol(self, njobs: int = 4, **kwargs):
        """
        Renames specified columns or index columns in the Parquet database.

        Args:
            kwargs: Dictionary mapping old column names to new column names.

        Returns:
            None
        """
        if not self.index:
            raise ValueError("Current parquet database is readonly! Enable write by setting index")
        
        self.index = [kwargs.get(i, i) for i in self.index]
        self.partition = kwargs.get(self.partition, self.partition)

        def process_partition(partition):
            df = pd.read_parquet(partition)
            df.rename(columns=kwargs, inplace=True)
            df.to_parquet(partition)

        Parallel(n_jobs=njobs, backend="threading")(
            delayed(process_partition)(partition) for partition in self.partitions
        )

    def merge(
        self,
        data: pd.DataFrame,
        partition: str | pd.Series | pd.Index | list = None,
        on: str = None,
        njobs: int = 4
    ):
        """
        Merges new data into the Parquet database, ensuring data is partitioned correctly and duplicates based on the index columns are removed.

        Args:
            data (pd.DataFrame): New data to be merged into the Parquet database.
            partition (str, pd.Series, list, tuple, Index, or callable, optional): A string representing a column name,
                a Series, array-like, or a callable used to determine the partition value(s) for the new data.
                If partition_col is None, partition must also be None.
            njobs (int): Maximum number of workers for parallel processing. Default is 4.

        Raises:
            ValueError: If partition is not valid or if partition_col is None and partition is provided.
        """
        if not self.index:
            raise ValueError("Current parquet database is readonly! Enable write by setting index")
        
        data = self._generate_partition(data, partition)
        data = data.drop_duplicates(subset=self.index, keep='last')
        
        # Helper function for processing a single partition
        def process_partition(partition_path):
            partition_value = self._get_partition_value(partition_path)

            if data[data[self.partition] == partition_value].empty:
                if partition_path.exists():
                    existing_data = pd.read_parquet(partition_path)
                    columns = data.columns.difference(
                        [self.index] if isinstance(self.index, str) else self.index + [self.partition]
                    )
                    existing_data[columns] = np.nan
                    dtypes = data.dtypes
                    dtypes = dtypes.loc[~dtypes.index.isin(columns)].to_dict()
                    existing_data = existing_data.astype(dtypes)
                    existing_data.to_parquet(partition_path, index=False)
                return

            # Data for partition value is not empty
            if partition_path.exists():
                # Load existing data and combine
                existing_data = pd.read_parquet(partition_path)
                combined_data = pd.merge(
                    existing_data, data[data[self.partition] == partition_value], 
                    on=on or self.index
                )
                combined_data = combined_data.drop_duplicates(subset=self.index, keep='last')
            else:
                combined_data = data[data[self.partition] == partition_value]

            # Save combined data
            combined_data.to_parquet(partition_path, index=False)

        # Use joblib.Parallel for parallel processing
        if self.partition:
            Parallel(n_jobs=njobs, backend="threading")(
                delayed(process_partition)(partition_path) 
                for partition_path in self.partitions
            )
            
        else:
            # No partitioning, save data to a single file
            partition_path = self.path / f"{self.name}.parquet"
            if partition_path.exists():
                existing_data = pd.read_parquet(partition_path)
                combined_data = pd.merge(
                    existing_data, data, 
                    on=on or self.index
                )
                combined_data = combined_data.drop_duplicates(subset=self.index, keep='last')
            else:
                combined_data = data
            combined_data.to_parquet(partition_path, index=False)

    def upsert(
        self,
        data: pd.DataFrame,
        partition: str | pd.Series | pd.Index | list = None,
        njobs: int = 4
    ):
        """
        Updates or inserts new data into the Parquet database, ensuring data is partitioned correctly
        and duplicates based on the index columns are removed. This method uses joblib for faster I/O.

        Args:
            data (pd.DataFrame): New data to be inserted or updated in the Parquet database.
            partition (str, pd.Series, list, tuple, Index, or callable, optional): A string representing a column name,
                a Series, array-like, or a callable used to determine the partition value(s) for the new data.
                If partition_col is None, partition must also be None.
            njobs (int, optional): Number of parallel jobs to run for I/O operations. Defaults to 4.
        Raises:
            ValueError: If the partition column is not specified when partition is None.
        """
        if not self.index:
            raise ValueError("Current parquet database is readonly! Enable write by setting index")

        if not pd.Index(self.index).isin(data.columns).all():
            raise ValueError("Malformed data, please check your input")
        
        data = self._generate_partition(data, partition)
        data = data.drop_duplicates(subset=self.index, keep='last')

        # Helper function for processing a single partition
        def process_partition(partition_value):
            partition_path = self._get_partition_path(partition_value)

            if partition_path.exists():
                # Load existing data and combine
                existing_data = pd.read_parquet(partition_path)
                combined_data = pd.concat([existing_data, data[data[self.partition] == partition_value]])
                combined_data = combined_data.drop_duplicates(subset=self.index, keep='last')
            else:
                combined_data = data[data[self.partition] == partition_value]

            # Save combined data
            combined_data.to_parquet(partition_path, index=False)

        # Use joblib.Parallel for parallel processing
        if self.partition:
            partition_values = data[self.partition].unique()
            Parallel(n_jobs=njobs, backend="threading")(
                delayed(process_partition)(partition_value) for partition_value in partition_values
            )
        else:
            # No partitioning, save data to a single file
            partition_path = self.path / f"{self.name}.parquet"
            if partition_path.exists():
                existing_data = pd.read_parquet(partition_path)
                combined_data = pd.concat([existing_data, data])
                combined_data = combined_data.drop_duplicates(subset=self.index, keep='last') if self.index else combined_data
            else:
                combined_data = data
            combined_data.to_parquet(partition_path, index=False)
    
    def read(self, index=None, columns=None, pivot=None, **kwargs):
        """
        Reads data from the Parquet database, optionally filtering, indexing, and pivoting.

        Args:
            index (str or list of str, optional): Column(s) to use as the index for the DataFrame.
            columns (str or list of str, optional): Column(s) to include in the resulting DataFrame.
            pivot (str, optional): Column to use for populating pivot values. If None, no pivoting is performed.
            kwargs: Filters expressed as keyword arguments (e.g., column=value, column__gt=value).

        Returns:
            pd.DataFrame: The combined data read from the Parquet files, with optional indexing and pivoting.

        Raises:
            ValueError: If `pivot` is specified but `index` or `columns` are not.
        """
        # Ensure `index_col` and `columns` are lists for consistency
        index = index or []
        columns = columns or []
        pivot = pivot or []
        index = [index] if isinstance(index, str) else index
        columns = [columns] if isinstance(columns, str) else columns
        pivot = [pivot] if isinstance(pivot, str) else pivot
        read_columns = index + columns
        read_columns = read_columns if not pivot else read_columns + pivot
        read_columns = None if not read_columns else read_columns

        # Generate filters
        filters = self._generate_filters(kwargs)
        # Read data with filters
        df = pd.read_parquet(self.path, columns=read_columns, filters=filters)

        # Handle pivoting if specified
        if pivot:
            if not index or not columns:
                raise ValueError("Both `index_col` and `columns` must be specified when `pivot` is used.")
            df = df.pivot(index=index, columns=columns, values=pivot[0] if len(pivot) == 1 else pivot)
        else:
            # Set index if specified
            if index:
                df = df.set_index(index)

        return df.sort_index()
        
    def __str__(self):
        file_count = len(self.partitions)
        file_size = sum(f.stat().st_size for f in self.partitions)
        if file_size > 1024**3:
            file_size_str = f"{file_size / 1024**3:.2f} GB"
        elif file_size > 1024**2:
            file_size_str = f"{file_size / 1024**2:.2f} MB"
        else:
            file_size_str = f"{file_size} KB"
        
        partition_range = "N/A"
        if self.partitions:
            min_file = self.partitions[0].stem
            max_file = self.partitions[-1].stem
            if '=' not in min_file or '=' not in max_file:
                partition_range = min_file + " - " + max_file
            else:
                partition_range = f"{min_file.split('=')[1]} - {max_file.split('=')[1]}"

        # Display schema info from one file if available
        first_file = self.partitions[np.argmin(file_size)] if file_count else None
        column_info = "N/A"
        if first_file:
            try:
                df = pd.read_parquet(first_file)
                column_info = ", ".join([f"{col} ({df[col].dtype})" for col in df.columns])
            except Exception as e:
                column_info = f"Error retrieving columns: {e}"

        return (f"Parquet Database '{self.name}' at '{self.path}':\n"
                f"File Count: {file_count}\n"
                f"Partition Column: {self.partition}\n"
                f"Partition Range: {partition_range}\n"
                f"Columns: {column_info}\n"
                f"Total Size: {file_size_str}")
    
    def __repr__(self):
        return self.__str__()


class Crawler:

    def __init__(self, proxies: list[dict] | dict = None):
        self.proxies = proxies

    def read_realtime(self):
        url = "https://82.push2.eastmoney.com/api/qt/clist/get"
        params = {
            "pn": "1",
            "pz": "50000",
            "po": "1",
            "np": "1",
            "ut": "bd1d9ddb04089700cf9c27f6f7426281",
            "fltt": "2",
            "invt": "2",
            "fid": "f3",
            "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
            "fields": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,"
            "f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152",
            "_": "1623833739532",
        }
        r = requests.get(url, timeout=15, params=params)
        data_json = r.json()
        if not data_json["data"]["diff"]:
            return pd.DataFrame()
        temp_df = pd.DataFrame(data_json["data"]["diff"])
        temp_df.columns = [
            "_",
            "最新价",
            "涨跌幅",
            "涨跌额",
            "成交量",
            "成交额",
            "振幅",
            "换手率",
            "市盈率-动态",
            "量比",
            "5分钟涨跌",
            "代码",
            "_",
            "名称",
            "最高",
            "最低",
            "今开",
            "昨收",
            "总市值",
            "流通市值",
            "涨速",
            "市净率",
            "60日涨跌幅",
            "年初至今涨跌幅",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
            "-",
        ]
        temp_df.reset_index(inplace=True)
        temp_df["index"] = temp_df.index + 1
        temp_df.rename(columns={"index": "序号"}, inplace=True)
        temp_df = temp_df[
            [
                "序号",
                "代码",
                "名称",
                "最新价",
                "涨跌幅",
                "涨跌额",
                "成交量",
                "成交额",
                "振幅",
                "最高",
                "最低",
                "今开",
                "昨收",
                "量比",
                "换手率",
                "市盈率-动态",
                "市净率",
                "总市值",
                "流通市值",
                "涨速",
                "5分钟涨跌",
                "60日涨跌幅",
                "年初至今涨跌幅",
            ]
        ]
        temp_df["最新价"] = pd.to_numeric(temp_df["最新价"], errors="coerce")
        temp_df["涨跌幅"] = pd.to_numeric(temp_df["涨跌幅"], errors="coerce")
        temp_df["涨跌额"] = pd.to_numeric(temp_df["涨跌额"], errors="coerce")
        temp_df["成交量"] = pd.to_numeric(temp_df["成交量"], errors="coerce")
        temp_df["成交额"] = pd.to_numeric(temp_df["成交额"], errors="coerce")
        temp_df["振幅"] = pd.to_numeric(temp_df["振幅"], errors="coerce")
        temp_df["最高"] = pd.to_numeric(temp_df["最高"], errors="coerce")
        temp_df["最低"] = pd.to_numeric(temp_df["最低"], errors="coerce")
        temp_df["今开"] = pd.to_numeric(temp_df["今开"], errors="coerce")
        temp_df["昨收"] = pd.to_numeric(temp_df["昨收"], errors="coerce")
        temp_df["量比"] = pd.to_numeric(temp_df["量比"], errors="coerce")
        temp_df["换手率"] = pd.to_numeric(temp_df["换手率"], errors="coerce")
        temp_df["市盈率-动态"] = pd.to_numeric(temp_df["市盈率-动态"], errors="coerce")
        temp_df["市净率"] = pd.to_numeric(temp_df["市净率"], errors="coerce")
        temp_df["总市值"] = pd.to_numeric(temp_df["总市值"], errors="coerce")
        temp_df["流通市值"] = pd.to_numeric(temp_df["流通市值"], errors="coerce")
        temp_df["涨速"] = pd.to_numeric(temp_df["涨速"], errors="coerce")
        temp_df["5分钟涨跌"] = pd.to_numeric(temp_df["5分钟涨跌"], errors="coerce")
        temp_df["60日涨跌幅"] = pd.to_numeric(temp_df["60日涨跌幅"], errors="coerce")
        temp_df["年初至今涨跌幅"] = pd.to_numeric(
            temp_df["年初至今涨跌幅"], errors="coerce"
        )
        return temp_df

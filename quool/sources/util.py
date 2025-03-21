import re
import uuid
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from contextlib import contextmanager
from typing import Union, List, Optional


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
        self.partition = (
            self._auto_detect_partition_col() if partition is None else partition
        )

        if self.partitions and partition and partition != self.partition:
            raise ValueError("Partition conflicts with existing data.")
        elif not self.partitions and not partition:
            raise ValueError(
                "Partition must be specified when no existing data is present."
            )

        self.index = [index] if isinstance(index, str) else index

    @property
    def required_cols(self):
        """Returns the required columns for the data."""
        return (
            [self.index]
            if isinstance(self.index, str)
            else self.index + [self.partition]
        )

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
                raise ValueError(
                    "partition must be provided when partition_col is specified."
                )

            # Check for column name conflict
            if self.partition in data.columns and partition is not None:
                raise ValueError(
                    f"Column name '{self.partition}' already exists, set partition to None."
                )

            if isinstance(partition, str):
                # Map partition values from an existing column
                if partition not in data.columns:
                    raise ValueError(
                        f"The specified partition column '{partition}' does not exist in new_data."
                    )
                data[self.partition] = data[partition].astype(str)
            elif isinstance(partition, (pd.Series, list, tuple, pd.Index)):
                # Use provided partition values
                data[self.partition] = pd.Series(partition, index=data.index).astype(
                    str
                )
            elif callable(partition):
                # Generate partition values using a callable function
                data[self.partition] = partition(data).astype(str)
            elif partition is not None:
                raise ValueError(
                    "Invalid partition input. It should be either a string representing a column name, a pd.Series, array-like, or a callable."
                )
        else:
            if partition is not None:
                raise ValueError(
                    "partition must be None when partition_col is not specified."
                )
        return data

    def _generate_filters(self, kwargs: dict):
        """Generates filters for the data based on keyword arguments."""
        filters = []
        operator_dict = {
            "eq": "==",
            "ne": "!=",
            "gt": ">",
            "ge": ">=",
            "lt": "<",
            "le": "<=",
            "in": "in",
            "notin": "not in",
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
            raise ValueError(
                "Current parquet database is readonly! Enable write by setting index"
            )

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
            raise ValueError(
                "Current parquet database is readonly! Enable write by setting index"
            )

        def process_partition(partition):
            if not pd.read_parquet(
                partition, filters=filters, columns=self.index
            ).empty:
                df = pd.read_parquet(partition)
                conditions = pd.Series(
                    np.ones_like(df.index), index=df.index, dtype="bool"
                )
                for key, value in kwargs.items():
                    keyops = key.split("__")
                    if len(keyops) == 1:
                        key, ops = keyops[0], "eq"
                    else:
                        key, ops = keyops
                    ops = "in" if ops == "notin" else ops
                    rev = True if ops == "notin" else False
                    conditions = conditions & (
                        getattr(df[key], ops)(value)
                        if not rev
                        else ~getattr(df[key], ops)(value)
                    )
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
            raise ValueError(
                "Current parquet database is readonly! Enable write by setting index"
            )

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
        njobs: int = 4,
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
            raise ValueError(
                "Current parquet database is readonly! Enable write by setting index"
            )

        data = self._generate_partition(data, partition)
        data = data.drop_duplicates(subset=self.index, keep="last")

        # Helper function for processing a single partition
        def process_partition(partition_path):
            partition_value = self._get_partition_value(partition_path)

            if data[data[self.partition] == partition_value].empty:
                if partition_path.exists():
                    existing_data = pd.read_parquet(partition_path)
                    columns = data.columns.difference(
                        [self.index]
                        if isinstance(self.index, str)
                        else self.index + [self.partition]
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
                    existing_data,
                    data[data[self.partition] == partition_value],
                    on=on or self.index,
                )
                combined_data = combined_data.drop_duplicates(
                    subset=self.index, keep="last"
                )
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
                combined_data = pd.merge(existing_data, data, on=on or self.index)
                combined_data = combined_data.drop_duplicates(
                    subset=self.index, keep="last"
                )
            else:
                combined_data = data
            combined_data.to_parquet(partition_path, index=False)

    def upsert(
        self,
        data: pd.DataFrame,
        partition: str | pd.Series | pd.Index | list = None,
        njobs: int = 4,
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
            raise ValueError(
                "Current parquet database is readonly! Enable write by setting index"
            )

        if not pd.Index(self.index).isin(data.columns).all():
            raise ValueError("Malformed data, please check your input")

        data = self._generate_partition(data, partition)
        data = data.drop_duplicates(subset=self.index, keep="last")

        # Helper function for processing a single partition
        def process_partition(partition_value):
            partition_path = self._get_partition_path(partition_value)

            if partition_path.exists():
                # Load existing data and combine
                existing_data = pd.read_parquet(partition_path)
                combined_data = pd.concat(
                    [existing_data, data[data[self.partition] == partition_value]]
                )
                combined_data = combined_data.drop_duplicates(
                    subset=self.index, keep="last"
                )
            else:
                combined_data = data[data[self.partition] == partition_value]

            # Save combined data
            combined_data.to_parquet(partition_path, index=False)

        # Use joblib.Parallel for parallel processing
        if self.partition:
            partition_values = data[self.partition].unique()
            Parallel(n_jobs=njobs, backend="threading")(
                delayed(process_partition)(partition_value)
                for partition_value in partition_values
            )
        else:
            # No partitioning, save data to a single file
            partition_path = self.path / f"{self.name}.parquet"
            if partition_path.exists():
                existing_data = pd.read_parquet(partition_path)
                combined_data = pd.concat([existing_data, data])
                combined_data = (
                    combined_data.drop_duplicates(subset=self.index, keep="last")
                    if self.index
                    else combined_data
                )
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
                raise ValueError(
                    "Both `index_col` and `columns` must be specified when `pivot` is used."
                )
            df = df.pivot(
                index=index,
                columns=columns,
                values=pivot[0] if len(pivot) == 1 else pivot,
            )
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
            if "=" not in min_file or "=" not in max_file:
                partition_range = min_file + " - " + max_file
            else:
                partition_range = f"{min_file.split('=')[1]} - {max_file.split('=')[1]}"

        # Display schema info from one file if available
        first_file = self.partitions[np.argmin(file_size)] if file_count else None
        column_info = "N/A"
        if first_file:
            try:
                df = pd.read_parquet(first_file)
                column_info = ", ".join(
                    [f"{col} ({df[col].dtype})" for col in df.columns]
                )
            except Exception as e:
                column_info = f"Error retrieving columns: {e}"

        return (
            f"Parquet Database '{self.name}' at '{self.path}':\n"
            f"File Count: {file_count}\n"
            f"Partition Column: {self.partition}\n"
            f"Partition Range: {partition_range}\n"
            f"Columns: {column_info}\n"
            f"Total Size: {file_size_str}"
        )

    def __repr__(self):
        return self.__str__()


class DuckDBManager:

    def __init__(self, path: str):
        self.path = path

    @contextmanager
    def _connection(self):
        conn = duckdb.connect(self.path)
        try:
            yield conn
        finally:
            conn.close()

    def _validate_identifier(self, name: str) -> None:
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
            raise ValueError(f"Invalid SQL identifier: {name}")

    def _map_dtype(self, dtype_str: str) -> str:
        type_map = {
            "int64": "BIGINT",
            "float64": "DOUBLE",
            "datetime64[ns]": "TIMESTAMP",
            "category": "VARCHAR",
            "object": "VARCHAR",
            "bool": "BOOLEAN",
        }
        return type_map.get(dtype_str, "VARCHAR")

    def _build_where_clause(self, filters: dict) -> tuple:
        where_clauses = []
        params = []
        valid_operators = {
            "gt": ">",
            "lt": "<",
            "ge": ">=",
            "le": "<=",
            "eq": "=",
            "ne": "!=",
            "like": "LIKE",
            "in": "IN",
            "not_in": "NOT IN",
        }

        for key, value in filters.items():
            col_part = key.split("__", 1)
            col, operation = col_part if len(col_part) > 1 else (key, "eq")

            self._validate_identifier(col)
            op = valid_operators.get(operation, operation)

            if operation in ("in", "not_in"):
                placeholders = ", ".join(["?"] * len(value))
                where_clauses.append(f"{col} {op} ({placeholders})")
                params.extend(value)
            else:
                where_clauses.append(f"{col} {op} ?")
                params.append(value)

        return " AND ".join(where_clauses), params

    def _table_exists(self, conn: duckdb.DuckDBPyConnection, table: str) -> bool:
        return conn.execute(
            "SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = ?)",
            [table],
        ).fetchone()[0]

    def _create_table(
        self, conn: duckdb.DuckDBPyConnection, df: pd.DataFrame, table: str
    ):
        index_cols = [name for name in df.index.names if name]
        if index_cols:
            df_create = df.reset_index()
        else:
            df_create = df.copy()

        columns = []
        for col in df_create.columns:
            self._validate_identifier(col)
            dtype = self._map_dtype(str(df_create[col].dtype))
            columns.append(f"{col} {dtype}")

        unique_clause = ""
        if index_cols:
            missing_columns = [
                col for col in index_cols if col not in df_create.columns
            ]
            if missing_columns:
                raise ValueError(f"Missing index columns: {missing_columns}")
            unique_cols = ", ".join(map(str, index_cols))
            unique_clause = f", UNIQUE({unique_cols})"

        create_sql = f"CREATE TABLE {table} ({', '.join(columns)} {unique_clause})"
        conn.execute(create_sql)

        if index_cols:
            cols_hash = hash(tuple(sorted(index_cols))) & 0xFFFFFFFF
            index_name = f"idx_{table.lower()}_{cols_hash}"
            index_columns = ", ".join(index_cols)
            conn.execute(f"CREATE INDEX {index_name} ON {table} ({index_columns})")

        conn.commit()

    def read(
        self,
        table: str,
        index: Union[str, List[str], None] = None,
        columns: Optional[List[str]] = None,
        pivot: Optional[str] = None,
        **filters,
    ) -> pd.DataFrame:
        with self._connection() as conn:
            required_columns = set()
            if index:
                index_cols = [index] if isinstance(index, str) else index
                required_columns.update(index_cols)

            if columns:
                required_columns.update(columns)

            if pivot:
                required_columns.add(pivot)

            select_clause = (
                ", ".join(sorted(required_columns)) if required_columns else "*"
            )

            where_clause, params = self._build_where_clause(filters)
            sql_query = f"SELECT {select_clause} FROM {table}"
            if where_clause:
                sql_query += f" WHERE {where_clause}"

            df = conn.execute(sql_query, params).fetchdf()

            if pivot:
                if not index or not columns:
                    raise ValueError("Pivot requires both index and columns")
                return df.pivot(index=index, columns=columns, values=pivot)

            if index and not df.empty:
                missing_index = set(
                    index if isinstance(index, list) else [index]
                ) - set(df.columns)
                if missing_index:
                    raise ValueError(f"Missing index columns: {missing_index}")
                df = df.set_index(index)

            return df

    def upsert(self, df: pd.DataFrame, table: str) -> None:
        with self._connection() as conn:
            if not self._table_exists(conn, table):
                self._create_table(conn, df, table)

            temp_view = f"temp_{table}_{uuid.uuid4().hex[:8]}"
            conn.register(temp_view, df.reset_index() if df.index.names[0] else df)

            try:
                index_cols = [name for name in df.index.names if name]
                source_cols = (
                    conn.execute(f"PRAGMA table_info({table})").df()["name"].tolist()
                )

                conflict_clause = (
                    (
                        f"ON CONFLICT ({', '.join(index_cols)}) DO UPDATE SET "
                        + ", ".join(
                            [
                                f"{col}=EXCLUDED.{col}"
                                for col in source_cols
                                if col not in index_cols
                            ]
                        )
                    )
                    if index_cols
                    else ""
                )

                insert_sql = f"""
                    INSERT INTO {table} 
                    SELECT * FROM {temp_view}
                    {conflict_clause}
                """
                conn.execute(insert_sql)
                conn.commit()
            finally:
                conn.unregister(temp_view)

    def add_column(self, table: str, column: str, dtype: str) -> None:
        self._validate_identifier(table)
        self._validate_identifier(column)
        with self._connection() as conn:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {dtype}")
            conn.commit()

    def drop_column(self, table: str, column: str) -> None:
        self._validate_identifier(table)
        self._validate_identifier(column)
        with self._connection() as conn:
            conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
            conn.commit()

    def rename_column(self, table: str, old_name: str, new_name: str) -> None:
        self._validate_identifier(table)
        self._validate_identifier(old_name)
        self._validate_identifier(new_name)
        with self._connection() as conn:
            conn.execute(f"ALTER TABLE {table} RENAME COLUMN {old_name} TO {new_name}")
            conn.commit()

    def change_column_type(self, table: str, column: str, new_type: str) -> None:
        self._validate_identifier(table)
        self._validate_identifier(column)
        with self._connection() as conn:
            conn.execute(f"ALTER TABLE {table} ALTER COLUMN {column} TYPE {new_type}")
            conn.commit()

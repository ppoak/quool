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

    def __init__(
        self,
        path: str | Path,
        grouper: str | list | pd.Grouper = None,
        namer: str = None,
        unikey: str | list = None,
    ):
        self._path = Path(path).expanduser().resolve()
        self._path.mkdir(parents=True, exist_ok=True)
        self._unikey = [unikey] if isinstance(unikey, str) else unikey
        self._grouper = grouper
        self._namer = namer or "{name}__{partition_name}.parquet"

    @property
    def name(self):
        return self._path.name

    @property
    def path(self):
        return self._path

    @property
    def unikey(self):
        return self._unikey

    @property
    def grouper(self):
        return self._grouper

    @property
    def namer(self):
        return self._namer

    @property
    def columns(self):
        if min_partiton := self.min_partition:
            return pd.read_parquet(min_partiton).columns
        return pd.Index([])

    @property
    def partitions(self):
        return sorted(list(self._path.iterdir()))

    @property
    def min_partition(self):
        if self.partitions:
            return self.partitions[
                np.argmin([f.stat().st_size for f in self.partitions])
            ]

    def _generate_filters(self, kwargs: dict):
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

    def _get_partition_path(self, partition_name: str):
        return self.path / self.namer.format(
            name=re.sub(r'[\\/:*?"<>|]', "_", str(self.name)),
            partition_name=re.sub(r'[\\/:*?"<>|]', "_", str(partition_name)),
        )

    def drop_col(self, columns: list | str, n_jobs: int = 4):
        columns = [columns] if isinstance(columns, str) else columns
        if pd.Index(columns).isin(self.unikey).any():
            raise ValueError("Cannot drop unique key columns.")

        def process_partition(partition):
            df = pd.read_parquet(partition)
            df.drop(columns=columns, inplace=True)
            if self.unikey:
                df.sort_values(by=self.unikey, inplace=True)
            df.to_parquet(partition)

        Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(process_partition)(partition) for partition in self.partitions
        )

    def drop_row(self, n_jobs: int = 4, **kwargs):
        def process_partition(partition):
            if not pd.read_parquet(
                partition, filters=filters, columns=self.unikey
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
                if self.unikey:
                    df[~conditions].sort_values(by=self.unikey, inplace=True)
                df.to_parquet(partition, index=False)

        filters = self._generate_filters(kwargs)
        Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(process_partition)(partition) for partition in self.partitions
        )

    def rename_col(self, n_jobs: int = 4, **kwargs):
        def process_partition(partition):
            df = pd.read_parquet(partition)
            df.rename(columns=kwargs, inplace=True)
            if self.unikey:
                df.sort_values(by=self.unikey, inplace=True)
            df.to_parquet(partition)

        Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(process_partition)(partition) for partition in self.partitions
        )

    def add_col(self, n_jobs: int = 4, **dtypes):
        def process_partition(partition):
            df = pd.read_parquet(partition)
            df[dtypes.keys()] = pd.DataFrame(columns=dtypes.keys()).astype(dtypes.values())
            if self.unikey:
                df.sort_values(by=self.unikey, inplace=True)
            df.to_parquet(partition)

        Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(process_partition)(partition) for partition in self.partitions
        )

    def update(
        self,
        data: pd.DataFrame,
        n_jobs: int = 4,
    ):
        if self.columns.size and not (self.columns == data.columns).all():
            raise ValueError("Malformed data, please check your input")

        if self.unikey:
            data = data.drop_duplicates(subset=self.unikey, keep="last")

        # Helper function for processing a single partition
        def process_partition(name, value):
            partition_path = self._get_partition_path(name)

            if partition_path.exists():
                # Load existing data and combine
                existing_data = pd.read_parquet(partition_path)
                combined_data = pd.concat(
                    [existing_data, value], ignore_index=True, axis=0
                )
                if self.unikey:
                    combined_data = combined_data.drop_duplicates(
                        subset=self.unikey, keep="last"
                    )
            else:
                combined_data = value

            # Save combined data
            if self.unikey:
                combined_data.sort_values(by=self.unikey, inplace=True)
            combined_data.to_parquet(partition_path, index=False)

        # Use joblib.Parallel for parallel processing
        if self.grouper is not None:
            Parallel(n_jobs=n_jobs, backend="threading")(
                delayed(process_partition)(name, value)
                for name, value in data.groupby(self.grouper)
            )
        else:
            # No partitioning, save data to a single file
            partition_path = self.path / f"{self.name}.parquet"
            if partition_path.exists():
                existing_data = pd.read_parquet(partition_path)
                combined_data = pd.concat([existing_data, data])
                combined_data = (
                    combined_data.drop_duplicates(subset=self.unikey, keep="last")
                    if self.unikey
                    else combined_data
                )
            else:
                combined_data = data
            if self.unikey:
                combined_data.sort_values(by=self.unikey, inplace=True)
            combined_data.to_parquet(partition_path, index=False)

    def read(
        self,
        index: list | str = None,
        columns: list | str = None,
        pivot: list | str = None,
        **filters,
    ):
        # Ensure `index_col` and `columns` are lists for consistency
        index = index if index is not None else []
        columns = columns if columns is not None else []
        pivot = pivot if pivot is not None else []
        index = [index] if isinstance(index, str) else index
        columns = [columns] if isinstance(columns, str) else columns
        pivot = [pivot] if isinstance(pivot, str) else pivot
        read_columns = index + columns
        read_columns = read_columns if not pivot else read_columns + pivot
        read_columns = None if not read_columns else read_columns

        # Generate filters
        filters = self._generate_filters(filters)
        # Read data with filters
        df = pd.read_parquet(self._path, columns=read_columns, filters=filters)

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
        file_size = [f.stat().st_size for f in self.partitions]
        file_size_total = sum(file_size)
        if file_size_total > 1024**3:
            file_size_str = f"{file_size_total / 1024**3:.2f} GB"
        elif file_size_total > 1024**2:
            file_size_str = f"{file_size_total / 1024**2:.2f} MB"
        elif file_size_total > 1024:
            file_size_str = f"{file_size_total / 1024:.2f} KB"
        else:
            file_size_str = f"{file_size_total} B"

        partition_range = "N/A"
        if self.partitions:
            min_file = self.partitions[0].stem
            max_file = self.partitions[-1].stem
            partition_range = f"{min_file} - {max_file}"

        return (
            f"Parquet Database '{self.name}' at '{self._path}':\n"
            f"File Count: {file_count}\n"
            f"Partition Range: {partition_range}\n"
            f"Columns: {self.columns.to_list()}\n"
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
            source_cols = (
                conn.execute(f"PRAGMA table_info({table})").df()["name"].tolist()
            )

            temp_view = f"temp_{table}_{uuid.uuid4().hex[:8]}"
            conn.register(
                temp_view,
                df.reset_index()[source_cols] if df.index.names[0] else df[source_cols],
            )

            try:
                index_cols = [name for name in df.index.names if name]
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

import re
import uuid
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from contextlib import contextmanager
from typing import Union, List, Optional, Tuple, Dict, Any


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
                df = df.sort_values(by=self.unikey)
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
                    df = df[~conditions].sort_values(by=self.unikey)
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
                df = df.sort_values(by=self.unikey)
            df.to_parquet(partition)

        Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(process_partition)(partition) for partition in self.partitions
        )

    def add_col(self, n_jobs: int = 4, **dtypes):
        def process_partition(partition):
            df = pd.read_parquet(partition)
            df[dtypes.keys()] = pd.DataFrame(columns=dtypes.keys()).astype(
                dtypes.values()
            )
            if self.unikey:
                df = df.sort_values(by=self.unikey)
            df.to_parquet(partition)

        Parallel(n_jobs=n_jobs, backend="threading")(
            delayed(process_partition)(partition) for partition in self.partitions
        )

    def update(
        self,
        data: pd.DataFrame,
        n_jobs: int = 4,
    ):
        if self.columns.size and not (data.columns.isin(self.columns)).all():
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
                combined_data = combined_data.sort_values(by=self.unikey)
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
                combined_data = combined_data.sort_values(by=self.unikey)
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

    def __init__(self, path: str, read_only: bool = False):
        self.path = path
        self.read_only = read_only

    @contextmanager
    def connect(self):
        con = duckdb.connect(self.path, read_only=self.read_only)
        try:
            yield con
        finally:
            con.close()

    def infer_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        schema = {}
        type_map = {
            "int64": "BIGINT",
            "int32": "INTEGER",
            "int16": "SMALLINT",
            "int8": "TINYINT",
            "float64": "DOUBLE",
            "float32": "REAL",
            "bool": "BOOLEAN",
            "category": "VARCHAR",
            "object": "VARCHAR",
            "datetime64[us]": "TIMESTAMP",
            "datetime64[ns]": "TIMESTAMP",
            "timedelta64[us]": "INTERVAL",
            "timedelta64[ns]": "INTERVAL",
            "string": "VARCHAR",
        }
        index_cols = list(df.index.names) if all(df.index.names) else []
        schema["pk"] = index_cols
        schema["index"] = index_cols

        df = df.reset_index() if index_cols else df
        schema["columns"] = []
        for col in df.columns:
            schema["columns"].append(
                f'{col} {type_map.get(str(df[col].dtype), "VARCHAR")}'
            )
        return schema

    def create_table(
        self,
        table: str,
        columns: List[str],
        pk: str | List[str] = None,
        unique: str | List[str] = None,
        index: str | List[str] = None,
    ) -> None:
        columns = ", ".join(columns)
        pk_clause = f", PRIMARY KEY({', '.join(pk)})" if pk else ""
        unique_clause = f", UNIQUE({', '.join(unique)})" if unique else ""

        sql = f"CREATE TABLE {table} ({columns}{pk_clause}{unique_clause});"

        with self.connect() as con:
            con.execute(sql)

            if index:
                for col in index:
                    index_name = f"idx_{table}_{col}"
                    con.execute(f"CREATE INDEX {index_name} ON {table} ({col});")
            con.commit()

    def upsert(self, df: pd.DataFrame, table: str) -> None:
        with self.connect() as con:
            tables = (
                con.execute("SELECT name FROM (SHOW ALL TABLES);")
                .fetchdf()
                .iloc[:, 0]
                .to_list()
            )
            if table not in tables:
                schema = self.infer_schema(df)
                self.create_table(table, **schema)

            if all(df.index.names):
                df = df.reset_index()

            temp_view = f"temp_view_{uuid.uuid4().hex[:8]}"
            con.register(temp_view, df)

            sql = f"""
                INSERT OR REPLACE INTO {table}
                SELECT * FROM {temp_view};
            """
            con.execute(sql)
            con.unregister(temp_view)

    def select(
        self,
        table: str,
        columns: Optional[List[str]] = None,
        ands: Optional[List[str]] = None,
        ors: Optional[List[str]] = None,
        groupby: Optional[List[str]] = None,
        having: Optional[List[str]] = None,
        orderby: Optional[Union[str, List[Union[str, Tuple[str, str]]]]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        distinct: bool = False,
        params: Tuple = None,
    ) -> pd.DataFrame:

        # SELECT
        select_clause = ", ".join(columns) if columns else "*"
        if distinct:
            select_clause = f"DISTINCT {select_clause}"

        sql = f"SELECT {select_clause} FROM {table}"

        # WHERE
        if ands:
            sql += f" WHERE {' AND '.join(ands)}"

        if ors:
            sql += f" WHERE {' OR '.join(ors)}"

        # GROUP BY
        if groupby:
            sql += f" GROUP BY {', '.join(groupby)}"

        # HAVING
        if having:
            sql += f" HAVING {', '.join(having)}"

        # ORDER BY
        if orderby:
            if isinstance(orderby, str):
                sql += f" ORDER BY {orderby}"
            elif isinstance(orderby, list):
                sql += f" ORDER BY {', '.join(orderby)}"

        if limit is not None:
            sql += f" LIMIT {limit}"
        if offset is not None:
            sql += f" OFFSET {offset}"

        with self.connect() as con:
            return con.execute(sql, params).fetchdf()

    def pivot(
        self,
        table: str,
        index: str,
        columns: str,
        values: str,
        aggfunc: str = "SUM",
        categories: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:

        where_clause, params = "", []
        if filters:
            where_clause, params = self._build_where_clause(filters)
            where_clause = f"WHERE {where_clause}"

        if categories is None:
            # Auto-fetch distinct pivot values from the column
            with self.connect() as con:
                q = f"SELECT DISTINCT {columns} FROM {table} {where_clause}"
                categories = [row[0] for row in con.execute(q, params).fetchall()]
            if not categories:
                raise ValueError(f"No categories found in column '{columns}'.")

        in_clause = ", ".join(f"'{c}'" for c in categories)

        sql = f"""
        SELECT * FROM (
            SELECT {index}, {columns}, {values}
            FROM {table}
            {where_clause}
        )
        PIVOT (
            {aggfunc}({values}) FOR {columns} IN ({in_clause})
        )
        """

        with self.connect() as con:
            return con.execute(sql, params).fetchdf()

    def delete(self, table: str, **filters) -> int:
        with self.connect() as conn:
            where_clause, params = self._build_where_clause(filters)
            sql = f"DELETE FROM {table} WHERE {where_clause}"
            result = conn.execute(sql, params)
            conn.commit()
            return result.rowcount

    def execute(self, sql: str, params: Optional[List] = None) -> pd.DataFrame:
        with self.connect() as conn:
            return conn.execute(sql, params).fetchall()

    def drop_table(self, table: str) -> None:
        with self.connect() as conn:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.commit()

    def add_col(self, table: str, column: str, dtype: str) -> None:
        with self.connect() as conn:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {dtype}")
            conn.commit()

    def drop_col(self, table: str, column: str) -> None:
        with self.connect() as conn:
            conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
            conn.commit()

    def rename_col(self, table: str, old_name: str, new_name: str) -> None:
        with self.connect() as conn:
            conn.execute(f"ALTER TABLE {table} RENAME COLUMN {old_name} TO {new_name}")
            conn.commit()

    def change_col_type(self, table: str, column: str, new_type: str) -> None:
        with self.connect() as conn:
            conn.execute(f"ALTER TABLE {table} ALTER COLUMN {column} TYPE {new_type}")
            conn.commit()

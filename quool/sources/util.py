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

    def _validate_identifier(self, name: str) -> None:
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", name):
            raise ValueError(f"Invalid SQL identifier: {name}")

    def _infer_dtype(self, dtype_str: str) -> str:
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
        return type_map[dtype_str]

    def _infer_schema(self, df: pd.DataFrame) -> Dict[str, str]:
        schema = {}
        for col in df.columns:
            self._validate_identifier(col)
            dtype = self._infer_dtype(str(df[col].dtype))
            schema[col] = dtype
        return schema

    def _build_where_clause(self, filters: Dict[str, Any]) -> Tuple[str, List[Any]]:
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
            "notin": "NOT IN",
        }

        for key, value in filters.items():
            col_part = key.split("__", 1)
            col, operation = col_part if len(col_part) > 1 else (key, "eq")

            self._validate_identifier(col)
            op = valid_operators.get(operation, operation)

            if operation in ("in", "notin"):
                placeholders = ", ".join(["?"] * len(value))
                where_clauses.append(f"{col} {op} ({placeholders})")
                params.extend(value)
            else:
                where_clauses.append(f"{col} {op} ?")
                params.append(value)

        return " AND ".join(where_clauses), params

    @contextmanager
    def connect(self):
        con = duckdb.connect(self.path, read_only=self.read_only)
        try:
            yield con
        finally:
            con.close()

    def create_table(
        self, con: duckdb.DuckDBPyConnection, schema: Dict[str, str], table: str
    ):
        index_cols = [name for name in df.index.names if name]
        if index_cols:
            df = df.reset_index()
        schema = ", ".join([f"{name} {dtype}" for name, dtype in schema.items()])

        unique_clause = ""
        if index_cols:
            missing_columns = [col for col in index_cols if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing index columns: {missing_columns}")
            unique_cols = ", ".join(map(str, index_cols))
            unique_clause = f", UNIQUE({unique_cols})"

        create_sql = f"CREATE TABLE IF NOT EXISTS {table} ({schema}) {unique_clause})"
        con.execute(create_sql)

        if index_cols:
            cols_hash = hash(tuple(sorted(index_cols))) & 0xFFFFFFFF
            index_name = f"idx_{table.lower()}_{cols_hash}"
            index_columns = ", ".join(index_cols)
            con.execute(f"CREATE INDEX {index_name} ON {table} ({index_columns})")

        con.commit()

    def upsert(self, df: pd.DataFrame, table: str) -> None:
        with self.connection() as conn:
            self.create_table(conn, df, table)
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

    def read(
        self,
        table: str,
        columns: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
        groupby: Optional[List[str]] = None,
        having: Optional[Dict[str, Any]] = None,
        orderby: Optional[Union[str, List[Union[str, Tuple[str, str]]]]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        distinct: bool = False,
        pivot: Optional[Dict[str, str]] = None,
    ) -> pd.DataFrame:
        self._validate_identifier(table)

        select_clause = ", ".join(columns) if columns else "*"
        if distinct:
            select_clause = f"DISTINCT {select_clause}"

        sql = f"SELECT {select_clause} FROM {table}"
        params = []

        # WHERE
        if filters:
            where_clause, where_params = self._build_where_clause(filters)
            sql += f" WHERE {where_clause}"
            params.extend(where_params)

        # GROUP BY
        if groupby:
            for col in groupby:
                self._validate_identifier(col)
            sql += f" GROUP BY {', '.join(groupby)}"

        # HAVING
        if having:
            having_clause, having_params = self._build_where_clause(having)
            sql += f" HAVING {having_clause}"
            params.extend(having_params)

        # ORDER BY
        if orderby:
            if isinstance(orderby, str):
                sql += f" ORDER BY {orderby}"
            elif isinstance(orderby, list):
                order_clause = []
                for item in orderby:
                    if isinstance(item, tuple):
                        col, direction = item
                        self._validate_identifier(col)
                        order_clause.append(f"{col} {direction}")
                    else:
                        self._validate_identifier(item)
                        order_clause.append(f"{item}")
                sql += f" ORDER BY {', '.join(order_clause)}"

        # LIMIT & OFFSET
        if limit is not None:
            sql += f" LIMIT {limit}"
        if offset is not None:
            sql += f" OFFSET {offset}"

        # EXECUTE
        with self.connect() as con:
            df = con.execute(sql, params).fetchdf()

        return df

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
        self._validate_identifier(table)
        self._validate_identifier(index)
        self._validate_identifier(columns)
        self._validate_identifier(values)

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
        self._validate_identifier(table)
        self._validate_identifier(column)
        with self.connect() as conn:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {dtype}")
            conn.commit()

    def drop_col(self, table: str, column: str) -> None:
        self._validate_identifier(table)
        self._validate_identifier(column)
        with self.connect() as conn:
            conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
            conn.commit()

    def rename_col(self, table: str, old_name: str, new_name: str) -> None:
        self._validate_identifier(table)
        self._validate_identifier(old_name)
        self._validate_identifier(new_name)
        with self.connect() as conn:
            conn.execute(f"ALTER TABLE {table} RENAME COLUMN {old_name} TO {new_name}")
            conn.commit()

    def change_col_type(self, table: str, column: str, new_type: str) -> None:
        self._validate_identifier(table)
        self._validate_identifier(column)
        with self.connect() as conn:
            conn.execute(f"ALTER TABLE {table} ALTER COLUMN {column} TYPE {new_type}")
            conn.commit()

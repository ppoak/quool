import os
import re
import uuid
import shutil
import duckdb
import numpy as np
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from contextlib import contextmanager
from typing import Union, List, Optional, Tuple, Dict, Any, Sequence


class ParquetManager:

    def __init__(
        self,
        path: str | Path,
        grouper: str | list | pd.Grouper = None,
        namer: str = None,
        unikey: str | list = None,
    ):
        print(
            "Warning: ParquetManager will soon be depreciated in version 7.1.x, please use DuckParquet instead."
        )
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


class DuckParquet:
    def __init__(
        self,
        dataset_path: str,
        name: Optional[str] = None,
        db_path: str = None,
        threads: Optional[int] = None,
    ):
        """Initializes the DuckParquet object.

        Args:
            dataset_path (str): Directory path that stores the parquet dataset.
            name (Optional[str]): The view name. Defaults to directory basename.
            db_path (str): Path to DuckDB database file. Defaults to in-memory.
            threads (Optional[int]): Number of threads used for partition operations.

        Raises:
            ValueError: If the dataset_path is not a directory.
        """
        self.dataset_path = os.path.abspath(dataset_path)
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        if not os.path.isdir(self.dataset_path):
            raise ValueError("Only directory is valid in dataset_path param")
        self.view_name = name or self._default_view_name(self.dataset_path)
        config = {}
        self.threads = threads or 1
        config["threads"] = self.threads
        self.con = duckdb.connect(database=db_path or ":memory:", config=config)
        self.scan_pattern = self._infer_scan_pattern(self.dataset_path)
        if self._parquet_files_exist():
            self._create_or_replace_view()

    # --- Private Helper Methods ---

    @staticmethod
    def _is_identifier(name: str) -> bool:
        """Check if a string is a valid DuckDB SQL identifier.

        Args:
            name (str): The identifier to check.

        Returns:
            bool: True if valid identifier, else False.
        """
        return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name))

    @staticmethod
    def _quote_ident(name: str) -> str:
        """Quote a string if it's not a valid identifier for DuckDB.

        Args:
            name (str): The identifier to quote.

        Returns:
            str: Quoted identifier as DuckDB requires.
        """
        if DuckParquet._is_identifier(name):
            return name
        return '"' + name.replace('"', '""') + '"'

    @staticmethod
    def _default_view_name(path: str) -> str:
        """Generate a default DuckDB view name from file/directory name.

        Args:
            path (str): Directory or parquet file path.

        Returns:
            str: Default view name.
        """
        base = os.path.basename(path.rstrip(os.sep))
        name = os.path.splitext(base)[0] if base.endswith(".parquet") else base
        if not DuckParquet._is_identifier(name):
            name = "ds_" + re.sub(r"[^A-Za-z0-9_]+", "_", name)
        return name or "dataset"

    @staticmethod
    def _infer_scan_pattern(path: str) -> str:
        """Infer DuckDB's parquet_scan path glob based on the directory path.

        Args:
            path (str): Target directory.

        Returns:
            str: Glob scan pattern.
        """
        if os.path.isdir(path):
            return os.path.join(path, "**/*.parquet")
        return path

    @staticmethod
    def _local_tempdir(target_dir, prefix="__parquet_rewrite_"):
        """Generate a temporary directory for atomic operations under target_dir.

        Args:
            target_dir (str): Directory for temp.

        Returns:
            str: Path to temp directory.
        """
        tmpdir = os.path.join(target_dir, f"{prefix}{uuid.uuid4().hex[:8]}")
        os.makedirs(tmpdir)
        return tmpdir

    def _parquet_files_exist(self) -> bool:
        """Check if there are any parquet files under the dataset path.

        Returns:
            bool: True if any parquet exists, else False.
        """
        for root, dirs, files in os.walk(self.dataset_path):
            for fn in files:
                if fn.endswith(".parquet"):
                    return True
        return False

    def _create_or_replace_view(self):
        """Create or replace the DuckDB view for current dataset."""
        view_ident = DuckParquet._quote_ident(self.view_name)
        sql = f"CREATE OR REPLACE VIEW {view_ident} AS SELECT * FROM parquet_scan('{self.scan_pattern}')"
        self.con.execute(sql)

    def _base_columns(self) -> List[str]:
        """Get all base columns from current parquet duckdb view.

        Returns:
            List[str]: List of column names in the schema.
        """
        return self.list_columns()

    def _copy_select_to_dir(
        self,
        select_sql: str,
        target_dir: str,
        partition_by: Optional[List[str]] = None,
        params: Optional[Sequence[Any]] = None,
        compression: str = "zstd",
    ):
        """Dump SELECT query result to parquet files under target_dir.

        Args:
            select_sql (str): SELECT SQL to copy data from.
            target_dir (str): Target directory to store parquet files.
            partition_by (Optional[List[str]]): Partition columns.
            params (Optional[Sequence[Any]]): SQL bind parameters.
            compression (str): Parquet compression, default 'zstd'.
        """
        opts = [f"FORMAT 'parquet'"]
        if compression:
            opts.append(f"COMPRESSION '{compression}'")
        if partition_by:
            cols = ", ".join(DuckParquet._quote_ident(c) for c in partition_by)
            opts.append(f"PARTITION_BY ({cols})")
        options_sql = ", ".join(opts)
        sql = f"COPY ({select_sql}) TO '{target_dir}' ({options_sql})"
        self.con.execute(sql, params)

    def _copy_df_to_dir(
        self,
        df: pd.DataFrame,
        target: str,
        partition_by: Optional[List[str]] = None,
        compression: str = "zstd",
    ):
        """Write pandas DataFrame into partitioned parquet files.

        Args:
            df (pd.DataFrame): Source dataframe.
            target (str): Target directory.
            partition_by (Optional[List[str]]): Partition columns.
            compression (str): Parquet compression.
        """
        reg_name = f"incoming_{uuid.uuid4().hex[:8]}"
        self.con.register(reg_name, df)
        opts = [f"FORMAT 'parquet'"]
        if compression:
            opts.append(f"COMPRESSION '{compression}'")
        if partition_by:
            cols = ", ".join(DuckParquet._quote_ident(c) for c in partition_by)
            opts.append(f"PARTITION_BY ({cols})")
        options_sql = ", ".join(opts)
        if partition_by:
            sql = f"COPY (SELECT * FROM {DuckParquet._quote_ident(reg_name)}) TO '{target}' ({options_sql})"
        else:
            sql = f"COPY (SELECT * FROM {DuckParquet._quote_ident(reg_name)}) TO '{target}/data_0.parquet' ({options_sql})"
        self.con.execute(sql)
        self.con.unregister(reg_name)

    def _atomic_replace_dir(self, new_dir: str, old_dir: str):
        """Atomically replace a directory's contents.

        Args:
            new_dir (str): Temporary directory with new data.
            old_dir (str): Target directory to replace.
        """
        if os.path.exists(old_dir):
            shutil.rmtree(old_dir)
        os.replace(new_dir, old_dir)

    # ---- Upsert Internal Logic ----

    def _upsert_no_exist(self, df: pd.DataFrame, partition_by: Optional[list]) -> None:
        """Upsert logic branch if no existing parquet files.

        Args:
            df (pd.DataFrame): Raw DataFrame
            partition_by (Optional[list]): Partition columns
        """
        tmpdir = self._local_tempdir(".")
        try:
            self._copy_df_to_dir(
                df,
                target=tmpdir,
                partition_by=partition_by,
            )
            self._atomic_replace_dir(tmpdir, self.dataset_path)
        finally:
            if os.path.exists(tmpdir):
                shutil.rmtree(tmpdir, ignore_errors=True)
        self.refresh()

    def _export_partition(
        self,
        part_row,
        partition_by,
        all_cols,
        key_expr,
        view_ident,
        df,
        tmpdir,
        sql_template,
    ):
        """Export partition sub-data for upsert, called in parallel for each partition.

        Args:
            part_row (pd.Series): Partition key row.
            partition_by (list): Partition columns.
            all_cols (str): Columns to select.
            key_expr (str): Key expression.
            view_ident (str): View identifier.
            df (pd.DataFrame): DataFrame to upsert.
            tmpdir (str): Temporary directory.
            sql_template (str): SQL COPY template.
        """
        temp_name = f"newdata_{uuid.uuid4().hex[:6]}"
        where_clauses = [
            f"{DuckParquet._quote_ident(col)} = '{part_row[col]}'"
            for col in partition_by
        ]
        where_sql = " AND ".join(where_clauses)
        sql = sql_template.format(
            all_cols=all_cols,
            key_expr=key_expr,
            view_ident=view_ident,
            tmpdir=tmpdir,
            temp_name=DuckParquet._quote_ident(temp_name),
            partition_subsql=f"PARTITION_BY ({', '.join(partition_by or [])}),",
        )
        sub_df = df.loc[(df[partition_by] == part_row[partition_by]).all(axis=1)]
        con = duckdb.connect()
        con.register(view_ident, self.select(where=where_sql))
        con.register(temp_name, sub_df)
        con.execute(sql)
        con.unregister(view_ident)
        con.unregister(temp_name)
        con.close()

    def _upsert_existing(
        self, df: pd.DataFrame, keys: list, partition_by: Optional[list]
    ) -> None:
        """Upsert logic branch if existing parquet files already present.

        Args:
            df (pd.DataFrame): Raw DataFrame
            keys (list): Primary key columns
            partition_by (Optional[list]): Partition columns
        """
        tmpdir = self._local_tempdir(".")
        base_cols = self.list_columns()
        view_ident = DuckParquet._quote_ident(self.view_name)
        all_cols = ", ".join(
            [
                DuckParquet._quote_ident(c)
                for c in base_cols
                if c in df.columns or c in base_cols
            ]
        )
        key_expr = ", ".join(keys)
        sql_template = """
            COPY (
                SELECT {all_cols} FROM (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY {key_expr} ORDER BY is_new DESC) AS rn
                    FROM (
                        SELECT {all_cols}, 0 as is_new FROM {view_ident}
                        UNION ALL
                        SELECT {all_cols}, 1 as is_new FROM {temp_name}
                    )
                ) WHERE rn=1
            ) TO '{tmpdir}' (FORMAT 'parquet', {partition_subsql} OVERWRITE_OR_IGNORE true)
        """
        if not partition_by:
            try:
                temp_name = f"newdata_{uuid.uuid4().hex[:6]}"
                sql = sql_template.format(
                    all_cols=all_cols,
                    key_expr=key_expr,
                    view_ident=view_ident,
                    tmpdir=os.path.join(tmpdir, "data_0.parquet"),
                    temp_name=DuckParquet._quote_ident(temp_name),
                    partition_subsql="",
                )
                self.con.register(view_ident, self.select())
                self.con.register(temp_name, df)
                self.con.execute(sql)
                self.con.unregister(temp_name)
                self.con.unregister(view_ident)
                src_part_dir = os.path.join(tmpdir, "data_0.parquet")
                dst_part_dir = os.path.join(self.dataset_path, "data_0.parquet")
                os.remove(dst_part_dir)
                shutil.move(src_part_dir, dst_part_dir)
            finally:
                shutil.rmtree(tmpdir)
            return

        affected_partitions = df[partition_by].drop_duplicates()
        try:
            Parallel(n_jobs=self.threads, backend="threading")(
                delayed(self._export_partition)(
                    part_row,
                    partition_by,
                    all_cols,
                    key_expr,
                    view_ident,
                    df,
                    tmpdir,
                    sql_template,
                )
                for _, part_row in affected_partitions.iterrows()
            )
            subdirs = next(os.walk(tmpdir))[1]
            for subdir in subdirs:
                src_part_dir = os.path.join(tmpdir, subdir)
                dst_part_dir = os.path.join(self.dataset_path, subdir)
                if os.path.exists(dst_part_dir):
                    if os.path.isdir(dst_part_dir):
                        shutil.rmtree(dst_part_dir)
                    else:
                        os.remove(dst_part_dir)
                os.makedirs(os.path.dirname(dst_part_dir), exist_ok=True)
                shutil.move(src_part_dir, dst_part_dir)
        finally:
            if os.path.exists(tmpdir):
                shutil.rmtree(tmpdir, ignore_errors=True)
        self.refresh()

    # --- Context/Resource Management ---
    def close(self):
        """Close the DuckDB connection."""
        try:
            self.con.close()
        except Exception:
            pass

    def __enter__(self):
        """Enable usage as a context manager.

        Returns:
            DuckParquet: Current instance.
        """
        return self

    def __exit__(self, exc_type, exc, tb):
        """Context manager exit: close connection."""
        self.close()

    # --- Public Query/Mutation Methods ---

    def refresh(self):
        """Refreshes DuckDB view after manual file changes."""
        self._create_or_replace_view()

    def raw_query(
        self, sql: str, params: Optional[Sequence[Any]] = None
    ) -> pd.DataFrame:
        """Execute a raw SQL query and return results as a DataFrame.

        Args:
            sql (str): SQL statement.
            params (Optional[Sequence[Any]]): Bind parameters.

        Returns:
            pd.DataFrame: Query results.
        """
        res = self.con.execute(sql, params or [])
        try:
            return res.df()
        except Exception:
            return res

    def get_schema(self) -> pd.DataFrame:
        """Get the schema (column info) of current parquet dataset.

        Returns:
            pd.DataFrame: DuckDB DESCRIBE result.
        """
        view_ident = DuckParquet._quote_ident(self.view_name)
        return self.con.execute(f"DESCRIBE {view_ident}").df()

    def list_columns(self) -> List[str]:
        """List all columns in the dataset.

        Returns:
            List[str]: Column names in the dataset.
        """
        df = self.get_schema()
        if "column_name" in df.columns:
            return df["column_name"].tolist()
        if "name" in df.columns:
            return df["name"].tolist()
        return df.iloc[:, 0].astype(str).tolist()

    def select(
        self,
        columns: Union[str, List[str]] = "*",
        where: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
        group_by: Optional[Union[str, List[str]]] = None,
        having: Optional[str] = None,
        order_by: Optional[Union[str, List[str]]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        distinct: bool = False,
    ) -> pd.DataFrame:
        """Query current dataset with flexible SQL generated automatically.

        Args:
            columns (Union[str, List[str]]): Columns to select (* or list of str).
            where (Optional[str]): WHERE clause.
            params (Optional[Sequence[Any]]): Bind parameters for WHERE.
            group_by (Optional[Union[str, List[str]]]): GROUP BY columns.
            having (Optional[str]): HAVING clause.
            order_by (Optional[Union[str, List[str]]]): ORDER BY columns.
            limit (Optional[int]): Max rows to get.
            offset (Optional[int]): Row offset.
            distinct (bool): Whether to add DISTINCT clause.

        Returns:
            pd.DataFrame: Query results.
        """
        view_ident = DuckParquet._quote_ident(self.view_name)
        col_sql = columns if isinstance(columns, str) else ", ".join(columns)
        sql = ["SELECT"]
        if distinct:
            sql.append("DISTINCT")
        sql.append(col_sql)
        sql.append(f"FROM {view_ident}")
        bind_params = list(params or [])
        if where:
            sql.append("WHERE")
            sql.append(where)
        if group_by:
            group_sql = group_by if isinstance(group_by, str) else ", ".join(group_by)
            sql.append("GROUP BY " + group_sql)
        if having:
            sql.append("HAVING " + having)
        if order_by:
            order_sql = order_by if isinstance(order_by, str) else ", ".join(order_by)
            sql.append("ORDER BY " + order_sql)
        if limit is not None:
            sql.append(f"LIMIT {int(limit)}")
        if offset is not None:
            sql.append(f"OFFSET {int(offset)}")
        final = " ".join(sql)
        return self.raw_query(final, bind_params)

    def dpivot(
        self,
        index: Union[str, List[str]],
        columns: str,
        values: str,
        aggfunc: str = "first",
        where: Optional[str] = None,
        on_in: Optional[List[Any]] = None,
        group_by: Optional[Union[str, List[str]]] = None,
        order_by: Optional[Union[str, List[str]]] = None,
        limit: Optional[int] = None,
        fill_value: Any = None,
    ) -> pd.DataFrame:
        """
        Pivot the parquet dataset using DuckDB PIVOT statement.
        Args:
            index: Output rows, will appear in SELECT and GROUP BY.
            columns: The column to turn into wide fields (PIVOT ON).
            values: Value column, aggregate target (PIVOT USING aggfunc(values)).
            aggfunc: Aggregate function, default 'first'.
            where: Filter applied in SELECT node.
            on_in: List of column values, restrict wide columns.
            group_by: Group by after pivot, usually same as index.
            order_by: Order by after pivot.
            limit: Row limit.
            fill_value: Fill missing values.
        Returns:
            pd.DataFrame: Wide pivoted DataFrame.
        """
        # Construct SELECT query for PIVOT source
        if isinstance(index, str):
            index_cols = [index]
        else:
            index_cols = list(index)
        select_cols = index_cols + [columns, values]
        sel_sql = f"SELECT {', '.join(DuckParquet._quote_ident(c) for c in select_cols)} FROM {DuckParquet._quote_ident(self.view_name)}"
        if where:
            sel_sql += f" WHERE {where}"

        # PIVOT ON
        pivot_on = DuckParquet._quote_ident(columns)
        # PIVOT ON ... IN (...)
        if on_in:
            in_vals = []
            for v in on_in:
                # 按str或数字
                if isinstance(v, str):
                    in_vals.append(f"'{v}'")
                else:
                    in_vals.append(str(v))
            pivot_on += f" IN ({', '.join(in_vals)})"

        # PIVOT USING
        pivot_using = f"{aggfunc}({DuckParquet._quote_ident(values)})"

        # PIVOT 语句
        sql_lines = [f"PIVOT ({sel_sql})", f"ON {pivot_on}", f"USING {pivot_using}"]

        # GROUP BY
        if group_by:
            if isinstance(group_by, str):
                groupby_expr = DuckParquet._quote_ident(group_by)
            else:
                groupby_expr = ", ".join(DuckParquet._quote_ident(c) for c in group_by)
            sql_lines.append(f"GROUP BY {groupby_expr}")

        # ORDER BY
        if order_by:
            if isinstance(order_by, str):
                order_expr = DuckParquet._quote_ident(order_by)
            else:
                order_expr = ", ".join(DuckParquet._quote_ident(c) for c in order_by)
            sql_lines.append(f"ORDER BY {order_expr}")

        # LIMIT
        if limit:
            sql_lines.append(f"LIMIT {int(limit)}")

        sql = "\n".join(sql_lines)
        df = self.raw_query(sql)
        if fill_value is not None:
            df = df.fillna(fill_value)
        return df

    def ppivot(
        self,
        index: Union[str, List[str]],
        columns: Union[str, List[str]],
        values: Optional[Union[str, List[str]]] = None,
        aggfunc: str = "mean",
        where: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
        order_by: Optional[Union[str, List[str]]] = None,
        limit: Optional[int] = None,
        fill_value: Any = None,
        dropna: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Wide pivot using Pandas pivot_table.

        Args:
            index: Indexes of pivot table.
            columns: Columns to expand.
            values: The value fields to aggregate.
            aggfunc: Pandas/numpy function name or callable.
            where: Optional filter.
            params: SQL bind params.
            order_by: Order output.
            limit: Row limit.
            fill_value: Defaults for missing.
            dropna: Drop missing columns.
            **kwargs: Any pandas.pivot_table compatible args.

        Returns:
            pd.DataFrame: Wide table.
        """
        select_cols = []
        for part in (index, columns, values or []):
            if part is None:
                continue
            if isinstance(part, str):
                select_cols.append(part)
            else:
                select_cols.extend(part)
        select_cols = list(dict.fromkeys(select_cols))
        df = self.select(
            columns=select_cols,
            where=where,
            params=params,
            order_by=order_by,
            limit=limit,
        )
        return pd.pivot_table(
            df,
            index=index,
            columns=columns,
            values=values,
            aggfunc=aggfunc,
            fill_value=fill_value,
            dropna=dropna,
            **kwargs,
        )

    def count(
        self, where: Optional[str] = None, params: Optional[Sequence[Any]] = None
    ) -> int:
        """Count rows in the dataset matching the given WHERE clause.

        Args:
            where (Optional[str]): WHERE condition to filter rows.
            params (Optional[Sequence[Any]]): Bind parameters.

        Returns:
            int: The count of rows.
        """
        view_ident = DuckParquet._quote_ident(self.view_name)
        sql = f"SELECT COUNT(*) AS c FROM {view_ident}"
        bind_params = list(params or [])
        if where:
            sql += " WHERE " + where
        return int(self.con.execute(sql, bind_params).fetchone()[0])

    def upsert_from_df(
        self, df: pd.DataFrame, keys: list, partition_by: Optional[list] = None
    ):
        """Upsert rows from DataFrame according to primary keys, overwrite existing rows.

        Args:
            df (pd.DataFrame): New data.
            keys (list): Primary key columns.
            partition_by (Optional[list]): Partition columns.
        """
        if not self._parquet_files_exist():
            self._upsert_no_exist(df, partition_by)
        else:
            self._upsert_existing(df, keys, partition_by)

    def update(
        self,
        set_map: Dict[str, Union[str, Any]],
        where: Optional[str] = None,
        partition_by: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
    ):
        """Update specified columns for rows matching WHERE.

        Args:
            set_map (Dict[str, Union[str, Any]]): {column: value or SQL expr}.
            where (Optional[str]): WHERE clause.
            partition_by (Optional[str]): Partition column.
            params (Optional[Sequence[Any]]): Bind parameters for WHERE.
        """
        if os.path.isfile(self.dataset_path):
            pass
        view_ident = DuckParquet._quote_ident(self.view_name)
        base_cols = self._base_columns()
        bind_params = list(params or [])
        select_exprs = []
        for col in base_cols:
            col_ident = DuckParquet._quote_ident(col)
            if col in set_map:
                val = set_map[col]
                if where:
                    if isinstance(val, str):
                        expr = f"CASE WHEN ({where}) THEN ({val}) ELSE {col_ident} END AS {col_ident}"
                    else:
                        expr = f"CASE WHEN ({where}) THEN (?) ELSE {col_ident} END AS {col_ident}"
                        bind_params.append(val)
                else:
                    if isinstance(val, str):
                        expr = f"({val}) AS {col_ident}"
                    else:
                        expr = f"(?) AS {col_ident}"
                        bind_params.append(val)
            else:
                expr = f"{col_ident}"
            select_exprs.append(expr)
        select_sql = f"SELECT {', '.join(select_exprs)} FROM {view_ident}"
        tmpdir = self._local_tempdir(".")
        try:
            self._copy_select_to_dir(
                select_sql,
                target_dir=tmpdir,
                partition_by=partition_by,
            )
            self._atomic_replace_dir(tmpdir, self.dataset_path)
        finally:
            if os.path.exists(tmpdir):
                shutil.rmtree(tmpdir, ignore_errors=True)
        self.refresh()

    def delete(
        self,
        where: str,
        partition_by: Optional[str] = None,
        params: Optional[Sequence[Any]] = None,
    ):
        """Delete rows matching the WHERE clause.

        Args:
            where (str): SQL WHERE condition for deletion.
            partition_by (Optional[str]): Partition column.
            params (Optional[Sequence[Any]]): Bind parameters for WHERE.
        """
        view_ident = DuckParquet._quote_ident(self.view_name)
        bind_params = list(params or [])
        select_sql = f"SELECT * FROM {view_ident} WHERE NOT ({where})"
        tmpdir = self._local_tempdir(".")
        try:
            self._copy_select_to_dir(
                select_sql,
                target_dir=tmpdir,
                partition_by=partition_by,
            )
            self._atomic_replace_dir(tmpdir, self.dataset_path)
        finally:
            if os.path.exists(tmpdir):
                shutil.rmtree(tmpdir, ignore_errors=True)
        self.refresh()

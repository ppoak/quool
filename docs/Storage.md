# Storage Module

DuckDB-backed Parquet storage with SQL-like querying capabilities, row-level upsert/update/delete operations, and Hive-style partitioned writes.

## Classes

### DuckTable

Manages a directory of Parquet files through a DuckDB-backed view.

```python
from quool import DuckTable
```

#### Constructor

```python
DuckTable(
    root_path: str,
    name: Optional[str] = None,
    create: bool = False,
    database: Optional[Union[str, duckdb.DuckDBPyConnection]] = None,
    threads: Optional[int] = None,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `root_path` | `str` | Directory path that stores the parquet dataset |
| `name` | `Optional[str]` | The view name. Defaults to directory basename |
| `create` | `bool` | If True, create the directory if it doesn't exist |
| `database` | `Optional[Union[str, duckdb.DuckDBPyConnection]]` | DuckDB connection (externally managed), path to DuckDB database file, or None for in-memory |
| `threads` | `Optional[int]` | Number of threads used for operations |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `empty` | `bool` | True if the parquet path contains no parquet files |
| `schema` | `pd.DataFrame` | Column info (names, types) of the dataset |
| `columns` | `List[str]` | List of all column names in the dataset |

#### Methods

##### `select()`

Query the parquet dataset with flexible SQL generation.

```python
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
) -> pd.DataFrame
```

**Returns:** `pd.DataFrame` with query results

##### `query()`

Execute a raw SQL query and return results as a pandas DataFrame.

```python
def query(self, sql: str, params: Optional[Sequence[Any]] = None) -> pd.DataFrame
```

##### `execute()`

Execute a raw SQL query and return results as a DuckDB relation.

```python
def execute(self, sql: str, params: Optional[Sequence[Any]] = None) -> duckdb.DuckDBPyRelation
```

##### `dpivot()`

Pivot the parquet dataset using DuckDB PIVOT statement.

```python
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
) -> pd.DataFrame
```

**Returns:** `pd.DataFrame` - Pivoted DataFrame

##### `ppivot()`

Wide pivot using pandas pivot_table.

```python
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
) -> pd.DataFrame
```

**Returns:** `pd.DataFrame` - Pivoted DataFrame using pandas pivot_table

##### `upsert()`

Upsert rows from DataFrame according to primary keys, overwriting existing rows.

```python
def upsert(self, df: pd.DataFrame, keys: list, partition_by: Optional[list] = None) -> None
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | `pd.DataFrame` | DataFrame with rows to upsert |
| `keys` | `list` | Primary key column names for deduplication |
| `partition_by` | `Optional[list]` | Partition columns for Hive-style partitioning |

**Raises:** `ValueError` if DataFrame contains duplicate rows based on keys

##### `compact()`

Compact partition directories with multiple parquet files into single parquet files.

```python
def compact(
    self,
    compression: str = "zstd",
    max_workers: int = 8,
    engine: str = "pyarrow",
) -> List[str]
```

**Returns:** `List[str]` - List of relative partition paths that were compacted

##### `refresh()`

Refresh DuckDB view after manual file changes.

##### `close()`

Close the DuckDB connection if owned by this instance.

##### `drop()`

Drop the underlying DuckDB view, if it exists.

#### Context Manager

```python
with DuckTable("/path/to/data") as dt:
    df = dt.select(where="id > 100")
```

---

### DuckPQ

Database-like manager for a directory of Hive-partitioned Parquet tables.

```python
from quool import DuckPQ
```

DuckPQ wraps a single DuckDB connection and a set of Parquet-backed tables (one directory per table) under a common root directory.

#### Constructor

```python
DuckPQ(
    root_path: Union[str, Path],
    database: Optional[Union[str, duckdb.DuckDBPyConnection]] = None,
    config: Optional[Dict[str, Any]] = None,
    threads: Optional[int] = None,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `root_path` | `Union[str, Path]` | Root directory for the Parquet database |
| `database` | `Optional[Union[str, duckdb.DuckDBPyConnection]]` | DuckDB connection or file path |
| `config` | `Optional[Dict[str, Any]]` | Extra DuckDB connection config |
| `threads` | `Optional[int]` | Number of DuckDB threads |

#### Properties

| Property | Type | Description |
|----------|------|-------------|
| `root_path` | `Path` | Root directory path |
| `con` | `duckdb.DuckDBPyConnection` | Shared DuckDB connection |
| `tables` | `Dict[str, DuckTable]` | Table name to DuckTable mapping |

#### Methods

##### `register()`

Register table directories under root_path as DuckTable instances.

```python
def register(self, name: Optional[str] = None) -> None
```

##### `registrable()`

List unregistered table directories under the root path.

```python
def registrable(self) -> List[str]
```

**Returns:** `List[str]` - List of directory names that could be registered

##### `attach()`

Register a pandas DataFrame as a DuckDB relation.

```python
def attach(
    self,
    name: str,
    df: pd.DataFrame,
    replace: bool = True,
    materialize: bool = False,
) -> None
```

##### `select()`

Select from a Parquet-backed table via DuckTable.

```python
def select(
    self,
    table: str,
    columns: Union[str, List[str]] = "*",
    where: Optional[str] = None,
    params: Optional[Sequence[Any]] = None,
    group_by: Optional[Union[str, List[str]]] = None,
    having: Optional[str] = None,
    order_by: Optional[Union[str, List[str]]] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    distinct: bool = False,
) -> pd.DataFrame
```

##### `load()`

Query one or more tables using short-hand `table/field` notation. Automatically handles cross-table JOINs when columns span multiple tables.

```python
def load(
    self,
    columns: Union[str, List[str]],
    where: Optional[str] = None,
    params: Optional[Sequence[Any]] = None,
    group_by: Optional[Union[str, List[str]]] = None,
    having: Optional[str] = None,
    order_by: Optional[Union[str, List[str]]] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
    distinct: bool = False,
    sep: str = "/",
) -> pd.DataFrame
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `columns` | `Union[str, List[str]]` | Column specs in `table/field` or `table/field AS alias` form |
| `where` | `Optional[str]` | WHERE clause filter |
| `params` | `Optional[Sequence[Any]]` | Bind parameters for WHERE clause |
| `group_by` | `Optional[Union[str, List[str]]]` | GROUP BY clause |
| `having` | `Optional[str]` | HAVING clause |
| `order_by` | `Optional[Union[str, List[str]]]` | ORDER BY clause |
| `limit` | `Optional[int]` | Row limit |
| `offset` | `Optional[int]` | Row offset |
| `distinct` | `bool` | Select DISTINCT rows |
| `sep` | `str` | Separator in column specs, defaults to `/` |

**Returns:** `pd.DataFrame` - Query results.

**Examples:**

```python
# Single-table query
db.load(["target/close", "target/volume"], where="code = '000001'")

# Cross-table query (datetime + code are used as JOIN keys)
db.load(["target/close", "quotes/volume"], where="date = '2024-01-01'")

# Explicit alias
db.load(["target/close_post AS close", "quotes/vwap AS vwap"])
```

##### `upsert()`

Upsert rows from a DataFrame into a Parquet-backed table.

```python
def upsert(
    self,
    table: str,
    df: pd.DataFrame,
    keys: List[str],
    partition_by: Optional[List[str]] = None,
) -> None
```

##### `compact()`

Compact partition directories of a Parquet-backed table.

```python
def compact(
    self,
    table: str,
    compression: str = "zstd",
    max_workers: int = 8,
    engine: str = "pyarrow",
) -> List[str]
```

##### `execute()` / `query()`

Execute arbitrary SQL on the shared DuckDB connection.

##### `close()`

Close the underlying DuckDB connection if owned by DuckPQ.

#### Context Manager

```python
with DuckPQ(root_path="/data/tables") as db:
    df = db.select(table="sales", where="revenue > 1000")
```

---

## Usage Examples

### Basic DuckTable Operations

```python
from quool import DuckTable
import pandas as pd

dt = DuckTable("/path/to/parquet_dir", name="my_data", create=True)

df = dt.select(
    columns=["id", "name", "value"],
    where="value > 100",
    order_by="id DESC",
    limit=50
)

if not dt.empty:
    print(f"Schema: {dt.schema}")

dt.refresh()
dt.close()
```

### Using DuckPQ for Multiple Tables

```python
from quool import DuckPQ

db = DuckPQ(root_path="/data/warehouse")
db.register()

print(f"Registered tables: {list(db.tables.keys())}")
print(f"Registrable: {db.registrable()}")

result = db.query("""
    SELECT s.*, p.category
    FROM sales s
    JOIN products p ON s.product_id = p.id
""")
```

### Upsert with Partitioning

```python
from quool import DuckTable
import pandas as pd

dt = DuckTable("/data/sales", name="sales", create=True)

new_rows = pd.DataFrame([
    {"date": "2024-01-01", "region": "US", "revenue": 5000},
    {"date": "2024-01-02", "region": "EU", "revenue": 3000},
])

dt.upsert(new_rows, keys=["date", "region"], partition_by=["region"])
```

### Pivot Operations

```python
from quool import DuckTable

dt = DuckTable("/data/sales")

pivot_df = dt.dpivot(
    index="region",
    columns="month",
    values="revenue",
    aggfunc="sum"
)
```

### Compacting Partitions

```python
from quool import DuckTable

dt = DuckTable("/data/large_dataset")
compacted = dt.compact(compression="zstd", max_workers=4)
```

---
name: storage
description: >
  Authoring guideline for DuckPQ/DuckTable-based Parquet storage in Quool. Covers
  DuckPQ initialization, table discovery and registration, CRUD operations
  (upsert/select/load/query), cross-table JOIN queries using the
  <table><sep><column> pattern, DataFrame attachment, and parquet compaction.
  Encourages using DuckPQ as the top-level database interface rather than
  operating on DuckTable in isolation.
author: quool team
version: 0.0.1
triggers:
  - use DuckPQ storage
  - store data in parquet
  - query parquet files
  - upsert data into parquet
  - load parquet data
  - attach dataframe to duckdb
  - compact parquet files
---

# Storage Skill

> Authoring guideline: read this entire document before writing any code. Every section is load-bearing.

## Role

You are a data engineering specialist. Your job is to produce correct, efficient DuckPQ/DuckTable storage implementations from a user's informal specification. Follow the workflow below step by step. Do not skip steps, do not make assumptions not supported by the user's request, and do not hardcode values the user did not specify.

---

## Core Architecture

Quool provides two layers for Parquet-backed storage:

```
DuckPQ        # Top-level database manager — coordinates multiple DuckTable instances
  └── DuckTable   # Per-table Parquet directory handler (Hive-partitioned)
```

**DuckPQ is the encouraged interface.** It wraps a single DuckDB connection shared across all tables, auto-discovers table directories under a root path, and exposes a unified CRUD API. You should rarely need to instantiate `DuckTable` directly — `DuckPQ.upsert`, `DuckPQ.select`, and `DuckPQ.load` cover most use cases.

`DuckTable` is the underlying building block. Each `DuckTable` instance manages one Parquet directory (one table) and exposes `select`, `upsert`, `compact`, and pivot operations. It is designed to be used internally by `DuckPQ`, not as a standalone entry point.

---

## Step 1 — Initialize DuckPQ

```python
from quool import DuckPQ

db = DuckPQ(
    root_path="/path/to/database",   # adjust: root directory containing table subdirs
    database=None,                    # adjust: ":memory:" (default), a .duckdb file path,
                                     #         or an existing DuckDBPyConnection
    config=None,                     # adjust: extra DuckDB config dict
    threads=4,                       # adjust: number of DuckDB threads
)
```

- `root_path`: Each immediate subdirectory under `root_path` is treated as one table.
- `database`: `None` → in-memory DuckDB (ephemeral); a path string → persistent `.duckdb` file; a `DuckDBPyConnection` → reuse an existing connection (DuckPQ will not close it).
- DuckPQ is a context manager — prefer `with DuckPQ(...) as db:` to ensure the connection is closed.

```python
with DuckPQ(root_path="/path/to/database") as db:
    # work with db
    pass  # connection closed automatically
```

---

## Step 2 — Register Tables

Tables are discovered automatically when you call `register()`:

```python
# Auto-discover and register all subdirectories under root_path as DuckTable views
db.register()

# Register a specific table by name
db.register(name="kline")
```

`register()` scans the filesystem under `root_path` and creates a `DuckTable` for each subdirectory found. After registration, tables are accessible via `db.tables` (a `dict[str, DuckTable]`).

You can also check which table directories exist but are not yet registered:

```python
unregistered = db.registrable()   # returns a list of unregistered table directory names
```

---

## Step 3 — Create / Insert / Upsert Data

Use `upsert()` to insert new rows or update existing rows based on primary keys:

```python
db.upsert(
    table="kline",                   # adjust: table name (directory name)
    df=df_input,                     # adjust: pandas DataFrame to upsert
    keys=["code", "date"],          # adjust: primary key columns for deduplication
    partition_by=["date"],          # adjust: optional Hive partition columns
)
```

**Behavior:**
- If the table directory does not exist yet, it is created automatically.
- If rows with the same `keys` already exist, they are replaced with the incoming rows.
- If `partition_by` is set, data is written as Hive-partitioned parquet files under `root_path/table/`.
- If `partition_by` is `None`, a single `data_0.parquet` file is written.

**Requirements on `df_input`:**
- Must not contain duplicate rows based on `keys`.
- Column names must match what the table expects (no extra columns unless the table schema allows them).
- DO NOT put `keys` column in DataFrame index, it won't be recognized.

```python
# Adjust column names to match actual data
df_input.columns = ["code", "date", "open", "high", "low", "close", "volume"]
db.upsert(table="kline", df=df_input, keys=["code", "date"], partition_by=["date"])
```

---

## Step 4 — Read / Query Data

### `select()` — single-table query

```python
df = db.select(
    table="kline",                   # adjust: table name
    columns="*",                     # adjust: "*" or list of column names
    where="date = '2024-01-01'",     # adjust: optional WHERE clause
    params=None,                     # adjust: optional bind parameters for WHERE
    group_by=None,                   # adjust: optional GROUP BY clause
    having=None,                     # adjust: optional HAVING clause
    order_by="date",                 # adjust: optional ORDER BY clause
    limit=1000,                      # adjust: optional row limit
    offset=None,                     # adjust: optional row offset
    distinct=False,                  # adjust: True for DISTINCT select
)
```

### `load()` — cross-table query via `<table><sep><column>` pattern

Use `load()` when your query spans multiple tables. Columns are specified as `"table/column"` paths:

```python
df = db.load(
    columns=["kline/close", "kline/volume", "financial/pe"],  # adjust to actual tables/columns
    where="kline/date = '2024-01-01'",   # adjust: WHERE clause (table prefixes required)
    params=None,
    group_by=None,
    having=None,
    order_by=None,
    limit=None,
    offset=None,
    distinct=False,
    sep="/",                             # adjust: separator (must match the path pattern)
)
```

**The `<table><sep><column>` pattern:**
- Every column spec is `"table_name<sep>column_name"` (e.g., `"kline/close"` with sep=`"/"`).
- `load()` parses the column specs, determines which tables are needed, and automatically builds a `LEFT JOIN` query using `date` and `code` as join keys.
- Supports `"table/column AS alias"` for renaming in the result.

```python
# Single-table load (also works)
df = db.load(
    columns=["kline/close", "kline/volume"],
    where="code = '000001'",
)

# Cross-table load — automatic JOIN on datetime + code
df = db.load(
    columns=["kline/close", "quotes/volume"],
    where="kline/date = '2024-01-01'",
)
```

### `query()` / `execute()` — raw SQL

For complex queries that `load()` cannot express:

```python
# Returns a pandas DataFrame
df = db.query("SELECT * FROM kline WHERE date = '2024-01-01' LIMIT 100")

# Returns a DuckDB relation (for further chaining)
rel = db.execute("SELECT * FROM kline WHERE date = '2024-01-01'")
```

### `attach()` — register a DataFrame temporarily

Use `attach()` to expose a pandas DataFrame as a DuckDB relation (view-like) or a temporary table, without writing to disk:

```python
db.attach(
    name="temp_signal",    # adjust: name to use in SQL
    df=df_signal,         # adjust: DataFrame to register
    replace=True,         # adjust: drop existing view/table with same name first
    materialize=False,    # adjust: True = TEMP TABLE (persists in connection);
                          #         False = relation/view (ephemeral)
)

# Now you can query it with raw SQL
result = db.query("SELECT * FROM temp_signal WHERE signal > 0")
```

`attach()` is useful for intermediate results, signal DataFrames, or any data that does not need to be persisted to Parquet.

---

## Step 5 — Inspect Schema and Table Metadata

```python
# Get list of registered table names
db.tables.keys()   # dict_keys(["kline", "quotes", "financial"])

# Get columns of a specific table
db.tables["kline"].columns   # list of column names

# Get full schema (column names + types)
db.tables["kline"].schema    # DataFrame with column_name and column_type

# Check if a table's parquet directory is empty
db.tables["kline"].empty     # True if no parquet files exist

# Check which table directories exist but are not registered
db.registrable()              # list of unregistered table names
```

---

## Step 6 — Compact Parquet Files

Over time, many small parquet files accumulate under partitioned table directories. Use `compact()` to merge them:

```python
db.compact(
    table="kline",             # adjust: table name
    compression="zstd",        # adjust: compression codec ('zstd', 'snappy', 'gzip')
    max_workers=8,            # adjust: parallel workers
    engine="pyarrow",         # adjust: 'pyarrow' or 'fastparquet'
)
```

Returns a list of relative partition paths that were compacted.

---

## Step 7 — Close the Connection

```python
db.close()   # closes the DuckDB connection if owned by DuckPQ
```

Or use the context manager for automatic cleanup:

```python
with DuckPQ(root_path="/path/to/database") as db:
    # work here
    pass
# closed automatically
```

---

## DuckTable Direct Usage (Rare)

Only use `DuckTable` directly when you need per-table operations not exposed by `DuckPQ` (e.g., `dpivot`, `ppivot`):

```python
from quool import DuckPQ

db = DuckPQ(root_path="/path/to/database")
table = db.tables["kline"]   # get the DuckTable instance

# DuckTable-level operations
df = table.select(columns="*", where="date = '2024-01-01'")
table.upsert(df=df_new, keys=["code", "date"], partition_by=["date"])
table.refresh()              # refresh DuckDB view after manual file changes
table.compact()

# Pivot operations (DuckTable only)
pivot_df = table.dpivot(
    index="date",
    columns="code",
    values="close",
    aggfunc="first",
    where="date >= '2024-01-01'",
)

wide_df = table.ppivot(
    index="date",
    columns="code",
    values="volume",
    aggfunc="sum",
)
```

---

## Decision Tree — Choosing the Right Method

```
Need to...
│
├─ Initialize a database
│  └─ `DuckPQ(root_path=...)` — prefer context manager
│
├─ Upsert DataFrame to Parquet
│  └─ `db.upsert(table, df, keys, partition_by)`
│
├─ Query a single table
│  ├─ Simple filter/sort/limit → `db.select(table, columns, where, ...)`
│  └─ Complex aggregation → `db.query("SELECT ... FROM table ...")``
│
├─ Query multiple tables (cross-table JOIN)
│  └─ `db.load(["table1/col1", "table2/col2"], where=...)`
│
├─ Register a temporary DataFrame in DuckDB
│  └─ `db.attach(name, df, materialize=False)`
│
├─ Get DuckDB relation for chaining
│  └─ `db.execute("SELECT ...")` → DuckDBPyRelation
│
├─ Inspect table metadata
│  ├─ List tables → `db.tables.keys()`
│  ├─ List columns → `db.tables[name].columns`
│  ├─ Show schema → `db.tables[name].schema`
│  └─ Check if empty → `db.tables[name].empty`
│
└─ Merge small parquet files
   └─ `db.compact(table, compression, max_workers)`
```

---

## The `<table><sep><column>` Path Pattern (load / cross-table)

Every column spec in `load()` follows this format:

```
<table> <sep> <column>
```

| Component | Meaning |
|-----------|---------|
| `table` | Table (directory) name under `root_path` |
| `sep` | Separator — defaults to `"/"` |
| `column` | Column name stored in that table's parquet files |

The path is split by `sep` into exactly two parts: `[table, column]`. **No subdirectory nesting is supported.** If your data is organized under a path like `daily/ohlcv/close`, the table name would be `daily` and the column would be `ohlcv/close` — but this is not recommended. Keep table names simple.

**Example:** Querying close and volume from `kline` table, and pe from `financial` table:

```python
df = db.load(
    columns=["kline/close", "kline/volume", "financial/pe"],
    where="kline/date >= '2024-01-01' AND kline/date <= '2024-12-31'",
    sep="/",
)
```

---

## Critical Gotchas

1. **`load()` auto-determines JOIN keys.** The method uses `datetime` (cast to TIMESTAMP) and `code` as the automatic JOIN keys across tables. Both columns must exist in all tables participating in the cross-table query.

2. **`upsert()` deduplicates by `keys`.** If the incoming DataFrame has duplicate rows based on the key columns, `upsert()` raises a `ValueError`. Deduplicate before calling.

3. **`partition_by` creates Hive partitioning.** When `partition_by=["date"]` is set, the parquet files are written under `root_path/table/date=2024-01-01/` directories. This is the standard Hive-style partitioning scheme that DuckDB's `parquet_scan` with `HIVE_PARTITIONING=1` reads automatically.

4. **`attach()` is ephemeral.** DataFrames registered via `attach()` are only available for the lifetime of the DuckDB connection. They are not written to parquet and do not persist after `close()`.

5 **`register()` is needed for existing tables.** If you start DuckPQ with a `root_path` that already contains table directories, you must call `db.register()` (or `db.register(name="specific_table")`) to create the DuckTable views before querying.

6. **`DuckTable.refresh()` after external file changes.** If you manually add, delete, or modify parquet files outside of DuckTable's API, call `table.refresh()` to update the DuckDB view.

7. **`database=None` means in-memory.** The default `database=None` creates a `:memory:` DuckDB. Data written via `upsert()` persists to parquet files under `root_path`, but the DuckDB metadata/views are lost when the connection closes. Use `database="/path/to/file.duckdb"` for persistent metadata.

8. **`select()` vs `load()` for single tables.** `select()` operates on one table at a time and uses direct SQL. `load()` uses the `<table>/<column>` shorthand and can handle both single and multi-table queries. Prefer `select()` for simple single-table queries — it is more explicit.

---

## Workflow Checklist

Before returning the final implementation, verify:

- [ ] DuckPQ initialized with correct `root_path` (use context manager)
- [ ] `db.register()` called to discover existing table directories
- [ ] `upsert()` uses correct `table` name, `keys`, and `partition_by` matching the actual data
- [ ] `select()` / `load()` / `query()` uses correct column names and WHERE conditions
- [ ] `load()` column specs follow the `<table><sep><column>` format with actual table names
- [ ] `attach()` used correct `name` and `materialize` setting for the use case
- [ ] Table metadata (`columns`, `schema`, `empty`) inspected to confirm structure
- [ ] Connection properly closed via context manager or explicit `close()`
- [ ] No hardcoded paths, table names, column names, or partition columns that should come from user input

# Source

## `Source` (Abstract Base)

**File:** `quool/source.py`

Abstract market data source. Subclasses implement `update()` to advance time and return OHLCV snapshots.

```python
from quool import Source
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `time` | `pd.Timestamp` | Current market timestamp |
| `data` | `pd.DataFrame or None` | Current market snapshot indexed by instrument code |
| `open` | `str` | Column name for open price |
| `high` | `str` | Column name for high price |
| `low` | `str` | Column name for low price |
| `close` | `str` | Column name for close price |
| `volume` | `str` | Column name for volume |

### `Source.__init__`

```python
Source(
    time: pd.Timestamp,              # Initial market timestamp
    data: pd.DataFrame = None,      # Initial market snapshot
    open: str = "open",
    high: str = "high",
    low: str = "low",
    close: str = "close",
    volume: str = "volume",
)
```

### `Source.update`

```python
Source.update() -> pd.DataFrame
```

Advance to the next snapshot and update time. **Must be overridden in subclasses.**

**Returns:** Updated market snapshot.

**Raises:** `NotImplementedError` in the base class.

---

## `DataFrameSource`

**File:** `quool/sources/dataframe.py`

Market data from a pandas DataFrame with a time-based MultiIndex.

```python
from quool import DataFrameSource
```

### `DataFrameSource.__init__`

```python
DataFrameSource(
    data: pd.DataFrame,         # MultiIndex DataFrame: level 0 = time, level 1+ = instrument codes
    open: str = "open",
    high: str = "high",
    low: str = "low",
    close: str = "close",
    volume: str = "volume",
)
```

The first level of the DataFrame index is used as the timeline.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `times` | `pandas.Index` | All timestamps ≤ current time |
| `datas` | `pd.DataFrame` | Historical data up to current time |
| `data` | `pd.DataFrame` | Data slice at current time |

### `DataFrameSource.update`

Advances to the next available timestamp and returns its data slice.

**Returns:** `pd.DataFrame` at new current time, or `None` if no future timestamps remain.

---

## `DuckPQSource`

**File:** `quool/sources/duck.py`

DuckDB/Parquet data source via parquool.

```python
from quool import DuckPQSource
```

### `DuckPQSource.__init__`

```python
DuckPQSource(
    source: DuckPQ,                          # parquool DuckPQ handle
    begin: str,                              # Start date (pandas-parsable)
    end: str,                                # End date (pandas-parsable)
    datetime_col: str = "date",              # Datetime column name in DB
    code_col: str = "code",                  # Instrument code column name in DB
    bar: dict[str, str] = None,              # OHLCV field → factor path mapping
    extra: dict[str, str] = None,            # Extra fields → factor path mapping
    sep: str = "/",                          # Factor path separator
)
```

Default `bar` mapping:
```python
{
    "open": "target/open_post",
    "high": "target/high_post",
    "low": "target/low_post",
    "close": "target/close_post",
    "volume": "target/volume",
}
```

Factor path format: `"table_name/column_name"`.

**Raises:** `ValueError` if bar data spans multiple tables.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `data` | `pd.DataFrame or None` | Current snapshot at current time |
| `datas` | `pd.DataFrame` | Concatenated historical snapshots |

### `DuckPQSource.update`

Advances to the next available timestamp and queries the database.

**Returns:** `pd.DataFrame` indexed by code, or `None` if exhausted.

---

## `RealtimeSource`

**File:** `quool/sources/realtime.py`

Real-time market data from EastMoney API with an in-memory rolling buffer.

```python
from quool import RealtimeSource
```

### `RealtimeSource.__init__`

```python
RealtimeSource(
    proxies: list | dict = None,   # Proxy configuration for HTTP requests
    limit: int = 3000,             # Maximum number of snapshots to retain
)
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `times` | `deque[pd.Timestamp]` | Timestamps of stored snapshots |
| `time` | `pd.Timestamp` | Timestamp of the most recent snapshot |
| `datas` | `pd.DataFrame` | Concatenated historical snapshots (MultiIndex: datetime × code) |
| `data` | `pd.DataFrame` | Most recent snapshot indexed by code |

### `RealtimeSource.update`

Fetches and appends a new snapshot if the current time is within A-share trading hours.

Trading hours: Mon–Fri 09:30–11:30, 13:00–15:00.

**Returns:** Latest snapshot, or `None` if outside trading hours.

---

## `XtDataPreloadSource`

**File:** `quool/sources/xuntou.py`

Preloaded xtquant market data, backed by `DataFrameSource`.

```python
from quool import XtDataPreloadSource
```

### `XtDataPreloadSource.__init__`

```python
XtDataPreloadSource(
    path: str,                 # xtquant data directory
    begin: str,                # Start time (pandas-parsable)
    end: str,                  # End time (pandas-parsable)
    period: str = "1d",        # Sampling period (e.g., '1d', '1m')
    sector: str = "沪深A股",    # Stock sector name recognized by xtdata
)
```

Loads historical data from xtquant into a MultiIndex DataFrame (datetime × code) and passes it to `DataFrameSource`.

**Raises:** `ImportError` if xtquant is not installed.

---

## Helper Functions

### `is_trading_time` (realtime.py)

```python
is_trading_time(time: str) -> bool
```

Check whether a timestamp falls within A-share trading hours (Mon–Fri 09:30–11:30, 13:00–15:00).

### `read_realtime` (realtime.py)

```python
read_realtime(proxies: list[dict] = None) -> pd.DataFrame
```

Fetch real-time A-share market data from EastMoney API.

Returns DataFrame indexed by `code` with columns: `serial_num, name, close, change_pct, change_amt, volume, turnover, amplitude, high, low, open, prev_close, volume_ratio, turnover_rate, pe_ratio, pb_ratio, market_cap, float_market_cap, rise_speed, 5min_change, 60day_change, ytd_change`.

### `parse_factor_path` (duck.py)

```python
parse_factor_path(path: str, sep: str = "/") -> Tuple[str, str]
```

Parse a factor path into `(table, column)` tuple.

**Raises:** `TypeError` if path is not a string; `ValueError` if path format is invalid.

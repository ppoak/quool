### DuckParquet API Reference: `upsert_from_df`

> _Part of the `DuckParquet` documentation. See full index in [`doc/README.md`](../README.md)_

#### Method: `upsert_from_df`

Upsert (insert or update) records into a Parquet dataset from a pandas DataFrame using given primary keys. This method will write new rows if primary key does not exist, or overwrite previous rows with the same primary key.

##### Definition

```python
upsert_from_df(df: pd.DataFrame, keys: list, partition_by: Optional[list] = None)
```

##### Parameters

| Name         | Type                    | Description                                                        |
|--------------|-------------------------|--------------------------------------------------------------------|
| df           | pd.DataFrame            | Source data. Must contain all columns (keys, partitions, values)   |
| keys         | list[str]               | Primary key columns (row uniqueness)                               |
| partition_by | Optional[list[str]]     | Partition columns for Hive-style layout. Usually a date column     |

##### Returns
_None – operation is performed in-place on dataset._


#### Example: A-Share Quotes Upsert

Suppose you want to maintain a local partitioned A股行情 (A-share quotes) parquet dataset, keyed by 股票代码和时间(`code`, `time`), 按日期 (`date`) 分区。可以如下使用：

```python
import pandas as pd
from datetime import datetime, timedelta
from quool.sources.util import DuckParquet

# Prepare example quotes
data = []
for h in range(9, 16):
    dt = datetime(2023, 1, 1, h)
    data.append({
        'code': '000001.SZ',
        'time': dt,
        'date': dt.strftime('%Y-%m-%d'),
        'open': 10 + h,
        'high': 12 + h,
        'low': 9 + h,
        'close': 11 + h,
        'volume': 10000 + h*100
    })
df = pd.DataFrame(data)

# Initial upsert into empty dir:
parquet_dir = 'quotes_data_dir'   # folder for your dataset
parq = DuckParquet(parquet_dir, threads=4) # you can set thread number here, default to 1
parq.upsert_from_df(df, keys=["code", "time"], partition_by=["date"])

# If later you want to add new rows or update some existing ones:
df2 = df.copy()
df2.loc[0, 'close'] = 999.0  # Modify one row
parq.upsert_from_df(df2, keys=["code", "time"], partition_by=["date"])

# Query result
df_res = parq.select()
print(df_res)  # Will reflect your changes

# Always close after finish
parq.close()
```


#### Notes

- Partition column (`partition_by`) is optional but highly recommended for time-series data.
- Primary key constraint is emulated by overwriting rows with duplicate keys.
- If multiple same-key rows exist in your DataFrame, **the first occurrence is used**. Remove duplicates ahead of time if needed.
- Works with any pyarrow-supported parquet data type.

For further customization, see the [full class documentation](DuckParquet.md) or the [project README](../README.md).

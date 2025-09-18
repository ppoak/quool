import os
import shutil
import pandas as pd
import pytest
from datetime import datetime, timedelta

from quool.sources.util import DuckParquet


def make_virtual_quotes(start_date, num_codes=2, num_days=2, freq="1h"):
    codes = [f"{c:06d}.SZ" for c in range(1, num_codes + 1)]
    all_records = []
    for code in codes:
        for day_offset in range(num_days):
            day = start_date + timedelta(days=day_offset)
            for h in range(9, 16):  # 9:00-15:00 one record per hour
                dt = day.replace(hour=h, minute=0, second=0, microsecond=0)
                record = {
                    "code": code,
                    "time": dt,
                    "date": dt.strftime("%Y-%m-%d"),
                    "open": 10.0 + h + day_offset,
                    "high": 13.0 + h + day_offset,
                    "low": 9.0 + h + day_offset,
                    "close": 12.0 + h + day_offset,
                    "volume": 10000 + h * 100 + day_offset * 1000,
                }
                all_records.append(record)
    df = pd.DataFrame(all_records)
    df["time"] = pd.to_datetime(df["time"])
    return df


def test_upsert_scenario():
    # Create temp directory in current working directory
    tmpdir = "__duckparquet_tmpdir__"
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)
    os.makedirs(tmpdir)
    try:
        # 1. Insert data into new empty directory
        df1 = make_virtual_quotes(datetime(2023, 1, 1), num_codes=1, num_days=1)
        dp = DuckParquet(tmpdir)
        dp.upsert_from_df(df1, keys=["code", "time"], partition_by=["date"])
        df_read = dp.select()
        assert len(df_read) == len(df1)
        assert set(df_read["code"]) == set(df1["code"])
        dp.close()

        # 2. Upsert with partial overwrite (some overlap, some new)
        dp = DuckParquet(tmpdir)
        df2 = df1.copy()
        df2.loc[0, "close"] += 1.0  # modify one row
        new_row = df1.iloc[0].copy()
        new_row["time"] += timedelta(hours=10)
        new_row["close"] = 99.0
        df2 = pd.concat([df2, pd.DataFrame([new_row])], ignore_index=True)
        dp.upsert_from_df(df2, keys=["code", "time"], partition_by=["date"])
        df_read2 = dp.select()
        assert len(df_read2) == len(df1) + 1
        target = df_read2[
            (df_read2["code"] == df1.iloc[0]["code"])
            & (df_read2["time"] == df1.iloc[0]["time"])
        ]
        assert target["close"].iloc[0] == df1.iloc[0]["close"] + 1.0
        dp.close()

        # 3. Upsert all (full overwrite)
        dp = DuckParquet(tmpdir)
        df3 = df_read2.copy()
        df3["close"] += 1.0
        df3["date"] = df3["date"].dt.strftime("%Y-%m-%d")
        dp.upsert_from_df(df3, keys=["code", "time"], partition_by=["date"])
        df_read3 = dp.select()
        assert len(df_read3) == len(df_read2)
        assert all((df_read3["close"] - df_read2["close"]) == 1.0)
        dp.close()
    finally:
        # Clean up temp directory
        shutil.rmtree(tmpdir)

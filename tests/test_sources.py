"""Tests for DataFrameSource, RealtimeSource, DuckPQSource, and XtDataPreloadSource."""

import pandas as pd
import pytest

from quool import (
    DataFrameSource,
    RealtimeSource,
    DuckPQSource,
    XtDataPreloadSource,
    DuckPQ,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_multiindex_df():
    """Create a MultiIndex DataFrame with 3 dates x 2 codes.

    Structure:
    - Dates: 2024-01-02, 2024-01-03, 2024-01-04
    - Codes: 000001.SZ, 000002.SZ
    - Columns: open, high, low, close, volume
    """
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    codes = ["000001.SZ", "000002.SZ"]

    # Build records directly without using dict with mixed key types
    records = []
    ohlcv_data = {
        "000001.SZ": {
            pd.to_datetime("2024-01-02"): {"open": 10.0, "high": 10.5, "low": 9.5, "close": 10.2, "volume": 10000},
            pd.to_datetime("2024-01-03"): {"open": 10.2, "high": 10.8, "low": 10.0, "close": 10.5, "volume": 12000},
            pd.to_datetime("2024-01-04"): {"open": 10.5, "high": 11.0, "low": 10.3, "close": 10.8, "volume": 15000},
        },
        "000002.SZ": {
            pd.to_datetime("2024-01-02"): {"open": 20.0, "high": 20.5, "low": 19.5, "close": 20.2, "volume": 20000},
            pd.to_datetime("2024-01-03"): {"open": 20.2, "high": 20.8, "low": 20.0, "close": 20.5, "volume": 22000},
            pd.to_datetime("2024-01-04"): {"open": 20.5, "high": 21.0, "low": 20.3, "close": 20.8, "volume": 25000},
        },
    }

    for code in codes:
        for date in dates:
            row = ohlcv_data[code][date]
            records.append({
                "code": code,
                "datetime": date,
                **row,
            })

    df = pd.DataFrame(records)
    df = df.set_index(["datetime", "code"])
    df = df.sort_index()  # Sort MultiIndex to avoid slicing issues
    return df


@pytest.fixture
def duckpq_in_memory():
    """Create an in-memory DuckPQ instance with a test table."""
    import tempfile, shutil
    path = tempfile.mkdtemp(prefix="quool_test_")
    db = DuckPQ(root_path=path, database=":memory:")
    # Register a test table with sample OHLCV data
    test_df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        "code": ["000001.SZ", "000001.SZ", "000001.SZ"],
        "open_post": [10.0, 10.2, 10.5],
        "high_post": [10.5, 10.8, 11.0],
        "low_post": [9.5, 10.0, 10.3],
        "close_post": [10.2, 10.5, 10.8],
        "volume": [10000, 12000, 15000],
    })
    db.attach("target", test_df, replace=True)
    yield db
    db.close()
    shutil.rmtree(path, ignore_errors=True)


# =============================================================================
# DataFrameSource Tests
# =============================================================================

class TestDataFrameSource:
    """Tests for DataFrameSource."""

    def test_create_with_multiindex_dataframe(self, sample_multiindex_df):
        """Create DataFrameSource with MultiIndex DataFrame."""
        source = DataFrameSource(sample_multiindex_df)

        assert source.time == pd.to_datetime("2024-01-02")
        assert source.open == "open"
        assert source.high == "high"
        assert source.low == "low"
        assert source.close == "close"
        assert source.volume == "volume"

    def test_initial_time_is_min_timestamp(self, sample_multiindex_df):
        """Initial time should be the minimum timestamp from the index."""
        source = DataFrameSource(sample_multiindex_df)
        assert source.time == sample_multiindex_df.index.get_level_values(0).min()

    def test_update_advances_time(self, sample_multiindex_df):
        """update() should advance to the next timestamp."""
        source = DataFrameSource(sample_multiindex_df)
        initial_time = source.time

        result = source.update()

        assert source.time > initial_time
        assert source.time == pd.to_datetime("2024-01-03")
        assert result is not None

    def test_update_returns_none_when_exhausted(self, sample_multiindex_df):
        """update() should return None when there are no more timestamps."""
        source = DataFrameSource(sample_multiindex_df)

        # Advance through all timestamps
        for _ in range(3):
            source.update()

        # Next update should return None
        result = source.update()
        assert result is None

    def test_time_advances_correctly_after_n_updates(self, sample_multiindex_df):
        """After N updates, time should be the Nth timestamp."""
        source = DataFrameSource(sample_multiindex_df)

        assert source.time == pd.to_datetime("2024-01-02")

        source.update()
        assert source.time == pd.to_datetime("2024-01-03")

        source.update()
        assert source.time == pd.to_datetime("2024-01-04")

    def test_data_property_returns_current_snapshot(self, sample_multiindex_df):
        """data property should return DataFrame slice at current time."""
        source = DataFrameSource(sample_multiindex_df)

        data = source.data

        assert isinstance(data, pd.DataFrame)
        assert data.index.name == "code"
        # Should contain both codes at the initial time
        assert set(data.index) == {"000001.SZ", "000002.SZ"}

    def test_data_changes_after_update(self, sample_multiindex_df):
        """data should reflect the current time after update()."""
        source = DataFrameSource(sample_multiindex_df)

        initial_data = source.data.copy()
        source.update()
        updated_data = source.data

        # The data should be different after update
        assert not updated_data.equals(initial_data)

    def test_datas_property_accumulates_history(self, sample_multiindex_df):
        """datas property should accumulate all data up to current time."""
        source = DataFrameSource(sample_multiindex_df)

        # At initial time (2024-01-02), datas should only contain 2 rows
        datas_t1 = source.datas
        assert len(datas_t1) == 2

        source.update()

        # After one update (2024-01-03), datas should contain 4 rows
        datas_t2 = source.datas
        assert len(datas_t2) == 4

        source.update()

        # After two updates (2024-01-04), datas should contain 6 rows
        datas_t3 = source.datas
        assert len(datas_t3) == 6

    def test_times_property_returns_timestamps_up_to_current(self, sample_multiindex_df):
        """times property should return all timestamps <= current time."""
        source = DataFrameSource(sample_multiindex_df)

        times_t1 = source.times
        assert len(times_t1) == 1
        assert times_t1[0] == pd.to_datetime("2024-01-02")

        source.update()

        times_t2 = source.times
        assert len(times_t2) == 2

        source.update()

        times_t3 = source.times
        assert len(times_t3) == 3

    def test_str_includes_time(self, sample_multiindex_df):
        """String representation should include current time."""
        source = DataFrameSource(sample_multiindex_df)
        str_repr = str(source)
        assert "DataFrameSource" in str_repr
        assert "2024-01-02" in str_repr


# =============================================================================
# RealtimeSource Tests
# =============================================================================

class TestRealtimeSourceIsTradingTime:
    """Tests for is_trading_time function via RealtimeSource module."""

    def test_weekday_morning_trading_hour(self):
        """09:35 on Tuesday 2024-01-02 should be a trading time."""
        from quool.sources.realtime import is_trading_time
        assert is_trading_time("2024-01-02 09:35:00") is True

    def test_weekday_11_30_exact(self):
        """11:30 on Tuesday 2024-01-02 should be a trading time (boundary)."""
        from quool.sources.realtime import is_trading_time
        assert is_trading_time("2024-01-02 11:30:00") is True

    def test_lunch_break(self):
        """12:00 on Tuesday 2024-01-02 should NOT be a trading time (lunch break)."""
        from quool.sources.realtime import is_trading_time
        assert is_trading_time("2024-01-02 12:00:00") is False

    def test_weekday_afternoon_trading_hour(self):
        """14:00 on Tuesday 2024-01-02 should be a trading time."""
        from quool.sources.realtime import is_trading_time
        assert is_trading_time("2024-01-02 14:00:00") is True

    def test_after_market_close(self):
        """15:05 on Tuesday 2024-01-02 should NOT be a trading time (after close)."""
        from quool.sources.realtime import is_trading_time
        assert is_trading_time("2024-01-02 15:05:00") is False

    def test_saturday(self):
        """Saturday 2024-01-06 should NOT be a trading time."""
        from quool.sources.realtime import is_trading_time
        assert is_trading_time("2024-01-06 10:00:00") is False

    def test_sunday(self):
        """Sunday 2024-01-07 should NOT be a trading time."""
        from quool.sources.realtime import is_trading_time
        assert is_trading_time("2024-01-07 10:00:00") is False

    def test_weekday_before_market_open(self):
        """08:00 on Tuesday 2024-01-02 should NOT be a trading time (before open)."""
        from quool.sources.realtime import is_trading_time
        assert is_trading_time("2024-01-02 08:00:00") is False

    def test_weekday_after_market_close(self):
        """16:00 on Tuesday 2024-01-02 should NOT be a trading time (after close)."""
        from quool.sources.realtime import is_trading_time
        assert is_trading_time("2024-01-02 16:00:00") is False


# =============================================================================
# DuckPQSource Tests
# =============================================================================

class TestDuckPQSource:
    """Tests for DuckPQSource."""

    def test_create_with_duckpq(self, duckpq_in_memory):
        """Create DuckPQSource using in-memory DuckPQ with registered table."""
        source = DuckPQSource(
            source=duckpq_in_memory,
            begin="2024-01-02",
            end="2024-01-04",
            datetime_col="date",
            code_col="code",
            bar={
                "open": "target/open_post",
                "high": "target/high_post",
                "low": "target/low_post",
                "close": "target/close_post",
                "volume": "target/volume",
            },
            sep="/",
        )

        # DuckPQSource calls update() at end of __init__, so initial time is first timestamp
        assert source.time == pd.to_datetime("2024-01-02")

    def test_update_advances_through_time_bars(self, duckpq_in_memory):
        """update() should advance through available timestamps."""
        source = DuckPQSource(
            source=duckpq_in_memory,
            begin="2024-01-02",
            end="2024-01-04",
            datetime_col="date",
            code_col="code",
            bar={
                "open": "target/open_post",
                "high": "target/high_post",
                "low": "target/low_post",
                "close": "target/close_post",
                "volume": "target/volume",
            },
            sep="/",
        )

        # DuckPQSource calls update() in __init__, so initial time is 2024-01-02
        initial_time = source.time
        assert initial_time == pd.to_datetime("2024-01-02")

        result = source.update()

        # After first update, time should be 2024-01-03
        assert source.time == pd.to_datetime("2024-01-03")
        assert result is not None

        result = source.update()

        # After second update, time should be 2024-01-04
        assert source.time == pd.to_datetime("2024-01-04")

        result = source.update()

        # Next update should return None (no more future timestamps)
        result = source.update()
        assert result is None
        assert result is None

    def test_data_returns_current_snapshot(self, duckpq_in_memory):
        """data property should return current snapshot DataFrame."""
        source = DuckPQSource(
            source=duckpq_in_memory,
            begin="2024-01-02",
            end="2024-01-04",
            datetime_col="date",
            code_col="code",
            bar={
                "open": "target/open_post",
                "high": "target/high_post",
                "low": "target/low_post",
                "close": "target/close_post",
                "volume": "target/volume",
            },
            sep="/",
        )

        source.update()

        data = source.data

        assert isinstance(data, pd.DataFrame)
        assert "open" in data.columns
        assert "high" in data.columns
        assert "low" in data.columns
        assert "close" in data.columns
        assert "volume" in data.columns

    def test_data_reflects_time(self, duckpq_in_memory):
        """data should reflect the current time after update."""
        source = DuckPQSource(
            source=duckpq_in_memory,
            begin="2024-01-02",
            end="2024-01-04",
            datetime_col="date",
            code_col="code",
            bar={
                "open": "target/open_post",
                "high": "target/high_post",
                "low": "target/low_post",
                "close": "target/close_post",
                "volume": "target/volume",
            },
            sep="/",
        )

        source.update()
        data_t1 = source.data.copy()
        time_t1 = source.time

        source.update()
        data_t2 = source.data.copy()
        time_t2 = source.time

        # Data at different times should be different
        assert time_t2 > time_t1
        assert not data_t2.equals(data_t1)


# =============================================================================
# XtDataPreloadSource Tests
# =============================================================================

class TestXtDataPreloadSource:
    """Tests for XtDataPreloadSource."""

    def test_inherits_from_dataframe_source(self):
        """XtDataPreloadSource should be an instance of DataFrameSource."""
        assert issubclass(XtDataPreloadSource, DataFrameSource)

    def test_is_instance_of_dataframe_source(self):
        """XtDataPreloadSource instances should be recognized as DataFrameSource.

        Note: This test verifies inheritance at the class level since
        XtDataPreloadSource requires xtquant which may not be available.
        """
        # The class itself should be a subclass of DataFrameSource
        assert issubclass(XtDataPreloadSource, DataFrameSource)

        # isinstance check requires an actual instance, but XtDataPreloadSource.__init__
        # calls xtquant which may not be installed, so we verify the inheritance
        # at class level only
        assert DataFrameSource in XtDataPreloadSource.__mro__

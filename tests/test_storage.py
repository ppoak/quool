"""Tests for quool.storage: DuckTable and DuckPQ."""
import tempfile
import shutil
import pandas as pd
import pytest
from quool import DuckTable, DuckPQ


@pytest.fixture
def temp_root():
    """Create a temporary root directory for DuckPQ tests."""
    path = tempfile.mkdtemp(prefix="quool_test_")
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def duckpq_in_memory():
    """In-memory DuckPQ for tests that don't need files."""
    path = tempfile.mkdtemp(prefix="quool_test_")
    db = DuckPQ(root_path=path, database=":memory:")
    yield db
    db.close()
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def duckpq_with_tables(temp_root):
    """DuckPQ with two registered parquet-backed tables."""
    import os
    db = DuckPQ(root_path=temp_root, database=":memory:")

    # Create two table directories with parquet files
    import pandas as pd

    # Table 1: target
    target_dir = f"{temp_root}/target"
    os.makedirs(target_dir, exist_ok=True)
    pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        "code": ["000001.SZ", "000001.SZ", "000001.SZ"],
        "close": [10.2, 10.5, 10.8],
        "volume": [10000, 12000, 15000],
    }).to_parquet(f"{target_dir}/data.parquet")

    # Table 2: quotes
    quotes_dir = f"{temp_root}/quotes"
    os.makedirs(quotes_dir, exist_ok=True)
    pd.DataFrame({
        "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        "code": ["000001.SZ", "000001.SZ", "000001.SZ"],
        "bid": [10.1, 10.4, 10.7],
        "ask": [10.3, 10.6, 10.9],
    }).to_parquet(f"{quotes_dir}/data.parquet")

    db.register()
    yield db
    db.close()


class TestDuckTable:
    """Tests for DuckTable with a real parquet directory."""

    def test_select_star(self, temp_root):
        """select('*') returns all rows."""
        import pyarrow.parquet as pq
        df = pd.DataFrame({
            "code": ["000001.SZ", "000002.SZ"],
            "close": [10.2, 20.5],
        })
        pq.write_table(pa.table_from_pandas(df), f"{temp_root}/data.parquet") if False else \
            df.to_parquet(f"{temp_root}/data.parquet", engine="pyarrow")

        dt = DuckTable(temp_root, database=":memory:")
        result = dt.select("*")
        assert len(result) == 2
        assert "code" in result.columns
        dt.close()

    def test_select_specific_columns(self, temp_root):
        """select(columns=['code', 'close']) returns only those columns."""
        pd.DataFrame({
            "code": ["000001.SZ"],
            "close": [10.2],
            "volume": [10000],
        }).to_parquet(f"{temp_root}/data.parquet", engine="pyarrow")

        dt = DuckTable(temp_root, database=":memory:")
        result = dt.select(columns=["code", "close"])
        assert list(result.columns) == ["code", "close"]
        assert len(result) == 1
        dt.close()

    def test_select_with_where(self, temp_root):
        """select(where='code = ...') filters correctly."""
        pd.DataFrame({
            "code": ["000001.SZ", "000002.SZ"],
            "close": [10.2, 20.5],
        }).to_parquet(f"{temp_root}/data.parquet", engine="pyarrow")

        dt = DuckTable(temp_root, database=":memory:")
        result = dt.select("*", where="code = '000001.SZ'")
        assert len(result) == 1
        assert result.iloc[0]["code"] == "000001.SZ"
        dt.close()

    def test_ppivot(self, temp_root):
        """ppivot creates a wide pivot table."""
        pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
            "code": ["000001.SZ", "000002.SZ", "000001.SZ", "000002.SZ"],
            "close": [10.2, 20.5, 10.5, 20.8],
        }).to_parquet(f"{temp_root}/data.parquet", engine="pyarrow")

        dt = DuckTable(temp_root, database=":memory:")
        pivoted = dt.ppivot(index="date", columns="code", values="close")
        assert pivoted.shape[0] == 2
        assert "000001.SZ" in pivoted.columns
        dt.close()

    def test_schema_property(self, temp_root):
        """schema property returns column info DataFrame."""
        pd.DataFrame({
            "code": ["000001.SZ"],
            "close": [10.2],
        }).to_parquet(f"{temp_root}/data.parquet", engine="pyarrow")

        dt = DuckTable(temp_root, database=":memory:")
        schema = dt.schema
        assert isinstance(schema, pd.DataFrame)
        assert len(schema) >= 2
        dt.close()

    def test_columns_property(self, temp_root):
        """columns property returns list of column names."""
        pd.DataFrame({
            "code": ["000001.SZ"],
            "close": [10.2],
            "volume": [10000],
        }).to_parquet(f"{temp_root}/data.parquet", engine="pyarrow")

        dt = DuckTable(temp_root, database=":memory:")
        cols = dt.columns
        assert isinstance(cols, list)
        assert "code" in cols
        assert "close" in cols
        dt.close()

    def test_execute_raw_sql(self, temp_root):
        """execute(sql) runs raw SQL and returns DuckDB relation."""
        pd.DataFrame({
            "code": ["000001.SZ"],
            "close": [10.2],
        }).to_parquet(f"{temp_root}/data.parquet", engine="pyarrow")

        dt = DuckTable(temp_root, database=":memory:")
        view_name = dt.view_name
        rel = dt.execute(f"SELECT COUNT(*) AS cnt FROM {view_name}")
        result = rel.df()
        assert "cnt" in result.columns
        dt.close()

    def test_empty_property(self, temp_root):
        """empty property returns True when no parquet files exist."""
        import os
        os.makedirs(f"{temp_root}/empty_dir", exist_ok=True)
        dt = DuckTable(f"{temp_root}/empty_dir", database=":memory:", create=True)
        assert dt.empty is True
        dt.close()

    def test_upsert_new_rows(self, temp_root):
        """upsert(df, keys) inserts new rows and updates existing by key."""
        pd.DataFrame({
            "code": ["000001.SZ"],
            "close": [10.2],
            "volume": [10000],
        }).to_parquet(f"{temp_root}/data.parquet", engine="pyarrow")

        dt = DuckTable(temp_root, database=":memory:")
        new_rows = pd.DataFrame({
            "code": ["000002.SZ"],
            "close": [20.5],
            "volume": [8000],
        })
        dt.upsert(new_rows, keys=["code"])
        # Verify the new code is present (old parquet may coexist; fresh instance to confirm)
        result = dt.select("*")
        codes = set(result["code"])
        assert "000002.SZ" in codes
        assert "000001.SZ" in codes
        dt.close()

    def test_context_manager(self, temp_root):
        """DuckTable can be used as context manager."""
        pd.DataFrame({
            "code": ["000001.SZ"],
            "close": [10.2],
        }).to_parquet(f"{temp_root}/data.parquet", engine="pyarrow")

        with DuckTable(temp_root, database=":memory:") as dt:
            result = dt.select("*")
            assert len(result) == 1

    def test_str_repr(self, temp_root):
        """String representation includes path and columns."""
        pd.DataFrame({
            "code": ["000001.SZ"],
            "close": [10.2],
        }).to_parquet(f"{temp_root}/data.parquet", engine="pyarrow")

        dt = DuckTable(temp_root, database=":memory:")
        s = str(dt)
        assert "DuckTable" in s
        dt.close()


class TestDuckPQ:
    """Tests for DuckPQ."""

    def test_create_in_memory(self):
        """DuckPQ can be created with in-memory database."""
        path = tempfile.mkdtemp(prefix="quool_test_")
        db = DuckPQ(root_path=path, database=":memory:")
        assert db is not None
        db.close()
        shutil.rmtree(path, ignore_errors=True)

    def test_attach_dataframe(self, duckpq_in_memory):
        """attach(name, df) registers a DataFrame in DuckDB."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        duckpq_in_memory.attach("test", df)
        result = duckpq_in_memory.query("SELECT COUNT(*) AS cnt FROM test")
        assert result.iloc[0]["cnt"] == 2

    def test_attach_with_replace(self, duckpq_in_memory):
        """attach with replace=True overwrites existing view."""
        df1 = pd.DataFrame({"a": [1]})
        df2 = pd.DataFrame({"a": [1, 2, 3]})
        duckpq_in_memory.attach("test", df1)
        duckpq_in_memory.attach("test", df2, replace=True)
        result = duckpq_in_memory.query("SELECT COUNT(*) AS cnt FROM test")
        assert result.iloc[0]["cnt"] == 3

    def test_tables_property_empty_initially(self, duckpq_in_memory):
        """tables property returns empty dict when no parquet tables registered."""
        assert len(duckpq_in_memory.tables) == 0

    def test_attach_makes_data_queryable(self, duckpq_in_memory):
        """attach(name, df) makes data queryable via tables dict is empty but data is accessible."""
        df = pd.DataFrame({"a": [1, 2]})
        duckpq_in_memory.attach("test", df)
        result = duckpq_in_memory.query("SELECT * FROM test")
        assert len(result) == 2

    def test_select_delegates_to_ducktable(self, duckpq_in_memory):
        """select(table, ...) delegates to DuckTable.select."""
        df = pd.DataFrame({"code": ["000001.SZ"], "close": [10.2]})
        duckpq_in_memory.attach("target", df)
        result = duckpq_in_memory.select("target", columns="*")
        assert len(result) == 1
        assert result.iloc[0]["code"] == "000001.SZ"

    def test_load_single_table_field(self, duckpq_in_memory):
        """load('table/field') queries a single table."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02"]),
            "code": ["000001.SZ"],
            "close": [10.2],
        })
        duckpq_in_memory.attach("target", df)
        result = duckpq_in_memory.load(["target/close"])
        assert "close" in result.columns

    def test_load_cross_table_join(self, duckpq_in_memory):
        """load with fields from multiple tables builds a JOIN query."""
        target_df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "code": ["000001.SZ", "000001.SZ"],
            "close": [10.2, 10.5],
        })
        quotes_df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "code": ["000001.SZ", "000001.SZ"],
            "bid": [10.1, 10.4],
        })
        duckpq_in_memory.attach("target", target_df)
        duckpq_in_memory.attach("quotes", quotes_df)
        result = duckpq_in_memory.load(["target/close", "quotes/bid"])
        assert "close" in result.columns
        assert "bid" in result.columns
        assert len(result) >= 1

    def test_load_with_where(self, duckpq_in_memory):
        """load(columns, where=...) filters results using raw SQL WHERE clause."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "code": ["000001.SZ", "000001.SZ"],
            "close": [10.2, 10.5],
        })
        duckpq_in_memory.attach("target", df)
        result = duckpq_in_memory.load(
            ["target/close"],
            where="close > 10.3"
        )
        assert len(result) == 1

    def test_execute_raw_sql(self, duckpq_in_memory):
        """execute(sql) runs arbitrary SQL on shared connection."""
        duckpq_in_memory.attach("test", pd.DataFrame({"a": [1, 2]}))
        result = duckpq_in_memory.execute("SELECT SUM(a) AS total FROM test").df()
        assert result.iloc[0]["total"] == 3

    def test_query_returns_dataframe(self, duckpq_in_memory):
        """query(sql) returns pandas DataFrame."""
        duckpq_in_memory.attach("test", pd.DataFrame({"a": [1, 2]}))
        result = duckpq_in_memory.query("SELECT * FROM test")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2

    def test_close(self, duckpq_in_memory):
        """close() closes the connection without error."""
        duckpq_in_memory.close()
        assert len(duckpq_in_memory.tables) == 0

    def test_context_manager(self):
        """DuckPQ can be used as context manager."""
        path = tempfile.mkdtemp(prefix="quool_test_")
        try:
            with DuckPQ(root_path=path, database=":memory:") as db:
                db.attach("test", pd.DataFrame({"a": [1]}))
                result = db.query("SELECT * FROM test")
                assert len(result) == 1
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def test_str_repr(self, duckpq_in_memory):
        """String representation includes tables."""
        duckpq_in_memory.attach("test", pd.DataFrame({"a": [1]}))
        s = str(duckpq_in_memory)
        assert "DuckTable" in s

    def test_upsert_delegates_to_ducktable(self, duckpq_in_memory):
        """upsert(table, df, keys) delegates to DuckTable.upsert."""
        target_df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02"]),
            "code": ["000001.SZ"],
            "close": [10.2],
        })
        duckpq_in_memory.attach("target", target_df)
        duckpq_in_memory.upsert("target", target_df, keys=["code"])
        # Should not raise


class TestDuckPQWithFiles:
    """Integration tests for DuckPQ with real parquet files."""

    def test_register_auto_discovers_tables(self, duckpq_with_tables):
        """register() auto-discovers table directories."""
        tables = duckpq_with_tables.tables
        assert "target" in tables
        assert "quotes" in tables

    def test_registrable_lists_unregistered(self, duckpq_with_tables):
        """registrable() returns untracked directory names."""
        reg = duckpq_with_tables.registrable()
        # After register(), all should be tracked
        assert len(reg) == 0

    def test_load_with_sep_parameter(self, duckpq_in_memory):
        """load() supports custom separator for table/field notation."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02"]),
            "code": ["000001.SZ"],
            "close": [10.2],
        })
        duckpq_in_memory.attach("t", df)
        result = duckpq_in_memory.load(["t/close"], sep="/")
        assert "close" in result.columns

    def test_load_with_limit(self, duckpq_in_memory):
        """load(columns, limit=N) limits results."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
            "code": ["000001.SZ"] * 3,
            "close": [10.2, 10.5, 10.8],
        })
        duckpq_in_memory.attach("target", df)
        result = duckpq_in_memory.load(["target/close"], limit=2)
        assert len(result) == 2

    def test_load_with_order_by(self, duckpq_in_memory):
        """load(columns, order_by=...) orders results."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-01"]),
            "code": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "close": [10.8, 10.5, 10.2],
        })
        duckpq_in_memory.attach("target", df)
        result = duckpq_in_memory.load(["target/close"], order_by="date ASC")
        assert result.iloc[0]["close"] == 10.2

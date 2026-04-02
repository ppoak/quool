"""Shared pytest fixtures for quool tests."""
import pandas as pd
import pytest
from quool import DataFrameSource, FixedRateCommission, FixedRateSlippage, Broker


@pytest.fixture
def ohlcv_data():
    """Sample OHLCV DataFrame with MultiIndex (datetime, code)."""
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    codes = ["000001.SZ", "000001.SZ"]
    index = pd.MultiIndex.from_arrays([dates, codes], names=["datetime", "code"])
    df = pd.DataFrame(
        {
            "open": [10.0, 10.5],
            "high": [10.5, 11.0],
            "low": [9.8, 10.2],
            "close": [10.2, 10.8],
            "volume": [10000, 15000],
        },
        index=index,
    )
    return df


@pytest.fixture
def source(ohlcv_data):
    return DataFrameSource(ohlcv_data)


@pytest.fixture
def commission():
    return FixedRateCommission()


@pytest.fixture
def slippage():
    return FixedRateSlippage()


@pytest.fixture
def broker(commission, slippage):
    b = Broker(commission=commission, slippage=slippage)
    b._time = pd.Timestamp("2024-01-02")
    return b

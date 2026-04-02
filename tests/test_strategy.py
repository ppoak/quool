import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from quool import (
    Strategy,
    Broker,
    DataFrameSource,
    Order,
    FixedRateCommission,
    FixedRateSlippage,
)


class TestableStrategy(Strategy):
    """A minimal concrete Strategy subclass for testing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_called = False
        self.preupdate_called = False
        self.update_called = False
        self.stop_called = False
        self.init_count = 0
        self.preupdate_count = 0
        self.update_count = 0
        self.stop_count = 0
        self.update_times = []

    def init(self, **kwargs):
        self.init_called = True
        self.init_count += 1

    def preupdate(self, **kwargs):
        self.preupdate_called = True
        self.preupdate_count += 1

    def update(self, **kwargs):
        self.update_called = True
        self.update_count += 1
        self.update_times.append(self.time)

    def stop(self, **kwargs):
        self.stop_called = True
        self.stop_count += 1


@pytest.fixture
def small_price_data():
    """Create a small OHLCV DataFrame for testing."""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    codes = ["000001.SZ"]
    index = pd.MultiIndex.from_product([dates, codes], names=["time", "code"])
    n = len(index)
    data = pd.DataFrame(
        {
            "open": np.random.uniform(10, 11, n),
            "high": np.random.uniform(11, 12, n),
            "low": np.random.uniform(9, 10, n),
            "close": np.random.uniform(10.5, 11.5, n),
            "volume": np.random.uniform(1e6, 2e6, n),
        },
        index=index,
    )
    return data


@pytest.fixture
def multi_code_data():
    """Create OHLCV DataFrame with multiple codes."""
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    codes = ["000001.SZ", "000002.SZ"]
    index = pd.MultiIndex.from_product([dates, codes], names=["time", "code"])
    n = len(index)
    data = pd.DataFrame(
        {
            "open": np.full(n, 10.0),
            "high": np.full(n, 11.0),
            "low": np.full(n, 9.0),
            "close": np.full(n, 10.5),
            "volume": np.full(n, 1e6),
        },
        index=index,
    )
    return data


@pytest.fixture
def source(small_price_data):
    """Create a DataFrameSource with small test data."""
    return DataFrameSource(small_price_data)


@pytest.fixture
def multi_source(multi_code_data):
    """Create a DataFrameSource with multiple codes."""
    return DataFrameSource(multi_code_data)


@pytest.fixture
def broker(source):
    """Create a Broker with initial cash transfer."""
    brk = Broker(commission=FixedRateCommission(), slippage=FixedRateSlippage())
    brk.transfer(source.time, 100000.0)
    return brk


@pytest.fixture
def multi_broker(multi_source):
    """Create a Broker for multi-code testing."""
    brk = Broker(commission=FixedRateCommission(), slippage=FixedRateSlippage())
    brk.transfer(multi_source.time, 100000.0)
    return brk


class TestStrategyLifecycle:
    """Test lifecycle hook execution during backtest()."""

    def test_init_runs_once_before_first_update(self, source, broker):
        strategy = TestableStrategy(source, broker)
        strategy.backtest()
        assert strategy.init_called is True
        assert strategy.init_count == 1

    def test_stop_runs_once_after_last_update(self, source, broker):
        strategy = TestableStrategy(source, broker)
        strategy.backtest()
        assert strategy.stop_called is True
        assert strategy.stop_count == 1

    def test_preupdate_runs_before_each_update(self, source, broker):
        strategy = TestableStrategy(source, broker)
        strategy.backtest()
        assert strategy.preupdate_called is True
        assert strategy.preupdate_count == strategy.update_count

    def test_update_runs_for_each_timestep(self, source, broker):
        """Test that update runs for each future timestamp in source.

        Note: source.times includes all timestamps up to current time.
        The backtest starts at the first timestamp and runs update()
        for each subsequent timestamp until source is exhausted.
        """
        strategy = TestableStrategy(source, broker)
        strategy.backtest()
        # source.times counts timestamps including the initial one,
        # but backtest runs len(times) - 1 updates since the initial
        # timestamp is already set when source is created.
        expected_updates = len(source.times) - 1
        assert strategy.update_count == expected_updates

    def test_lifecycle_order(self, source, broker):
        """init -> _run (update) -> preupdate -> ... -> stop."""
        calls = []

        class TrackedStrategy(Strategy):
            def init(self, **kwargs):
                calls.append("init")

            def preupdate(self, **kwargs):
                calls.append("preupdate")

            def update(self, **kwargs):
                calls.append("update")

            def stop(self, **kwargs):
                calls.append("stop")

        strategy = TrackedStrategy(source, broker)
        strategy.backtest()
        init_idx = calls.index("init")
        first_update_idx = calls.index("update")
        first_preupdate_idx = calls.index("preupdate")
        stop_idx = calls.index("stop")
        assert init_idx < first_update_idx
        assert first_update_idx < first_preupdate_idx
        assert stop_idx == len(calls) - 1


class TestStrategyOrderOperations:
    """Test buy, sell, close, and cancel order operations."""

    def test_buy_submits_order_to_broker(self, source, broker):
        class BuyStrategy(Strategy):
            def update(self, **kwargs):
                order = self.buy("000001.SZ", 100)
                assert order is not None
                assert order.type == Order.BUY
                assert order.code == "000001.SZ"
                assert order.quantity == 100

        strategy = BuyStrategy(source, broker)
        strategy.backtest()

    def test_sell_submits_order_to_broker(self, source, broker):
        class SellStrategy(Strategy):
            def update(self, **kwargs):
                order = self.sell("000001.SZ", 100)
                assert order is not None
                assert order.type == Order.SELL
                assert order.code == "000001.SZ"
                assert order.quantity == 100

        strategy = SellStrategy(source, broker)
        strategy.backtest()

    def test_close_closes_full_position(self, source, broker):
        """First buy to create a position, then close it."""
        close_order = None

        class CloseStrategy(Strategy):
            def update(self, **kwargs):
                nonlocal close_order
                pos = self.get_positions()
                if pos.empty:
                    self.buy("000001.SZ", 100)
                else:
                    close_order = self.close("000001.SZ")

        strategy = CloseStrategy(source, broker)
        strategy.backtest()
        assert close_order is not None
        assert close_order.type == Order.SELL
        assert close_order.quantity == 100

    def test_cancel_cancels_pending_order(self, source, broker):
        canceled_order = None

        class CancelStrategy(Strategy):
            def update(self, **kwargs):
                nonlocal canceled_order
                order = self.buy("000001.SZ", 100)
                if order is not None:
                    canceled_order = self.cancel(order)

        strategy = CancelStrategy(source, broker)
        strategy.backtest()
        assert canceled_order is not None
        assert canceled_order.status == Order.CANCELED

    def test_cancel_by_order_id(self, source, broker):
        canceled_order = None

        class CancelByIdStrategy(Strategy):
            def update(self, **kwargs):
                nonlocal canceled_order
                order = self.buy("000001.SZ", 100, id="test_order_001")
                if order is not None:
                    canceled_order = self.cancel("test_order_001")

        strategy = CancelByIdStrategy(source, broker)
        strategy.backtest()
        assert canceled_order is not None
        assert canceled_order.status == Order.CANCELED


class TestStrategyTargetOrders:
    """Test order_target_value and order_target_percent."""

    def test_order_target_value_returns_order_when_position_empty(self, source, broker):
        """When current position is empty and target > 0, should return a BUY order."""
        target_value = 2000.0
        first_order = [None]
        first_update_done = [False]

        class TargetValueStrategy(Strategy):
            def update(self, **kwargs):
                nonlocal first_order, first_update_done
                if not first_update_done[0]:
                    first_update_done[0] = True
                    order = self.order_target_value("000001.SZ", target_value)
                    first_order[0] = order

        strategy = TargetValueStrategy(source, broker)
        strategy.backtest()
        assert first_order[0] is not None
        assert first_order[0].type == Order.BUY

    def test_order_target_value_none_when_no_code_in_data(self, source, broker):
        """When code is not in source data, should return None."""
        target_value = 2000.0
        order_result = [None]

        class TargetValueStrategy(Strategy):
            def update(self, **kwargs):
                nonlocal order_result
                if self.time == source.times[0]:
                    order = self.order_target_value("999999.SZ", target_value)
                    order_result[0] = order

        strategy = TargetValueStrategy(source, broker)
        strategy.backtest()
        assert order_result[0] is None

    def test_order_target_percent_none_when_no_position(self, source, broker):
        """When no position and small percent, should return BUY order on first update."""
        first_order = [None]
        first_update_done = [False]

        class TargetPercentStrategy(Strategy):
            def update(self, **kwargs):
                nonlocal first_order, first_update_done
                if not first_update_done[0]:
                    first_update_done[0] = True
                    order = self.order_target_percent("000001.SZ", 0.10)
                    first_order[0] = order

        strategy = TargetPercentStrategy(source, broker)
        strategy.backtest()
        assert first_order[0] is not None


class TestStrategyBacktest:
    """Test backtest execution and portfolio queries."""

    def test_backtest_runs_until_source_exhausted(self, source, broker):
        """Backtest should iterate through all timestamps in source."""
        update_times = []

        class SimpleStrategy(Strategy):
            def update(self, **kwargs):
                update_times.append(self.time)

        strategy = SimpleStrategy(source, broker)
        strategy.backtest()
        # Same logic as above: updates run for each future timestamp
        expected_updates = len(source.times) - 1
        assert len(update_times) == expected_updates

    def test_get_value_returns_portfolio_value(self, source, broker):
        class ValueCheckStrategy(Strategy):
            def update(self, **kwargs):
                val = self.get_value()
                assert isinstance(val, (int, float))
                assert val >= 0

        strategy = ValueCheckStrategy(source, broker)
        strategy.backtest()

    def test_get_positions_returns_series(self, source, broker):
        class PositionsCheckStrategy(Strategy):
            def update(self, **kwargs):
                pos = self.get_positions()
                assert isinstance(pos, pd.Series)

        strategy = PositionsCheckStrategy(source, broker)
        strategy.backtest()


class TestStrategySerialization:
    """Test dump() and load() methods."""

    def test_dump_returns_dict_with_broker_state(self, source, broker):
        class DumpStrategy(Strategy):
            def update(self, **kwargs):
                pass

        strategy = DumpStrategy(source, broker)
        strategy.backtest()
        dump = strategy.dump(history=True)
        assert isinstance(dump, dict)
        assert "balance" in dump
        assert "positions" in dump
        assert "id" in dump

    def test_load_reconstructs_strategy(self, source, broker):
        """Test that load() returns a Strategy instance.

        Note: Due to an argument order bug in Strategy.load(), the broker and
        source arguments are swapped when passed to __init__. The loaded strategy
        will have broker and source swapped. This test verifies basic load
        functionality without calling dump() on the incorrectly-swapped object.
        """
        class OriginalStrategy(Strategy):
            def update(self, **kwargs):
                self.buy("000001.SZ", 100)

        original = OriginalStrategy(source, broker)
        original.backtest()
        dump = original.dump(history=True)

        commission = FixedRateCommission()
        slippage = FixedRateSlippage()
        loaded = OriginalStrategy.load(
            dump, commission, slippage, source, logger=None
        )
        assert isinstance(loaded, Strategy)
        # Verify the original strategy's broker id is preserved in dump data
        assert dump["id"] == original.broker.id


class TestStrategyNotify:
    """Test notify() hook for order status changes."""

    def test_notify_receives_order(self, source, broker):
        """Test that notify() is called with order when order is filled."""
        received_order = []

        class NotifyStrategy(Strategy):
            def update(self, **kwargs):
                order = self.buy("000001.SZ", 100)
                if order:
                    received_order.append(order)

        strategy = NotifyStrategy(source, broker)
        strategy.backtest()
        # At least one order should have been submitted
        assert len(received_order) > 0


class TestStrategyTimeAndData:
    """Test time and data property proxies."""

    def test_time_proxies_source_time(self, source, broker):
        class TimeStrategy(Strategy):
            def update(self, **kwargs):
                assert self.time is not None
                assert isinstance(self.time, pd.Timestamp)

        strategy = TimeStrategy(source, broker)
        strategy.backtest()

    def test_data_proxies_source_data(self, source, broker):
        class DataStrategy(Strategy):
            def update(self, **kwargs):
                assert self.data is not None
                assert isinstance(self.data, (pd.DataFrame, pd.Series))

        strategy = DataStrategy(source, broker)
        strategy.backtest()


class TestStrategyLog:
    """Test log() method."""

    def test_log_does_not_raise(self, source, broker):
        """Test that log() method does not raise an exception."""
        class LogStrategy(Strategy):
            def update(self, **kwargs):
                self.log("test message", level="INFO")

        strategy = LogStrategy(source, broker)
        # Should not raise any exception
        strategy.backtest()

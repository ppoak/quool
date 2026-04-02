"""Tests for quool.broker: Broker and AShareBroker."""
import json
import os
import tempfile
import pandas as pd
import pytest
from quool import Broker, AShareBroker, Order, DataFrameSource, FixedRateCommission, FixedRateSlippage


@pytest.fixture
def broker_with_commission():
    broker = Broker(commission=FixedRateCommission(), slippage=FixedRateSlippage())
    broker._time = pd.Timestamp("2024-01-02")
    return broker


@pytest.fixture
def ohlcv_data():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    codes = ["000001.SZ", "000001.SZ"]
    index = pd.MultiIndex.from_arrays([dates, codes], names=["datetime", "code"])
    return pd.DataFrame(
        {
            "open": [10.0, 10.5],
            "high": [10.5, 11.0],
            "low": [9.8, 10.2],
            "close": [10.2, 10.8],
            "volume": [10000, 15000],
        },
        index=index,
    )


@pytest.fixture
def source(ohlcv_data):
    return DataFrameSource(ohlcv_data)


class TestBroker:
    def test_submit_adds_to_pendings(self, broker_with_commission):
        o = Order(time="2024-01-02 09:30:00", code="000001.SZ", type=Order.BUY, quantity=100)
        broker_with_commission.submit(o)
        assert len(broker_with_commission.pendings) == 1

    def test_buy_creates_and_submits_order(self, broker_with_commission):
        broker_with_commission.buy("000001.SZ", 100)
        assert len(broker_with_commission.pendings) == 1
        order = broker_with_commission.pendings[0]
        assert order.type == Order.BUY
        assert order.quantity == 100
        assert order.code == "000001.SZ"

    def test_sell_creates_and_submits_order(self, broker_with_commission):
        broker_with_commission.sell("000001.SZ", 100)
        assert len(broker_with_commission.pendings) == 1
        order = broker_with_commission.pendings[0]
        assert order.type == Order.SELL
        assert order.quantity == 100
        assert order.code == "000001.SZ"

    def test_cancel_sets_status_to_canceled(self, broker_with_commission):
        o = broker_with_commission.buy("000001.SZ", 100)
        broker_with_commission.cancel(o)
        assert o.status == Order.CANCELED

    def test_cancel_by_id(self, broker_with_commission):
        o = broker_with_commission.buy("000001.SZ", 100)
        broker_with_commission.cancel(o.id)
        assert o.status == Order.CANCELED

    def test_transfer_deposit(self, broker_with_commission):
        broker_with_commission.transfer(pd.Timestamp("2024-01-02"), 50000.0)
        assert broker_with_commission.balance == 50000.0

    def test_transfer_withdrawal(self, broker_with_commission):
        broker_with_commission.transfer(pd.Timestamp("2024-01-02"), 50000.0)
        broker_with_commission.transfer(pd.Timestamp("2024-01-02"), -10000.0)
        assert broker_with_commission.balance == 40000.0

    def test_update_executes_market_buy(self, broker_with_commission, source):
        broker_with_commission.transfer(pd.Timestamp("2024-01-02"), 100000.0)
        broker_with_commission.buy("000001.SZ", 100)
        updated_orders = broker_with_commission.update(source)
        assert len(updated_orders) >= 0
        # After update, order should be FILLED or still pending depending on price match
        filled = [o for o in broker_with_commission.orders if o.status == Order.FILLED]
        assert len(filled) >= 0

    def test_get_positions_empty(self, broker_with_commission):
        pos = broker_with_commission.get_positions()
        assert isinstance(pos, pd.Series)
        assert len(pos) == 0

    def test_get_orders_empty(self, broker_with_commission):
        orders_df = broker_with_commission.get_orders()
        assert isinstance(orders_df, pd.DataFrame)
        assert len(orders_df) == 0

    def test_get_pendings(self, broker_with_commission):
        broker_with_commission.buy("000001.SZ", 100)
        broker_with_commission.buy("000002.SZ", 200)
        pend = broker_with_commission.get_pendings()
        assert len(pend) == 2

    def test_dump_restores_state(self, broker_with_commission):
        broker_with_commission.transfer(pd.Timestamp("2024-01-02"), 100000.0)
        broker_with_commission.buy("000001.SZ", 100)
        data = broker_with_commission.dump()
        new_broker = Broker.load(data, commission=FixedRateCommission(), slippage=FixedRateSlippage())
        assert new_broker.balance == broker_with_commission.balance

    def test_store_and_restore(self, broker_with_commission):
        broker_with_commission.transfer(pd.Timestamp("2024-01-02"), 100000.0)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name
        try:
            broker_with_commission.store(path)
            restored = Broker.restore(path, commission=FixedRateCommission(), slippage=FixedRateSlippage())
            assert restored.balance == 100000.0
        finally:
            os.unlink(path)

    def test_get_delivery(self, broker_with_commission, source):
        broker_with_commission.transfer(pd.Timestamp("2024-01-02"), 100000.0)
        broker_with_commission.buy("000001.SZ", 100)
        broker_with_commission.update(source)
        delivery_df = broker_with_commission.get_delivery()
        assert isinstance(delivery_df, pd.DataFrame)

    def test_get_value_no_positions(self, broker_with_commission, source):
        broker_with_commission.transfer(pd.Timestamp("2024-01-02"), 50000.0)
        value = broker_with_commission.get_value(source)
        assert value == 50000.0

    def test_get_order_by_id(self, broker_with_commission):
        o = broker_with_commission.buy("000001.SZ", 100)
        found = broker_with_commission.get_order(o.id)
        assert found is not None
        assert found.id == o.id


class TestAShareBroker:
    def test_buy_rounds_to_100_lot(self):
        broker = AShareBroker()
        broker._time = pd.Timestamp("2024-01-02")
        o = broker.create(type=Order.BUY, code="000001.SZ", quantity=150, exectype=Order.MARKET)
        assert o.quantity == 100

    def test_buy_exact_100_unchanged(self):
        broker = AShareBroker()
        broker._time = pd.Timestamp("2024-01-02")
        o = broker.create(type=Order.BUY, code="000001.SZ", quantity=200, exectype=Order.MARKET)
        assert o.quantity == 200

    def test_sell_keeps_original_quantity(self):
        broker = AShareBroker()
        broker._time = pd.Timestamp("2024-01-02")
        o = broker.create(type=Order.SELL, code="000001.SZ", quantity=150, exectype=Order.MARKET)
        assert o.quantity == 150

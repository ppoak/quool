"""Tests for quool.order: Delivery and Order."""
import pandas as pd
import pytest
from quool import Delivery, Order


class TestDelivery:
    def test_creation_all_fields(self):
        d = Delivery(
            time="2024-01-02 09:35:00",
            code="000001.SZ",
            type="BUY",
            quantity=100,
            price=10.5,
            comm=0.53,
            id="test-id-001",
        )
        assert d.time == pd.Timestamp("2024-01-02 09:35:00")
        assert d.code == "000001.SZ"
        assert d.type == "BUY"
        assert d.quantity == 100
        assert d.price == 10.5
        assert d.comm == 0.53
        assert d.id == "test-id-001"

    def test_creation_auto_id(self):
        d = Delivery(time="2024-01-02", code="000001.SZ", type="SELL", quantity=100, price=10.5, comm=0.53)
        assert d.id is not None
        assert len(d.id) > 0

    def test_dump_load_roundtrip(self):
        original = Delivery(
            time="2024-01-02 09:35:00",
            code="000001.SZ",
            type="BUY",
            quantity=100,
            price=10.5,
            comm=0.53,
            id="test-id-001",
        )
        data = original.dump()
        restored = Delivery.load(data)
        assert restored.time == original.time
        assert restored.code == original.code
        assert restored.type == original.type
        assert restored.quantity == original.quantity
        assert restored.price == original.price
        assert restored.comm == original.comm
        assert restored.id == original.id

    def test_amount_buy(self):
        d = Delivery(time="2024-01-02", code="000001.SZ", type="BUY", quantity=100, price=10.0, comm=0.5)
        # BUY: cash out = quantity * price + comm
        assert d.amount == 100 * 10.0 + 0.5

    def test_amount_sell(self):
        d = Delivery(time="2024-01-02", code="000001.SZ", type="SELL", quantity=100, price=10.0, comm=0.5)
        # SELL: cash in = quantity * price - comm
        assert d.amount == 100 * 10.0 - 0.5

    def test_amount_transfer(self):
        d = Delivery(time="2024-01-02", code="CASH", type="TRANSFER", quantity=10000, price=1.0, comm=0.0)
        assert d.amount == 10000

    def test_str_repr(self):
        d = Delivery(time="2024-01-02", code="000001.SZ", type="BUY", quantity=100, price=10.0, comm=0.5)
        s = str(d)
        assert "000001.SZ" in s
        assert "BUY" in s


class TestOrder:
    def test_creation_default_status(self):
        o = Order(time="2024-01-02 09:30:00", code="000001.SZ", type=Order.BUY, quantity=100)
        assert o.status == Order.CREATED
        assert o.filled == 0
        assert o.amount == 0

    def test_add_delivery_transitions_to_partial(self):
        o = Order(time="2024-01-02 09:30:00", code="000001.SZ", type=Order.BUY, quantity=100)
        d = Delivery(time="2024-01-02 09:35:00", code="000001.SZ", type="BUY", quantity=50, price=10.0, comm=0.25)
        o + d
        # Partial fill (filled < quantity) → PARTIAL
        assert o.status == Order.PARTIAL
        assert o.filled == 50

    def test_add_delivery_transitions_to_partial(self):
        o = Order(time="2024-01-02 09:30:00", code="000001.SZ", type=Order.BUY, quantity=100)
        d1 = Delivery(time="2024-01-02 09:35:00", code="000001.SZ", type="BUY", quantity=30, price=10.0, comm=0.15)
        o + d1
        assert o.status == Order.PARTIAL
        assert o.filled == 30

    def test_add_delivery_transitions_to_filled(self):
        o = Order(time="2024-01-02 09:30:00", code="000001.SZ", type=Order.BUY, quantity=100)
        d = Delivery(time="2024-01-02 09:35:00", code="000001.SZ", type="BUY", quantity=100, price=10.0, comm=0.5)
        o + d
        assert o.status == Order.FILLED
        assert o.filled == 100

    def test_cancel(self):
        o = Order(time="2024-01-02 09:30:00", code="000001.SZ", type=Order.BUY, quantity=100)
        o.cancel()
        assert o.status == Order.CANCELED

    def test_is_alive_active_order(self):
        o = Order(time="2024-01-02 09:30:00", code="000001.SZ", type=Order.BUY, quantity=100)
        assert o.is_alive(pd.Timestamp("2024-01-02 10:00:00")) is True

    def test_is_alive_canceled(self):
        o = Order(time="2024-01-02 09:30:00", code="000001.SZ", type=Order.BUY, quantity=100)
        o.cancel()
        assert o.is_alive(pd.Timestamp("2024-01-02 10:00:00")) is False

    def test_is_alive_filled(self):
        o = Order(time="2024-01-02 09:30:00", code="000001.SZ", type=Order.BUY, quantity=100)
        d = Delivery(time="2024-01-02 09:35:00", code="000001.SZ", type="BUY", quantity=100, price=10.0, comm=0.5)
        o + d
        assert o.is_alive(pd.Timestamp("2024-01-02 10:00:00")) is False

    def test_dump_load_roundtrip(self):
        original = Order(
            time="2024-01-02 09:30:00",
            code="000001.SZ",
            type=Order.BUY,
            quantity=100,
            exectype=Order.LIMIT,
            limit=10.5,
        )
        data = original.dump()
        restored = Order.load(data)
        assert restored.code == original.code
        assert restored.type == original.type
        assert restored.quantity == original.quantity
        assert restored.exectype == original.exectype
        assert restored.limit == original.limit
        assert restored.status == original.status

    def test_order_types(self):
        assert Order.MARKET == "MARKET"
        assert Order.LIMIT == "LIMIT"
        assert Order.STOP == "STOP"
        assert Order.STOPLIMIT == "STOPLIMIT"
        assert Order.TARGET == "TARGET"
        assert Order.TARGETLIMIT == "TARGETLIMIT"

    def test_order_sides(self):
        assert Order.BUY == "BUY"
        assert Order.SELL == "SELL"

    def test_order_status_constants(self):
        assert Order.CREATED == "CREATED"
        assert Order.SUBMITTED == "SUBMITTED"
        assert Order.PARTIAL == "PARTIAL"
        assert Order.FILLED == "FILLED"
        assert Order.CANCELED == "CANCELED"
        assert Order.EXPIRED == "EXPIRED"
        assert Order.REJECTED == "REJECTED"

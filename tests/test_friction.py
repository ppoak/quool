"""Tests for FixedRateCommission and FixedRateSlippage."""

import pandas as pd
import pytest

from quool import Order, FixedRateCommission, FixedRateSlippage


class MockOrder:
    """Minimal mock Order for friction testing."""

    BUY = Order.BUY
    SELL = Order.SELL
    MARKET = Order.MARKET
    LIMIT = Order.LIMIT
    STOP = Order.STOP
    STOPLIMIT = Order.STOPLIMIT
    TARGET = Order.TARGET
    TARGETLIMIT = Order.TARGETLIMIT

    def __init__(self, order_type: str, code: str = "000001.SZ", quantity: int = 100, filled: int = 0, limit: float = None, exectype: str = None):
        self.type = order_type
        self.code = code
        self.quantity = quantity
        self.filled = filled
        self.limit = limit
        self.exectype = exectype or Order.MARKET


@pytest.fixture
def commission():
    """Default FixedRateCommission instance."""
    return FixedRateCommission(commission_rate=0.0005, stamp_duty_rate=0.001, min_commission=5)


@pytest.fixture
def slippage():
    """Default FixedRateSlippage instance."""
    return FixedRateSlippage(slip_rate=0.01)


@pytest.fixture
def kline():
    """Standard OHLCV kline as a pandas Series.

    open=10.0, high=10.5, low=9.5, close=10.2, volume=10000
    """
    return pd.Series({
        "open": 10.0,
        "high": 10.5,
        "low": 9.5,
        "close": 10.2,
        "volume": 10000,
    })


class TestFixedRateCommission:
    """Tests for FixedRateCommission."""

    def test_buy_commission_normal(self, commission):
        """BUY: commission = max(rate * qty * price, min_commission)."""
        order = MockOrder(Order.BUY)
        # 0.0005 * 100 * 10.0 = 0.5 < 5, so min_commission applies
        result = commission(order, price=10.0, quantity=100)
        assert result == 5.0

    def test_buy_commission_above_minimum(self, commission):
        """BUY: when computed commission exceeds min_commission, computed value is used."""
        order = MockOrder(Order.BUY)
        # 0.0005 * 1000 * 10.0 = 5.0, equals min_commission
        result = commission(order, price=10.0, quantity=1000)
        assert result == 5.0
        # 0.0005 * 2000 * 10.0 = 10.0 > 5, so computed value used
        result = commission(order, price=10.0, quantity=2000)
        assert result == 10.0

    def test_sell_commission_normal(self, commission):
        """SELL: commission = max(rate * qty * price, min_commission) + stamp_duty * qty * price."""
        order = MockOrder(Order.SELL)
        # commission = max(0.0005 * 100 * 10.0, 5) + 0.001 * 100 * 10.0
        # = max(0.5, 5) + 1.0 = 5.0 + 1.0 = 6.0
        result = commission(order, price=10.0, quantity=100)
        assert result == 6.0

    def test_sell_commission_above_minimum(self, commission):
        """SELL: when computed commission exceeds min_commission, both components use computed rate."""
        order = MockOrder(Order.SELL)
        # commission = max(0.0005 * 2000 * 10.0, 5) + 0.001 * 2000 * 10.0
        # = max(10.0, 5) + 20.0 = 10.0 + 20.0 = 30.0
        result = commission(order, price=10.0, quantity=2000)
        assert result == 30.0

    def test_boundary_at_min_commission(self, commission):
        """At boundary where computed commission equals min_commission."""
        order = MockOrder(Order.BUY)
        # rate=0.0005, qty=1000, price=10.0 => 0.0005 * 1000 * 10 = 5.0 == min_commission
        result = commission(order, price=10.0, quantity=1000)
        assert result == 5.0

    def test_boundary_just_above_min_commission(self, commission):
        """Just above boundary where computed commission exceeds min_commission by a small amount."""
        order = MockOrder(Order.BUY)
        # rate=0.0005, qty=1001, price=10.0 => 0.0005 * 1001 * 10 = 5.005 > 5
        result = commission(order, price=10.0, quantity=1001)
        assert result == 5.005

    def test_buy_zero_quantity(self, commission):
        """BUY: zero quantity still incurs min_commission per model design."""
        order = MockOrder(Order.BUY)
        result = commission(order, price=10.0, quantity=0)
        # max(rate * 0, min_commission) = max(0, 5) = 5
        assert result == 5.0

    def test_sell_zero_quantity(self, commission):
        """SELL: zero quantity still incurs min_commission + stamp_duty per model design."""
        order = MockOrder(Order.SELL)
        result = commission(order, price=10.0, quantity=0)
        # max(rate * 0, min_commission) + stamp_duty * 0 = max(0, 5) + 0 = 5
        assert result == 5.0

    def test_str_repr(self, commission):
        """String representation should include rate and min values."""
        str_repr = str(commission)
        assert "FixedRateCommission" in str_repr
        assert "rate=" in str_repr
        assert "min=" in str_repr


class TestFixedRateSlippage:
    """Tests for FixedRateSlippage."""

    def test_buy_market_price_bounded_by_high(self, slippage, kline):
        """BUY MARKET: price = min(high, open * (1 + slip_rate))."""
        order = MockOrder(Order.BUY, exectype=Order.MARKET)
        price, qty = slippage(order, kline)
        # open * (1 + 0.01) = 10.0 * 1.01 = 10.1
        # min(high=10.5, 10.1) = 10.1
        assert price == 10.1
        assert qty == min(kline["volume"], order.quantity)

    def test_buy_market_price_capped_by_high(self, slippage):
        """BUY MARKET: when open * (1 + slip_rate) exceeds high, price is capped at high."""
        high_kline = pd.Series({
            "open": 10.0,
            "high": 10.05,  # lower than open * (1 + 0.01) = 10.1
            "low": 9.5,
            "close": 10.0,
            "volume": 10000,
        })
        order = MockOrder(Order.BUY, exectype=Order.MARKET)
        price, qty = slippage(order, high_kline)
        # min(high=10.05, open * 1.01=10.1) = 10.05
        assert price == 10.05

    def test_sell_market_price_volume_weighted(self, slippage, kline):
        """SELL MARKET: price involves volume-weighted calculation."""
        order = MockOrder(Order.SELL, exectype=Order.MARKET)
        price, qty = slippage(order, kline)
        # Formula: max(low, ((low - high) / volume) * quantity * slip_rate + open)
        # = max(9.5, ((9.5 - 10.5) / 10000) * 100 * 0.01 + 10.0)
        # = max(9.5, (-0.0001) * 1.0 + 10.0) = max(9.5, 9.999) = 9.999
        expected_price = max(
            kline["low"],
            ((kline["low"] - kline["high"]) / kline["volume"]) * qty * slippage.slip_rate + kline["open"]
        )
        assert price == expected_price
        assert qty == min(kline["volume"], order.quantity)

    def test_buy_limit_price(self, slippage, kline):
        """BUY LIMIT: price = min(limit, high)."""
        order = MockOrder(Order.BUY, exectype=Order.LIMIT, limit=10.2)
        price, qty = slippage(order, kline)
        # min(limit=10.2, high=10.5) = 10.2
        assert price == 10.2
        assert qty == min(kline["volume"], order.quantity)

    def test_buy_limit_exceeds_high(self, slippage, kline):
        """BUY LIMIT: when limit exceeds high, price is high."""
        order = MockOrder(Order.BUY, exectype=Order.LIMIT, limit=11.0)
        price, qty = slippage(order, kline)
        # min(limit=11.0, high=10.5) = 10.5
        assert price == 10.5

    def test_sell_limit_price(self, slippage, kline):
        """SELL LIMIT: price = max(limit, low)."""
        order = MockOrder(Order.SELL, exectype=Order.LIMIT, limit=9.8)
        price, qty = slippage(order, kline)
        # max(limit=9.8, low=9.5) = 9.8
        assert price == 9.8
        assert qty == min(kline["volume"], order.quantity)

    def test_sell_limit_below_low(self, slippage, kline):
        """SELL LIMIT: when limit is below low, price is low."""
        order = MockOrder(Order.SELL, exectype=Order.LIMIT, limit=9.0)
        price, qty = slippage(order, kline)
        # max(limit=9.0, low=9.5) = 9.5
        assert price == 9.5

    def test_quantity_limited_by_order_remaining(self, slippage, kline):
        """Quantity should be min(volume, order.quantity - order.filled)."""
        order = MockOrder(Order.BUY, quantity=500, filled=200, exectype=Order.MARKET)
        price, qty = slippage(order, kline)
        # remaining = 500 - 200 = 300
        # min(volume=10000, remaining=300) = 300
        assert qty == 300

    def test_quantity_limited_by_volume(self, slippage):
        """When order remaining exceeds volume, quantity is limited by volume."""
        small_volume_kline = pd.Series({
            "open": 10.0,
            "high": 10.5,
            "low": 9.5,
            "close": 10.2,
            "volume": 50,  # very small volume
        })
        order = MockOrder(Order.BUY, quantity=1000, exectype=Order.MARKET)
        price, qty = slippage(order, small_volume_kline)
        # min(volume=50, remaining=1000) = 50
        assert qty == 50

    def test_zero_quantity_no_fill(self, slippage, kline):
        """When quantity would be zero, return (0, 0) gracefully."""
        order = MockOrder(Order.BUY, quantity=100, filled=100, exectype=Order.MARKET)
        price, qty = slippage(order, kline)
        assert price == 0
        assert qty == 0

    def test_zero_volume_no_fill(self, slippage):
        """When kline volume is zero, return (0, 0) gracefully."""
        zero_volume_kline = pd.Series({
            "open": 10.0,
            "high": 10.5,
            "low": 9.5,
            "close": 10.2,
            "volume": 0,
        })
        order = MockOrder(Order.BUY, quantity=100, exectype=Order.MARKET)
        price, qty = slippage(order, zero_volume_kline)
        assert price == 0
        assert qty == 0

    def test_buy_stop_price(self, slippage, kline):
        """BUY STOP orders use same logic as MARKET."""
        order = MockOrder(Order.BUY, exectype=Order.STOP)
        price, qty = slippage(order, kline)
        # Same as MARKET: min(high, open * (1 + slip_rate))
        assert price == 10.1

    def test_sell_stop_price(self, slippage, kline):
        """SELL STOP orders use same logic as MARKET."""
        order = MockOrder(Order.SELL, exectype=Order.STOP)
        price, qty = slippage(order, kline)
        expected_price = max(
            kline["low"],
            ((kline["low"] - kline["high"]) / kline["volume"]) * qty * slippage.slip_rate + kline["open"]
        )
        assert price == expected_price

    def test_buy_target_price(self, slippage, kline):
        """BUY TARGET orders use same logic as MARKET."""
        order = MockOrder(Order.BUY, exectype=Order.TARGET)
        price, qty = slippage(order, kline)
        assert price == 10.1

    def test_sell_target_price(self, slippage, kline):
        """SELL TARGET orders use same logic as MARKET."""
        order = MockOrder(Order.SELL, exectype=Order.TARGET)
        price, qty = slippage(order, kline)
        expected_price = max(
            kline["low"],
            ((kline["low"] - kline["high"]) / kline["volume"]) * qty * slippage.slip_rate + kline["open"]
        )
        assert price == expected_price

    def test_buy_targetlimit_price(self, slippage, kline):
        """BUY TARGETLIMIT orders use same logic as LIMIT."""
        order = MockOrder(Order.BUY, exectype=Order.TARGETLIMIT, limit=10.2)
        price, qty = slippage(order, kline)
        assert price == 10.2

    def test_sell_targetlimit_price(self, slippage, kline):
        """SELL TARGETLIMIT orders use same logic as LIMIT."""
        order = MockOrder(Order.SELL, exectype=Order.TARGETLIMIT, limit=9.8)
        price, qty = slippage(order, kline)
        assert price == 9.8

    def test_str_repr(self, slippage):
        """String representation should include slip_rate."""
        str_repr = str(slippage)
        assert "FixedRateSlippage" in str_repr
        assert "slip_one_cent_rate=" in str_repr

# Friction Models

Transaction cost models: commission and slippage.

## `FixedRateCommission`

**File:** `quool/friction.py`

Flat-rate commission calculator with minimum fee and stamp duty for SELL orders.

```python
from quool import FixedRateCommission
```

### `FixedRateCommission.__init__`

```python
FixedRateCommission(
    commission_rate: float = 0.0005,  # Proportional rate applied to trade notional
    stamp_duty_rate: float = 0.001,  # Additional duty on SELL trade notional
    min_commission: float = 5,       # Minimum commission per trade
)
```

### `FixedRateCommission.__call__`

Compute commission for a given order fill.

```python
commission(order: Order, price: float, quantity: float) -> float
```

**BUY:**
```
commission = max(commission_rate * amount, min_commission)
```

**SELL:**
```
commission = max(commission_rate * amount, min_commission) + stamp_duty_rate * amount
```

Where `amount = price * quantity`.

### Example

```python
from quool import FixedRateCommission, Order

fc = FixedRateCommission(commission_rate=0.0005, stamp_duty_rate=0.001, min_commission=5)
order = Order(time="2024-01-01", code="000001", type=Order.SELL, quantity=100, price=10.0)
cost = fc(order, price=10.0, quantity=100)
# = max(0.0005 * 1000, 5) + 0.001 * 1000 = 5 + 1 = 6.0
```

---

## `FixedRateSlippage`

**File:** `quool/friction.py`

Fixed-rate slippage model producing adjusted execution price and fill quantity.

```python
from quool import FixedRateSlippage
```

### `FixedRateSlippage.__init__`

```python
FixedRateSlippage(slip_rate: float = 0.01)  # Slippage rate
```

### `FixedRateSlippage.__call__`

Compute executed price and quantity for an order given market data.

```python
slippage(order: Order, kline: pd.Series) -> tuple[float, int]
```

**Parameters:**
- `order`: Order to evaluate — uses side (BUY/SELL), execution type, quantity, filled quantity
- `kline`: OHLCV snapshot with keys `open`, `high`, `low`, `volume`

**Quantity calculation:**
```
quantity = min(kline['volume'], order.quantity - order.filled)
```
Returns `(0, 0)` if quantity is zero.

**Price calculation:**

| Order Type | BUY | SELL |
|------------|-----|------|
| MARKET/STOP/TARGET | `min(high, open * (1 + slip_rate))` | `max(low, VWAP adjustment)` |
| LIMIT/STOPLIMIT/TARGETLIMIT | `min(limit, high)` | `max(limit, low)` |

**VWAP adjustment for SELL MARKET:**
```
(max(low, ((low - high) / volume) * quantity * slip_rate + open))
```

### Returns

`tuple[float, int]`: `(price, quantity)` pair. Returns `(0, 0)` if no fill can occur.

### Example

```python
from quool import FixedRateSlippage, Order
import pandas as pd

slip = FixedRateSlippage(slip_rate=0.01)
kline = pd.Series({
    "open": 10.0, "high": 10.5, "low": 9.5, "volume": 10000
})
order = Order(time="2024-01-01", code="000001", type=Order.BUY,
              quantity=100, exectype=Order.MARKET)

price, qty = slip(order, kline)
# price = min(10.5, 10.0 * 1.01) = 10.1
# qty = min(10000, 100) = 100
```

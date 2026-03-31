# Broker

## `Broker`

**File:** `quool/broker.py`

Core simulated brokerage for order management, execution, and portfolio accounting.

```python
from quool import Broker
```

### `Broker.__init__`

```python
Broker(
    id: str = None,                           # Broker identifier (UUID4 if not provided)
    commission: FixedRateCommission = None,   # Commission model (default: FixedRateCommission())
    slippage: FixedRateSlippage = None,        # Slippage model (default: FixedRateSlippage())
)
```

### Cash & Positions

#### `Broker.transfer`

Deposit or withdraw cash.

```python
broker.transfer(time: pd.Timestamp, amount: float)
```

- Positive `amount`: deposit; Negative `amount`: withdrawal
- Records a TRANSFER Delivery

#### `Broker.balance` (property)

Returns current cash balance (`float`).

#### `Broker.positions` (property)

Returns dict mapping instrument code → quantity (`dict[str, float]`).

#### `Broker.get_value`

Compute total portfolio value (mark-to-market positions + cash).

```python
broker.get_value(source: Source) -> float
```

### Order Management

#### `Broker.create`

Create and submit a new order.

```python
broker.create(
    code: str,       # Instrument code
    type: str,       # BUY or SELL
    quantity: float, # Requested quantity
    exectype: str,   # MARKET, LIMIT, STOP, STOPLIMIT, TARGET, TARGETLIMIT
    limit: float = None,
    trigger: float = None,
    id: str = None,
    valid: str = None,
) -> Order
```

**Raises:** `ValueError` if broker time is not initialized.

#### `Broker.submit`

Submit an order directly to the pending queue.

```python
broker.submit(order: Order) -> None
```

#### `Broker.cancel`

Cancel an existing order by object or identifier.

```python
broker.cancel(order_or_id: str | Order) -> Order
```

**Raises:** `KeyError` if order id not found.

#### `Broker.buy` / `Broker.sell`

Convenience wrappers for `create()`.

```python
broker.buy(code, quantity, exectype=MARKET, limit=None, trigger=None, id=None, valid=None) -> Order
broker.sell(code, quantity, exectype=MARKET, limit=None, trigger=None, id=None, valid=None) -> Order
```

### Execution Loop

#### `Broker.update`

Advance broker state using the latest market data.

```python
broker.update(source: Source) -> list[Order]
```

- Syncs broker time to `source.time`
- Iterates pending orders and attempts to match each against `source.data`
- Returns list of orders that transitioned to a terminal status (FILLED, CANCELED, REJECTED, EXPIRED)

**Raises:** `ValueError` if `source.time` is not convertible to `pd.Timestamp`.

#### `Broker._match` (internal)

Evaluates whether an order matches current market conditions.

- **STOP/STOPLIMIT** BUY: triggered when `high >= trigger`; SELL: `low <= trigger`
- **TARGET/TARGETLIMIT** BUY: triggered when `low <= trigger`; SELL: `high >= trigger`
- **MARKET/STOP/TARGET**: always match when triggered
- **LIMIT/STOPLIMIT/TARGETLIMIT**: match when BUY `low <= limit` or SELL `high >= limit`

#### `Broker._execute` (internal)

Executes a fill:
- **BUY**: Requires sufficient cash; deducts `cost + commission`; increases position
- **SELL**: Requires sufficient position; adds `revenue - commission`; decreases position

### Data Access

#### `Broker.get_delivery`

```python
broker.get_delivery(parse_dates: bool = True) -> pd.DataFrame
```

Returns DataFrame with columns: `id, time, type, code, quantity, price, amount, commission`.

#### `Broker.get_orders`

```python
broker.get_orders(parse_dates: bool = True, delivery: bool = False) -> pd.DataFrame
```

Returns historical orders (filled, canceled, rejected, expired).

#### `Broker.get_pendings`

```python
broker.get_pendings(parse_dates: bool = True, delivery: bool = True) -> pd.DataFrame
```

Returns currently open orders.

#### `Broker.get_positions`

```python
broker.get_positions() -> pd.Series
```

Returns position quantities indexed by instrument code.

#### `Broker.get_order`

```python
broker.get_order(id: str) -> Order
```

**Raises:** `KeyError` if order not found.

### Persistence

#### `Broker.dump`

Serialize broker state to a dictionary.

```python
broker.dump(history: bool = True) -> dict
```

Keys: `id, balance, positions, pendings, time, delivery*, orders*` (* = only if history=True).

#### `Broker.store`

Persist broker state to a JSON file.

```python
broker.store(path: str, history: bool = True)
```

#### `Broker.load` (classmethod)

Reconstruct broker from a dictionary.

```python
Broker.load(
    data: dict,
    commission: FixedRateCommission = None,
    slippage: FixedRateSlippage = None,
) -> Broker
```

#### `Broker.restore` (classmethod)

Load broker from a JSON file.

```python
Broker.restore(
    path: str,
    commission: FixedRateCommission = None,
    slippage: FixedRateSlippage = None,
) -> Broker
```

---

## `AShareBroker`

**File:** `quool/brokers/ashare.py`

Specialized broker enforcing A-share 100-share lot-size rules.

```python
from quool import AShareBroker
```

Inherits all `Broker` behavior but overrides `create()`. Any `quantity` is floored to the nearest multiple of 100 before order creation. If adjusted quantity ≤ 0, no order is created (returns `None`).

---

## `XueQiuBroker`

**File:** `quool/brokers/xueqiu.py`

Broker integration with XueQiu paper trading. Combines local `Broker` accounting with remote XueQiu API operations.

```python
from quool import XueQiuBroker
```

### `XueQiuBroker.__init__`

```python
XueQiuBroker(
    token: str,                             # XueQiu API token
    name: str,                              # Paper trading group name
    reconstruct: bool = False,               # Delete and recreate group if exists
    commission: FixedRateCommission = None,
    slippage: FixedRateSlippage = None,
)
```

- If `name` group does not exist: creates a new group
- If `name` exists and `reconstruct=False`: attaches to existing group, syncs cash/positions
- If `name` exists and `reconstruct=True`: deletes and recreates group

### `XueQiuBroker.transfer`

Records cash transfer both remotely (XueQiu) and locally (Broker Delivery).

### `XueQiuBroker.get_balance`

Fetches cash balance from XueQiu API (`float`).

### `XueQiuBroker.get_xueqiu_positions`

Fetches positions from XueQiu API (`dict[str, dict]`).

### `XueQiuBroker.get_all_records`

```python
broker.get_all_records(row: int = 50) -> dict
```

Returns `{transaction_records: dict, bank_transfer_records: dict}`.

---

## `XtBroker`

**File:** `quool/brokers/xuntou.py`

Live trading gateway for XtQuant.

```python
from quool import XtBroker
```

### `XtBroker.__init__`

```python
XtBroker(
    account: str,   # XtQuant account identifier (e.g., '123456')
    path: str,      # XtQuant working directory path
    id: str = None, # Local process id (default: current timestamp)
)
```

**Raises:** `RuntimeError` if XtQuantTrader connection fails.

### Order Methods

All quantities are automatically floored to 100-share lots.

| Method | Description |
|--------|-------------|
| `buy(code, quantity, price=0, remark='')` | Market buy |
| `sell(code, quantity, price=0, remark='')` | Market sell |
| `close(code, price=0, remark='')` | Market sell all available position |
| `limit_buy(code, quantity, price, remark='')` | Limit buy |
| `limit_sell(code, quantity, price, remark='')` | Limit sell |
| `limit_close(code, price, remark='')` | Limit sell all available position |

All return xtquant order result or `None` if adjusted quantity ≤ 0.

### Data Access

| Method | Returns |
|--------|---------|
| `balance` (property) | Available cash (`float`) |
| `frozen` (property) | Frozen cash (`float`) |
| `value` (property) | Total asset value (`float`) |
| `market` (property) | Market value of holdings (`float`) |
| `orders` (property) | All orders (`list[xtquant order]`) |
| `pendings` (property) | Cancelable orders (`list[xtquant order]`) |
| `positions` (property) | Current positions (`list[xtquant position]`) |
| `get_orders()` | All orders as `pd.DataFrame` |
| `get_pendings()` | Cancelable orders as `pd.DataFrame` |
| `get_positions()` | Positions as `pd.DataFrame` |
| `get_value(data)` | Mark-to-market value using provided close prices |

Note: `get_value(data)` requires `data` to be a DataFrame with a 'close' column indexed by `stock_code`.

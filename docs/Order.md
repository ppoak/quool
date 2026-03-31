# Order & Delivery

## `Delivery`

**File:** `quool/order.py`

Represents a single transaction record (execution fill or cash/position movement).

```python
from quool import Delivery
```

### AMOUNT_SIGN / QUANTITY_SIGN

```python
Delivery.AMOUNT_SIGN  # {'TRANSFER': 1, 'WITHDRAW': -1, 'BUY': -1, 'SELL': 1, 'DIVIDEND': 1}
Delivery.QUANTITY_SIGN # {'TRANSFER': 1, 'WITHDRAW': -1, 'BUY': 1, 'SELL': -1, 'SPLIT': 1}
```

Sign conventions:
- `AMOUNT_SIGN` maps delivery type to conventional cash-flow sign
- `QUANTITY_SIGN` maps delivery type to quantity sign (including commission direction)

### `Delivery.__init__`

```python
Delivery(
    time: str,       # Event timestamp (pandas-parsable, e.g., '2024-03-01 10:00:00')
    code: str,       # Instrument identifier (e.g., ticker)
    type: str,       # Delivery type: BUY, SELL, TRANSFER, WITHDRAW, DIVIDEND, SPLIT
    quantity: float, # Executed quantity (units or shares)
    price: float,    # Execution price per unit
    comm: float,     # Commission for the event
    id: str = None,  # Unique identifier (auto-generated UUID4 if not provided)
)
```

**Net amount** is computed internally as:
```python
amount = quantity * price + QUANTITY_SIGN[type] * comm
```

### `Delivery.dump`

Serialize to a JSON-friendly dictionary.

```python
dumped = delivery.dump()
# Returns: {
#   'id': str, 'time': str (ISO), 'type': str, 'code': str,
#   'quantity': float, 'price': float, 'amount': float, 'commission': float
# }
```

### `Delivery.load`

Reconstruct from a dictionary.

```python
restored = Delivery.load(data: dict)
```

---

## `Order`

**File:** `quool/order.py`

Tracks an order through its full lifecycle: CREATED → SUBMITTED → PARTIAL → FILLED/CANCELED/EXPIRED/REJECTED.

```python
from quool import Order
```

### Status Constants

```python
Order.CREATED   # Order created but not yet submitted
Order.SUBMITTED # Order submitted to broker, awaiting execution
Order.PARTIAL   # Order partially filled
Order.FILLED    # Order fully filled
Order.CANCELED  # Order canceled by user
Order.EXPIRED   # Order expired (good-till timestamp passed)
Order.REJECTED  # Order rejected (insufficient cash/position or other error)
```

### Execution Type Constants

```python
Order.MARKET     # Execute immediately at next tick
Order.LIMIT      # Execute at specified price or better
Order.STOP       # Execute once price crosses trigger
Order.STOPLIMIT  # Trigger STOP, then execute as LIMIT
Order.TARGET     # Execute once price reaches target (reverse trigger)
Order.TARGETLIMIT# Trigger TARGET, then execute as LIMIT
```

### Side Constants

```python
Order.BUY   # Buy side
Order.SELL  # Sell side
```

### `Order.__init__`

```python
Order(
    time: str,       # Creation timestamp (pandas-parsable)
    code: str,       # Instrument identifier
    type: str,       # Order side: BUY or SELL
    quantity: int,   # Total requested quantity
    exectype: str = MARKET,   # Execution type
    limit: float = None,      # Limit price for LIMIT/STOPLIMIT orders
    trigger: float = None,    # Trigger price for STOP/STOPLIMIT orders
    id: str = None,           # Unique identifier (auto-generated if not provided)
    valid: str = None,        # Good-till timestamp (pandas-parsable)
)
```

### `Order.__add__`

Register a fill (Delivery) against the order using the `+` operator:

```python
order + delivery  # Mutates order, returns self
```

Updates: `filled`, `amount`, `comm`, `price_slip`, `price_eff`, `status` (PARTIAL or FILLED), `time`.

**Raises:** `ValueError` if `delivery.quantity > order.quantity - order.filled`.

### `Order.cancel`

Cancel the order if it is still open (CREATED, SUBMITTED, or PARTIAL).

```python
order.cancel()
```

### `Order.is_alive`

Check whether the order is still active at a given time.

```python
order.is_alive(time: pd.Timestamp) -> bool
```

Returns `False` if status is not CREATED/SUBMITTED/PARTIAL, or if `valid` timestamp has passed (order is marked EXPIRED).

### `Order.dump`

Serialize to a JSON-friendly dictionary.

```python
order.dump(delivery: bool = True) -> dict
```

### `Order.load`

Reconstruct from a dictionary.

```python
Order.load(data: dict) -> Order
```

### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `create` | `pd.Timestamp` | Creation timestamp |
| `time` | `pd.Timestamp` | Last updated timestamp |
| `code` | `str` | Instrument identifier |
| `type` | `str` | BUY or SELL |
| `quantity` | `int` | Total requested quantity |
| `exectype` | `str` | MARKET, LIMIT, STOP, STOPLIMIT, TARGET, TARGETLIMIT |
| `limit` | `float` | Limit price |
| `trigger` | `float` | Trigger price |
| `price_slip` | `float` | Volume-weighted average fill price (excl. commission) |
| `price_eff` | `float` | Effective average price (incl. commission) |
| `comm` | `float` | Total commissions accrued |
| `status` | `str` | Current status |
| `filled` | `int` | Total filled quantity |
| `amount` | `float` | Accumulated notional |
| `id` | `str` | Unique identifier |
| `valid` | `pd.Timestamp` | Good-till timestamp |
| `delivery` | `list[Delivery]` | Associated fills |

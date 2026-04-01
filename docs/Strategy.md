# Strategy

## `Strategy`

**File:** `quool/strategy.py`

Base class for trading strategies. Coordinates `Source` (market data) and `Broker` (execution).

```python
from quool import Strategy
```

### Lifecycle

```python
class MyStrategy(Strategy):
    def init(self, **kwargs):       # Called once before the first iteration
        pass

    def preupdate(self, **kwargs):  # Called after each iteration, before update()
        pass

    def update(self, **kwargs):     # Core logic — place orders here
        pass

    def stop(self, **kwargs):       # Called once at the end
        pass
```

### `Strategy.__init__`

```python
Strategy(
    source: Source,                  # Market data provider
    broker: Broker,                  # Execution and accounting interface
    logger: logging.Logger = None,   # Logger (default: DEBUG-level class-named logger)
)
```

### Running

#### `Strategy.backtest`

```python
strategy.backtest(
    benchmark: pd.Series = None,  # Benchmark net value series
    history: bool = False,         # Store full history during backtest
    **kwargs,                      # Forwarded to init(), preupdate(), update()
) -> Any
```

Returns the evaluation summary from `Evaluator.report()`.

**Raises:** `ValueError` if no delivery data is available for evaluation.

#### `Strategy.run`

Blocking scheduler (foreground process).

```python
strategy.run(
    store: str = None,                    # Path to persist broker state
    history: bool = False,
    trigger: str = "interval",           # APScheduler trigger type
    trigger_kwargs: dict = None,         # Trigger arguments (default: {'seconds': 30})
    **kwargs,
)
```

#### `Strategy.arun`

Background scheduler (non-blocking).

```python
strategy.arun(
    store: str = None,
    history: bool = False,
    trigger: str = "interval",
    trigger_kwargs: dict = None,
    scheduler: BackgroundScheduler = None,  # Existing scheduler to reuse
    **kwargs,
) -> BackgroundScheduler
```

Resumes an existing job with the class name if found; otherwise creates a new one.

### Execution Helpers

All return the submitted `Order` (or `None` if no adjustment needed).

#### `Strategy.buy`

```python
strategy.buy(
    code: str,              # Instrument code
    quantity: int,         # Requested quantity
    exectype: str = MARKET,  # Execution type
    limit: float = None,
    trigger: float = None,
    id: str = None,
    valid: str = None,
) -> Order
```

#### `Strategy.sell`

```python
strategy.sell(code, quantity, exectype=MARKET, limit=None, trigger=None, id=None, valid=None) -> Order
```

#### `Strategy.close`

Close the entire position in an instrument (submits SELL for full position).

```python
strategy.close(
    code: str,
    exectype: str = MARKET,
    limit: float = None,
    trigger: float = None,
    id: str = None,
    valid: str = None,
) -> Order or None
```

Returns `None` if no position exists.

#### `Strategy.order_target_value`

Adjust position to a target notional value.

```python
strategy.order_target_value(
    code: str,
    value: float,           # Target position value in currency units
    exectype: str = MARKET,
    limit: float = None,
    trigger: float = None,
    id: str = None,
    valid: str = None,
) -> Order or None
```

Computes: `delta = target_value - current_position * close_price`, then submits a BUY/SELL for the delta quantity.

Returns `None` if instrument not in source data index.

#### `Strategy.order_target_percent`

Adjust position to a target portfolio percentage.

```python
strategy.order_target_percent(
    code: str,
    percent: float,          # Target fraction (e.g., 0.10 for 10%)
    exectype: str = MARKET,
    limit: float = None,
    trigger: float = None,
    id: str = None,
    valid: str = None,
) -> Order or None
```

Computes: `target_value = portfolio_value * percent`, then delegates to `order_target_value()`.

### Portfolio State

#### `Strategy.get_value`

```python
strategy.get_value() -> float  # Total portfolio value (positions + cash)
```

#### `Strategy.get_positions`

```python
strategy.get_positions() -> pd.Series  # Position quantities indexed by code
```

### Order Management

#### `Strategy.cancel`

```python
strategy.cancel(order_or_id: str | Order) -> Order
```

**Raises:** `KeyError` if order id not found.

### Persistence

#### `Strategy.dump`

Serialize broker state to a dictionary.

```python
strategy.dump(history: bool = True) -> dict
```

#### `Strategy.load` (classmethod)

Reconstruct a Strategy from serialized broker state.

```python
Strategy.load(
    cls,
    data: dict,                          # Serialized broker state
    commission: FixedRateCommission,      # Commission model
    slippage: FixedRateSlippage,          # Slippage model
    source: Source,                      # Market data source
    logger: logging.Logger = None,
) -> Strategy
```

### Logging

#### `Strategy.log`

```python
strategy.log(message: str, level: str = "DEBUG")
```

Prefixes log message with current timestamp.

### Notification

#### `Strategy.notify`

Notification hook for order status changes. Default behavior logs the order.

```python
strategy.notify(order: Order)
```

Override to handle filled orders, rejections, or cancellations.

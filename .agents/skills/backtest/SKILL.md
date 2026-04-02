---
name: backtest
description: >
  Authoring guideline for quantitative strategy backtesting in Quool. Covers
  Source/Broker/Strategy component selection, order placement, portfolio state
  access, historical data queries, and backtest execution. Always read this
  before writing or modifying any backtest strategy code.
author: quool team
version: 0.0.1
triggers:
  - write a backtest strategy
  - implement a quantitative strategy
  - run backtest with quool
  - create strategy with source and broker
---

# Backtest Skill

> Authoring guideline: read this entire document before writing any code. Every section is load-bearing.

## Role

You are a quantitative strategy engineer. Your job is to produce a complete, runnable backtest implementation from a user's informal specification. Follow the workflow below step by step. Do not skip steps, do not make assumptions not supported by the user's request, and do not hardcode values the user did not specify.

---

## Core Architecture

Every backtest is built from three tightly coupled components:

```
Source (market data)  →  Strategy (logic)  →  Broker (execution & portfolio)
```

The event loop (inside `Strategy.backtest()`) runs in this exact order every iteration:

```
1. source.update()          # advance to next timestamp
2. broker.update(source)    # match pending orders, execute fills
3. notify(order)             # receive filled/cancelled orders (default: log only)
4. preupdate()               # housekeeping hook
5. update()                  # YOUR trading logic — place new orders here
```

Lifecycle hooks: `init()` runs once before the loop; `stop()` runs once after the loop ends.

---

## Step 1 — Choose and Initialize the Source

### DataFrameSource (in-memory, simplest)

Use when data fits in RAM as a pandas DataFrame with a MultiIndex (time, code).

```python
from quool import DataFrameSource

source = DataFrameSource(
    data=df,              # DataFrame with MultiIndex (timestamp, code) and OHLCV columns
    open="open",          # adjust to your actual column name
    high="high",          # adjust to your actual column name
    low="low",            # adjust to your actual column name
    close="close",        # adjust to your actual column name
    volume="volume",      # adjust to your actual column name
)
```

- `source.time` — current timestamp
- `source.data` — snapshot at current time (rows = instruments, indexed by code)
- `source.datas` — all data from start up to and including current time
- `source.update()` returns `None` when data is exhausted (signals end of backtest)

### DuckPQSource (DuckDB/Parquet, large datasets)

Use when data lives in Parquet files on disk. You **must construct `bar` manually** — there is no default that applies to all datasets.

#### The `<table><sep><column>` path pattern

Every value in `bar` and `extra` is a **factor path** with the format:

```
<table> <sep> <column>
```

| Component | Meaning |
|-----------|---------|
| `table` | Table (or top-level directory) name in the DuckPQ store |
| `sep` | Separator between table and column — defaults to `"/"` |
| `column` | The column name stored inside that table |

The path is split by `sep` into exactly two parts: `[table, column]`. **No subdirectory nesting is supported.** If your data is organized under a path like `daily/ohlcv/close`, you must register and query it at whatever the top-level table name actually is — check your DuckPQ store's actual table and column names before writing `bar`.

**Example:** If your Parquet store has a table `kline` with columns `open, high, low, close, vol`:

```python
sep = "/"
bar = {
    "open":   "kline/open",
    "high":   "kline/high",
    "low":    "kline/low",
    "close":  "kline/close",
    "volume": "kline/vol",   # use your actual volume column name
}
```

#### DuckPQSource initialization

```python
from quool import DuckPQ, DuckPQSource

db = DuckPQ(root_path="/path/to/data")   # adjust to your actual data directory
source = DuckPQSource(
    source=db,
    begin="2024-01-01",       # adjust to your actual backtest start date
    end="2024-12-31",         # adjust to your actual backtest end date
    datetime_col="date",       # adjust to your actual datetime column name
    code_col="code",           # adjust to your actual instrument code column name
    sep="/",                   # adjust if your factor paths use a different separator
    bar={
        # Construct bar using the <table><sep><column> pattern.
        # All bar entries must come from the same table.
        # Replace table name and column names with what actually exists in your data.
        "open":   "my_table/open",      # adjust table and column
        "high":   "my_table/high",      # adjust
        "low":    "my_table/low",       # adjust
        "close":  "my_table/close",     # adjust
        "volume": "my_table/volume",    # adjust
    },
    extra={
        # Optional extra fields using the same <table><sep><column> pattern.
        # e.g. "turnover": "my_table/turnover"
    },
    limit=None,   # adjust: set an integer to bound the historical window
)
```

- `source.datas` — queries DuckDB for historical window (respects `limit` if set)
- `source.data` — snapshot at current time

### Source data access patterns

```python
# Current snapshot — all instruments at current time
snapshot = self.source.data       # DataFrame, index = code

# Single instrument price at current time
price = self.source.data.loc["000001", "close"]

# Historical data (DataFrameSource: all past data; DuckPQSource: windowed)
hist = self.source.datas

# Current timestamp
ts = self.source.time
```

---

## Step 2 — Choose and Initialize the Broker

### Broker (generic simulated)

```python
from quool import Broker, FixedRateCommission, FixedRateSlippage

broker = Broker(
    commission=FixedRateCommission(),   # uses default A-share rates; adjust if needed
    slippage=FixedRateSlippage(),      # uses default slippage; adjust if needed
)
```

### AShareBroker (A-share market, 100-share lot enforcement)

```python
from quool import AShareBroker

broker = AShareBroker(
    commission=FixedRateCommission(),
    slippage=FixedRateSlippage(),
)
```

> **Adjust the initial capital** to what the user specifies.

```python
import pandas as pd
broker.transfer(pd.Timestamp("2024-01-01"), 1_000_000)  # adjust amount and date
```

Available `Broker` attributes accessed via `self.broker` in a Strategy:
- `broker.balance` — current cash (float)
- `broker.positions` — dict `{code: quantity}`
- `broker.pendings` — `deque[Order]` of open orders
- `broker.orders` — `list[Order]` of processed orders (filled/canceled/rejected)
- `broker.delivery` — `list[Delivery]` of all executions

---

## Step 3 — Write the Strategy Class

### Mandatory structure

```python
from quool import Strategy

class MyStrategy(Strategy):

    def init(self, **kwargs):
        # One-time setup: compute indicators, set parameters
        # Runs once before the backtest loop starts
        pass

    def update(self, **kwargs):
        # Core logic: called every iteration AFTER source update and broker matching.
        # Place orders here using self.buy(), self.sell(), self.order_target_value(), etc.
        #
        # WRITE COMMENTS inside this function so the user can understand
        # what each block of trading logic does.
        raise NotImplementedError("`update` must be implemented")

    def preupdate(self, **kwargs):
        # Optional: run before each update() — logging, risk checks, housekeeping
        pass

    def stop(self, **kwargs):
        # Optional: run once at the end — persist state, final logs
        pass
```

### The `update()` method is the only required hook.

Everything else is optional. `init()`, `preupdate()`, `stop()`, and `notify()` all have no-op defaults.

---

## Step 4 — Place Orders (Inside `update()`)

> **Adjust `code`, `quantity`, `limit`, and `trigger`** to match the user's actual instrument identifiers, sizing rules, and price levels in the data.

### Order types

| Constant | Meaning |
|----------|---------|
| `Order.MARKET` | Execute immediately at next tick |
| `Order.LIMIT` | Execute at limit price or better |
| `Order.STOP` | Trigger then execute as MARKET when trigger hit |
| `Order.STOPLIMIT` | Trigger then execute as LIMIT when trigger hit |
| `Order.TARGET` | BUY: execute when low <= trigger; SELL: execute when high >= trigger |
| `Order.TARGETLIMIT` | Like TARGET but with a limit price on execution |

### Basic order methods

```python
# BUY MARKET
self.buy(code="000001", quantity=100)   # adjust code and quantity

# BUY LIMIT
self.buy(code="000001", quantity=100, exectype=Order.LIMIT, limit=10.5)

# SELL MARKET
self.sell(code="000001", quantity=100)

# SELL with trigger (stop-loss)
self.sell(code="000001", quantity=100, exectype=Order.STOP, trigger=9.5)

# Cancel an order
self.cancel(order_or_id=order)           # pass the Order object
self.cancel(order_or_id="order-uuid-string")

# Close entire position
self.close(code="000001")
```

### Target-value and target-percent orders

```python
# Target notional value (computes quantity automatically)
self.order_target_value(code="000001", value=50_000)

# Target portfolio percentage (10% of current portfolio value)
self.order_target_percent(code="000001", percent=0.10)
```

### Order execution flow inside `update()`

Orders placed during `update()` are submitted to the broker. They will be **matched in the next iteration's `broker.update()`** after `source.update()` provides the new snapshot.

- MARKET orders fill immediately if volume is available.
- LIMIT orders only fill when price conditions are met.
- STOP/STOPLIMIT orders are dormant until the trigger is hit.

---

## Step 5 — Access Portfolio State (Inside `update()`)

```python
# Current total portfolio value (cash + mark-to-market positions)
total_value = self.get_value()

# Current positions: pandas Series indexed by code
positions = self.get_positions()
# positions["000001"] -> quantity (float)  # adjust code to actual instrument

# Check if holding a position
if "000001" in self.broker.positions:   # adjust code
    qty = self.broker.positions["000001"]

# Current cash
cash = self.broker.balance

# All pending (open) orders
pending = self.broker.get_pendings()

# All processed orders (filled/canceled/rejected)
orders = self.broker.get_orders()

# All deliveries (executions)
delivery = self.broker.get_delivery()
```

---

## Step 6 — Access Historical Data (Inside `update()`)

> **Adjust column names** (`"close"`, `"open"`, etc.) to match what was actually passed to the Source constructor.

```python
# All historical data up to current time
hist = self.source.datas

# Compute a simple moving average — adjust window and column as needed
sma20 = hist["close"].unstack("code").rolling(20).mean()

# Latest close price of a single instrument at current time
last_close = self.source.data.loc["000001", "close"]   # adjust code and column

# Price at N bars ago
lookback = 5
prev_close = hist["close"].unstack("code").shift(lookback).loc[self.source.time]
```

---

## Step 7 — Run the Backtest

```python
results = MyStrategy(source, broker).backtest(benchmark=benchmark)
```

The returned `results` is a `dict` with keys:
- `values` — DataFrame with `net`, `total`, `market`, `cash`, `turnover` columns
- `positions` — DataFrame of position quantities over time
- `trades` — DataFrame of trade summaries (open_at, close_at, return, duration)
- `evaluation` — Series of metrics (Sharpe, Calmar, Sortino, max_drawdown, etc.)
- `orders` — DataFrame of order history
- `pendings` — DataFrame of open orders at end
- `delivery` — DataFrame of all execution records

### Key evaluation metrics available in `results["evaluation"]`

| Metric | Key |
|--------|-----|
| Total return | `evaluation["total_return"]` |
| Annual return | `evaluation["annual_return"]` |
| Annual volatility | `evaluation["annual_volatility"]` |
| Sharpe ratio | `evaluation["sharpe_ratio"]` |
| Calmar ratio | `evaluation["calmar_ratio"]` |
| Sortino ratio | `evaluation["sortino_ratio"]` |
| Max drawdown | `evaluation["max_drawdown"]` |
| VaR 5% | `evaluation["VaR_5%"]` |
| CVaR 5% | `evaluation["CVaR_5%"]` |
| Beta | `evaluation["beta"]` |
| Alpha | `evaluation["alpha"]` |
| Turnover ratio | `evaluation["turnover_ratio"]` |

---

## Step 8 — Logging

```python
from quool import setup_logger

logger = setup_logger("MyStrategy", level="DEBUG", file="strategy.log")
self.log("Order submitted", level="INFO")
self.log(f"Position size: {qty}", level="DEBUG")
```

---

## Putting It All Together — Full Minimal Example

> **Adjust all placeholder values** (paths, column names, instrument codes, dates, capital, rates) before running.

```python
import pandas as pd
from quool import (
    DataFrameSource,
    Broker,
    Strategy,
    Order,
    FixedRateCommission,
    FixedRateSlippage,
)

# 1. Prepare data — DataFrame with MultiIndex (timestamp, code) and OHLCV columns
data = pd.read_parquet("data.parquet")   # adjust path
source = DataFrameSource(
    data=data,
    open="open",     # adjust to your actual column name
    high="high",     # adjust
    low="low",       # adjust
    close="close",   # adjust
    volume="volume", # adjust
)

# 2. Initialize broker
broker = Broker(
    commission=FixedRateCommission(),
    slippage=FixedRateSlippage(),
)
broker.transfer(pd.Timestamp("2024-01-01"), 1_000_000)  # adjust date and capital

# 3. Define strategy
class DemoStrategy(Strategy):

    def init(self):
        # One-time setup before the loop starts
        self.lookback = 20   # adjust to your actual lookback period

    def update(self):
        # Get current price for the target instrument
        code = "000001"      # adjust to your actual instrument code (e.g. "600000.SH", "AAPL")
        close = self.source.data.loc[code, "close"]

        # Build historical series for the instrument
        hist = self.source.datas["close"].unstack("code")
        # Adjust column name if your data uses a different close column name

        # Compute dual-MA signal
        sma_short = hist[code].rolling(5).mean().iloc[-1]   # adjust window
        sma_long  = hist[code].rolling(20).mean().iloc[-1]  # adjust window

        # Trading logic: golden cross buy, death cross sell
        if sma_short > sma_long and code not in self.broker.positions:
            # Golden cross — buy signal
            self.buy(code=code, quantity=100, exectype=Order.MARKET)  # adjust quantity

        elif sma_short < sma_long and code in self.broker.positions:
            # Death cross — sell signal
            self.close(code=code, exectype=Order.MARKET)

# 4. Run backtest — adjust benchmark if provided
results = DemoStrategy(source, broker).backtest()

# 5. Read results
print(results["evaluation"][["annual_return", "sharpe_ratio", "max_drawdown"]])
```

---

## Decision Tree — Choosing Components

```
User wants to backtest on...
│
├─ In-memory data (fits in RAM)
│  └─ DataFrameSource + Broker (or AShareBroker)
│
├─ Parquet/DuckDB data on disk
│  └─ DuckPQSource + Broker
│
└─ A-share market (100-share lots)
   └─ DataFrameSource or DuckPQSource + AShareBroker
```

---

## Order Validity and Expiration

```python
# Good-till-time order — expires at the specified timestamp
self.buy(code="000001", quantity=100, valid="2024-01-15")
# If not filled by 2024-01-15, the order expires automatically
```

---

## Critical Gotchas

1. **Orders execute in the next iteration.** When you place an order in `update()`, it sits in the pending queue and gets matched when `broker.update()` runs on the next tick. Do not assume the order fills synchronously.

2. **Insufficient cash.** BUY orders are rejected if `cost + commission > broker.balance`. Ensure sufficient capital is transferred upfront via `broker.transfer()`.

3. **Insufficient position.** SELL orders are rejected if the position quantity is less than the sell quantity. Use `self.close()` (which queries the actual position) rather than hardcoding quantities.

4. **`source.data` is a snapshot.** It reflects the current time only. Use `source.datas` for historical series.

5. **DuckPQSource `datas` is bounded.** If `limit=N` is set, only the most recent N timestamps are visible. The evaluator also uses `source.datas` — if `limit` is too small, evaluation will be inaccurate.

6. **`broker.transfer()` sets broker time.** Call `broker.transfer(time, amount)` before `backtest()` to initialize both cash balance and the broker's internal clock.

7. **SELL commission includes stamp duty.** `FixedRateCommission` adds `stamp_duty_rate * amount` on SELL fills. Account for this in cost estimates.

8. **`notify()` is called per filled order.** The default logs the order. Override to handle fills (e.g., record fill prices for risk management).

---

## Workflow Checklist

Before returning the final implementation, verify:

- [ ] Source chosen and initialized with correct date range
- [ ] `bar` dict constructed using `<table><sep><column>` pattern with actual table/column names from the user's data
- [ ] Broker initialized with commission and slippage matching actual market fees
- [ ] `broker.transfer()` called with correct initial capital amount
- [ ] Strategy class with `init()` and `update()` implemented
- [ ] `update()` contains comments explaining every trading signal and order block
- [ ] `update()` places orders using correct `code`, `quantity`, and `exectype`
- [ ] `backtest()` called on the strategy instance
- [ ] Results extracted and displayed (evaluation metrics, optionally values/trades)
- [ ] No hardcoded values (codes, dates, column names, capital, rates) that should come from user input

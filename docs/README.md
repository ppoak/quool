# Quool Documentation

**Quool** is an event-driven quantitative backtesting and trading framework. This directory contains detailed documentation for every public module and class.

---

## Index

| Document | Description |
|----------|-------------|
| [Order.md](Order.md) | `Order`, `Delivery` — order lifecycle and execution records |
| [Broker.md](Broker.md) | `Broker`, `AShareBroker`, `XueQiuBroker`, `XtBroker` |
| [Strategy.md](Strategy.md) | `Strategy` — strategy lifecycle and helpers |
| [Source.md](Source.md) | `Source`, `DataFrameSource`, `DuckPQSource`, `RealtimeSource`, `XtDataPreloadSource` |
| [Evaluator.md](Evaluator.md) | `Evaluator` — performance metrics computation |
| [Friction.md](Friction.md) | `FixedRateCommission`, `FixedRateSlippage` |
| [Storage.md](Storage.md) | `DuckTable`, `DuckPQ` — DuckDB-backed Parquet storage |
| [Util.md](Util.md) | `setup_logger`, `notify_task`, `proxy_request`, `generate_usage` |

---

## Module Overview

### Core (`quool/`)

| File | Exports |
|------|---------|
| `order.py` | `Order`, `Delivery` |
| `broker.py` | `Broker` |
| `strategy.py` | `Strategy` |
| `source.py` | `Source` |
| `evaluator.py` | `Evaluator` |
| `friction.py` | `FixedRateCommission`, `FixedRateSlippage` |
| `storage.py` | `DuckTable`, `DuckPQ` — Parquet storage with DuckDB |
| `util.py` | `setup_logger`, `notify_task`, `proxy_request`, `generate_usage` |

### Brokers (`quool/brokers/`)

| File | Exports |
|------|---------|
| `ashare.py` | `AShareBroker` |
| `xueqiu.py` | `XueQiuBroker`, `XueQiu` |
| `xuntou.py` | `XtBroker` |

### Sources (`quool/sources/`)

| File | Exports |
|------|---------|
| `dataframe.py` | `DataFrameSource` |
| `duck.py` | `DuckPQSource`, `parse_factor_path` |
| `realtime.py` | `RealtimeSource`, `read_realtime`, `is_trading_time` |
| `xuntou.py` | `XtDataPreloadSource` |

---

## Design Principles

### Event-Driven Loop

Every backtest iteration follows this sequence:

```
Strategy.backtest()
  → Strategy.init()                          # once at start
  → loop:
      Source.update()                         # advance time
      Broker.update(source)                  # match pending orders
        → Broker._match()                    # evaluate triggers/limits
        → Broker._execute()                   # fill if matched
      Strategy.notify(order)                 # receive filled/cancelled orders
      Strategy.preupdate()                   # housekeeping hook
      Strategy.update()                      # place new orders
  → Strategy.stop()                          # once at end
  → Evaluator.evaluate()                     # compute metrics
```

### Order Execution Types

| Type | Description | Matching Logic |
|------|-------------|----------------|
| `MARKET` | Execute immediately at next tick | Always fills if volume available |
| `LIMIT` | Execute at specified price or better | BUY: low ≤ limit; SELL: high ≥ limit |
| `STOP` | Execute once price crosses trigger | BUY: high ≥ trigger; SELL: low ≤ trigger |
| `STOPLIMIT` | Trigger STOP, then execute as LIMIT | Trigger activates, then limit price rules |
| `TARGET` | Execute once price reaches target | BUY: low ≤ trigger; SELL: high ≥ trigger |
| `TARGETLIMIT` | Trigger TARGET, then execute as LIMIT | Trigger activates, then limit price rules |

### Portfolio Value Calculation

- **net value**: `total_value / capital_base` — normalized by the initial capital + transfers
- **total value**: `cash + sum(positions * close_prices)` — mark-to-market
- **turnover**: `sum(|delta_position * price|) / previous_total_value`

### Commission Model

`FixedRateCommission` applies:
- **BUY**: `max(rate * amount, min_commission)`
- **SELL**: `max(rate * amount, min_commission) + stamp_duty_rate * amount`

### Slippage Model

`FixedRateSlippage` adjusts execution price based on `slip_rate` and available volume:
- For BUY MARKET: `min(high, open * (1 + slip_rate))`
- For SELL MARKET: `max(low, volume-weighted_adjustment)`
- LIMIT orders are capped/floored by both the kline bounds and the order's limit price

---

## Usage Patterns

### Backtesting

```python
from quool import DataFrameSource, Broker, Strategy
from quool import FixedRateCommission, FixedRateSlippage, Evaluator

source = DataFrameSource(data)
broker = Broker(commission=FixedRateCommission(), slippage=FixedRateSlippage())
broker.transfer(pd.Timestamp("2024-01-01"), 1_000_000)

class MyStrategy(Strategy):
    def init(self): ...
    def update(self): ...

results = MyStrategy(source, broker).backtest(benchmark=benchmark)
```

### DuckDB/Parquet Storage

```python
from quool import DuckPQ, DuckTable

# Manage multiple tables
db = DuckPQ(root_path='/data/warehouse')
db.register()

# Query with SQL
df = db.select(table='ohlcv', columns=['date', 'close'], where="symbol = '000001'")

# Upsert with partitioning
db.upsert(table='ohlcv', df=new_data, keys=['date', 'symbol'], partition_by=['symbol'])
```

### Live Trading — XueQiu Paper Trading

```python
from quool import XueQiuBroker, RealtimeSource

broker = XueQiuBroker(token="<xq_token>", name="MyPortfolio")
source = RealtimeSource()
# Use XueQiuBroker.transfer(), broker.get_balance(), etc.
```

### Live Trading — XtQuant

```python
from quool import XtBroker, XtDataPreloadSource

broker = XtBroker(account="<account_id>", path="<xtquant_path>")
source = XtDataPreloadSource(path="<data_path>", begin="20240101", end="20241231")
```

### A-Share Market

```python
from quool import AShareBroker  # quantities automatically floored to 100-share lots
```

### Logging

```python
from quool import setup_logger

logger = setup_logger('my_strategy', level='INFO', file='/var/log/strategy.log')
logger.info('Backtest started')
```

### Task Notification

```python
from quool import notify_task

@notify_task(sender='bot@example.com', receiver='admin@example.com', smtp_server='smtp.example.com')
def long_running_task():
    # ... task code ...
    return result
```

---

## Broker Integrations Comparison

| Feature | `Broker` | `AShareBroker` | `XueQiuBroker` | `XtBroker` |
|---------|----------|----------------|----------------|------------|
| Simulated execution | ✓ | ✓ | Partial (remote) | ✗ |
| Lot-size enforcement | ✗ | ✓ (100 shares) | ✗ | ✓ (via XtQuant) |
| Paper trading | — | — | ✓ (XueQiu groups) | — |
| Live trading | — | — | — | ✓ (XtQuant) |
| Custom commission | ✓ | ✓ | ✓ | ✗ |
| Custom slippage | ✓ | ✓ | ✗ | ✗ |

---

## Data Source Comparison

| Feature | `DataFrameSource` | `DuckPQSource` | `RealtimeSource` | `XtDataPreloadSource` |
|---------|-------------------|----------------|-------------------|----------------------|
| Historical | ✓ | ✓ | ✗ | ✓ |
| Real-time | ✗ | ✗ | ✓ | ✗ |
| Backend | pandas | DuckDB/Parquet | EastMoney API | xtquant |
| Multi-instrument | ✓ | ✓ | ✓ | ✓ |

---

## Key Metrics Reference

See [Evaluator.md](Evaluator.md) for the full list. Key metrics:

| Metric | Formula |
|--------|---------|
| annual_return | `(1 + total_return) ** (365 / days) - 1` |
| sharpe_ratio | `annual_return / annual_volatility` |
| calmar_ratio | `annual_return / \|max_drawdown\|` |
| sortino_ratio | `annual_return / (downside_std * sqrt(252))` |
| beta | `cov(returns, benchmark) / var(benchmark)` |
| alpha | `(mean(returns) - beta * mean(benchmark)) * 252` |
| information_ratio | `excess_return / tracking_error` |
| VaR_5% | `percentile(returns, 5)` |
| CVaR_5% | `mean(returns[returns <= VaR_5%])` |

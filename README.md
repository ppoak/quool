# Quool

**Quantitative Toolkit** — an event-driven backtesting and live trading framework for quantitative strategies.

Quool provides a modular architecture built around three pillars: `Source` (market data), `Broker` (execution & accounting), and `Strategy` (logic). It supports both backtesting with historical data and paper/live trading with broker integrations.

## Installation

```bash
pip install quool
```

Requires Python >= 3.10.

## Quick Start

```python
import pandas as pd
from quool import DataFrameSource, Broker, Strategy
from quool import FixedRateCommission, FixedRateSlippage

# 1. Market data source (MultiIndex DataFrame: time x code)
source = DataFrameSource(market_data)

# 2. Broker with commission and slippage models
broker = Broker(
    commission=FixedRateCommission(),
    slippage=FixedRateSlippage(),
)
broker.transfer(pd.Timestamp("2024-01-01"), 1_000_000)  # initial cash

# 3. Implement strategy
class MyStrategy(Strategy):
    def init(self):       # called once before backtest
        pass

    def update(self):     # called every timestamp
        # self.buy("000001", 100)   # buy 100 shares at market
        # self.order_target_percent("000001", 0.1)  # target 10% portfolio
        pass

# 4. Run backtest
strategy = MyStrategy(source, broker)
results = strategy.backtest()
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      Strategy                            │
│  init() → preupdate() → update() → stop()               │
└──────────────┬─────────────────────────┬────────────────┘
               │                         │
        ┌──────▼──────┐            ┌──────▼──────┐
        │   Source    │            │   Broker    │
        │ (market    │            │ (execution  │
        │   data)    │            │  & accounting)
        └────────────┘            └──────┬──────┘
                                          │
                                   ┌──────▼──────┐
                                   │   Order /   │
                                   │  Delivery   │
                                   └─────────────┘
```

### Core Concepts

#### Source

`Source` is the abstract market data provider. Subclasses implement `update()` to advance time and return OHLCV snapshots.

| Class | Description |
|-------|-------------|
| `DataFrameSource` | Historical data from a pandas DataFrame (MultiIndex: time × code) |
| `DuckPQSource` | DuckDB/Parquet queries via parquool |
| `RealtimeSource` | Real-time EastMoney API with a rolling buffer |
| `XtDataPreloadSource` | XtQuant historical data preloaded into a DataFrame |

#### Broker

`Broker` manages order execution, portfolio accounting (cash & positions), and order matching for backtesting. For live trading, broker subclasses integrate with external systems.

| Class | Description |
|-------|-------------|
| `Broker` | Core simulated broker with pluggable commission/slippage models |
| `AShareBroker` | Enforces A-share 100-share lot-size rules |
| `XueQiuBroker` | XueQiu paper trading integration |
| `XtBroker` | XtQuant live trading gateway |

#### Order & Delivery

- `Order` tracks the full lifecycle: CREATED → SUBMITTED → PARTIAL → FILLED/CANCELED/EXPIRED/REJECTED
- `Delivery` records individual fills (execution details: price, quantity, commission)
- Execution types: MARKET, LIMIT, STOP, STOPLIMIT, TARGET, TARGETLIMIT

#### Strategy

Base class for trading strategies. Provides:
- Lifecycle hooks: `init()`, `preupdate()`, `update()`, `stop()`
- Execution helpers: `buy()`, `sell()`, `close()`, `order_target_value()`, `order_target_percent()`
- Backtesting: `backtest()` — blocking loop; `run()` / `arun()` — real-time scheduling
- Persistence: `dump()`, `load()`, `store()`, `restore()`

#### Evaluator

Computes comprehensive performance metrics from broker deliveries:

- **Return**: total_return, annual_return, annual_volatility
- **Risk-adjusted**: sharpe_ratio, calmar_ratio, sortino_ratio
- **Drawdown**: max_drawdown, max_drawdown_period
- **Risk**: VaR_5%, CVaR_5%
- **Benchmark**: beta, alpha, excess_return, information_ratio
- **Trading**: position_duration, trade_win_rate, trade_return
- **Distribution**: skewness, kurtosis, day_return_win_rate, monthly_win_rate

#### Friction Models

| Class | Description |
|-------|-------------|
| `FixedRateCommission` | Flat-rate commission with minimum fee and stamp duty |
| `FixedRateSlippage` | Slippage model adjusting execution price based on volume |

## API Reference

For detailed API documentation, see:

- [doc/README.md](doc/README.md) — Full module index and detailed documentation
- [doc/Order.md](doc/Order.md) — Order and Delivery models
- [doc/Broker.md](doc/Broker.md) — Broker and execution
- [doc/Strategy.md](doc/Strategy.md) — Strategy lifecycle and helpers
- [doc/Source.md](doc/Source.md) — Data source implementations
- [doc/Evaluator.md](doc/Evaluator.md) — Performance evaluation
- [doc/Friction.md](doc/Friction.md) — Transaction cost models

## License

MIT

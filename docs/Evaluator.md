# Evaluator

**File:** `quool/evaluator.py`

Portfolio and strategy performance evaluator.

```python
from quool import Evaluator
```

## `Evaluator.__init__`

```python
Evaluator(broker: Broker, source: Source)
```

Initializes with broker (providing deliveries, orders, pending orders) and source (providing market data for evaluation).

---

## `Evaluator.report`

Produce a full evaluation report.

```python
Evaluator.report(benchmark: pd.Series = None) -> dict
```

**Returns:** A dictionary containing:

| Key | Type | Description |
|-----|------|-------------|
| `values` | `pd.DataFrame` | Columns: `net`, `total`, `market`, `cash`, `turnover` — indexed by time |
| `positions` | `pd.DataFrame` | Position quantities per instrument over time |
| `trades` | `pd.DataFrame` | Trade-level summary with open/close timing and returns |
| `evaluation` | `pd.Series` | Performance metrics (see below) |
| `orders` | `pd.DataFrame` | Historical orders from the broker |
| `pendings` | `pd.DataFrame` | Currently pending orders from the broker |
| `delivery` | `pd.DataFrame` | Delivery log from the broker |

---

## `Evaluator.evaluate`

Compute performance metrics from a net value series.

```python
Evaluator.evaluate(
    net: pd.Series,              # Net value series indexed by time (starting near 1.0)
    benchmark: pd.Series = None,  # Benchmark value series
    turnover: pd.Series = None,   # Turnover ratio per period
    trades: pd.DataFrame = None,  # Trade summary
) -> pd.Series
```

### Computed Metrics

#### Return Metrics

| Metric | Description |
|--------|-------------|
| `total_return` | `net.iloc[-1] - 1` |
| `annual_return` | `(1 + total_return) ** (365 / days) - 1` |
| `annual_volatility` | `returns.std() * sqrt(252)` |
| `sharpe_ratio` | `annual_return / annual_volatility` |
| `calmar_ratio` | `annual_return / \|max_drawdown\|` |
| `sortino_ratio` | `annual_return / (downside_std * sqrt(252))` |

#### Drawdown Metrics

| Metric | Description |
|--------|-------------|
| `max_drawdown` | Minimum value of `net / net.cummax() - 1` |
| `max_drawdown_period` | Number of periods between max drawdown start and end |

#### Risk Metrics

| Metric | Description |
|--------|-------------|
| `VaR_5%` | 5th percentile of returns |
| `CVaR_5%` | Mean of returns below VaR_5% |
| `turnover_ratio` | Mean turnover (if provided) |

#### Benchmark-Relative Metrics

| Metric | Description |
|--------|-------------|
| `beta` | `cov(returns, benchmark) / var(benchmark)` |
| `alpha` | `(mean(returns) - beta * mean(benchmark)) * 252` |
| `excess_return` | Final excess value - 1 |
| `excess_volatility` | `excess_returns.std() * sqrt(252)` |
| `information_ratio` | `excess_return / tracking_error` |

#### Trading Behavior Metrics (if `trades` provided)

| Metric | Description |
|--------|-------------|
| `position_duration` | Mean duration of trades in periods |
| `trade_win_rate` | Fraction of trades with positive profit |
| `trade_return` | Total profit / total open amount |

#### Distribution Metrics

| Metric | Description |
|--------|-------------|
| `skewness` | Returns skewness |
| `kurtosis` | Returns kurtosis |
| `day_return_win_rate` | Fraction of periods with positive returns |
| `monthly_return_std` | Standard deviation of monthly returns |
| `monthly_win_rate` | Fraction of months with positive returns |

---

## `Evaluator.evaluate_delivery`

Evaluate portfolio from signed delivery records and a price table.

```python
Evaluator.evaluate_delivery(
    delivery: pd.DataFrame,   # Indexed by [time, code]; columns: type, amount, quantity
    prices: pd.DataFrame,     # Wide table: rows=time, columns=code, values=close price
    benchmark: pd.Series = None,
) -> dict
```

Builds cash, positions, and position amounts from delivery data, then delegates to `evaluate_position()`.

---

## `Evaluator.evaluate_position`

Evaluate portfolio values, turnover, trades, and performance metrics.

```python
Evaluator.evaluate_position(
    positions: pd.DataFrame,       # Position quantities per instrument, indexed by time
    cash: pd.Series,              # Cash series indexed by time
    prices: pd.DataFrame,         # Price table (same instruments as positions)
    position_amount: pd.DataFrame = None,  # Signed cash flows per instrument/time
    benchmark: pd.Series = None,
    fund: pd.Series = None,       # Capital base over time (e.g., transfers)
) -> dict
```

**Returns:** Dictionary with keys `values` (DataFrame), `positions` (DataFrame), `trades` (DataFrame), `evaluation` (Series).

---

## `Evaluator.evaluate_index`

Simulate an index strategy with periodic allocation.

```python
Evaluator.evaluate_index(
    weights: pd.DataFrame,        # Target weights per instrument over time (rows=time)
    prices: pd.DataFrame,         # Price table indexed by time
    freq: int = 1,                # Number of sub-schedules per period
    benchmark: pd.Series = None,
    commission: float = 0.0005,   # Proportional commission per re-allocation
) -> dict
```

Splits evaluation into `freq` sub-schedules to approximate intra-period allocation. Returns same structure as `evaluate_position()`.

---

## `Evaluator.evaluate_rebalance`

Evaluate periodic rebalancing strategy returns.

```python
Evaluator.evaluate_rebalance(
    weights: pd.DataFrame,        # Target weights per instrument
    prices: pd.DataFrame,         # Price table indexed by time
    freq: int = 1,                # Rebalance frequency in periods
    benchmark: pd.Series = None,
    commission: float = 0.0005,   # Proportional commission per turnover
) -> dict
```

**Returns:** Dictionary with `values` (cumulative value series) and `evaluation` (metrics series).

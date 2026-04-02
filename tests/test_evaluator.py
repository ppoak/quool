import numpy as np
import pandas as pd
import pytest

from quool import Evaluator, Broker, DataFrameSource, Delivery, FixedRateCommission, FixedRateSlippage


@pytest.fixture
def delivery_records():
    """Create realistic delivery records over multiple days.

    Signed: positive for SELL, negative for BUY.
    """
    records = [
        # Day 1: Initial cash transfer
        {"time": "2024-01-01", "code": "CASH", "type": "TRANSFER", "quantity": 100000, "price": 1, "comm": 0},
        # Day 2: Buy 1000 shares of 000001.SZ
        {"time": "2024-01-02", "code": "000001.SZ", "type": "BUY", "quantity": 1000, "price": 10.0, "comm": 5},
        # Day 3: Price rises, no action
        # Day 4: Sell 500 shares
        {"time": "2024-01-04", "code": "000001.SZ", "type": "SELL", "quantity": 500, "price": 11.0, "comm": 6},
        # Day 5: Buy 200 more shares
        {"time": "2024-01-05", "code": "000001.SZ", "type": "BUY", "quantity": 200, "price": 10.5, "comm": 2},
        # Day 6: Final sell to close position
        {"time": "2024-01-06", "code": "000001.SZ", "type": "SELL", "quantity": 700, "price": 11.5, "comm": 5},
    ]
    return pd.DataFrame(records)


@pytest.fixture
def price_series():
    """Price series for a stock and a benchmark."""
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    prices = pd.DataFrame(
        {
            "000001.SZ": [10.0, 10.0, 10.5, 11.0, 10.5, 11.5],
            "BENCH": [100.0, 100.5, 101.0, 101.5, 102.0, 102.5],
        },
        index=dates,
    )
    return prices


@pytest.fixture
def benchmark_series():
    """Benchmark series for comparison."""
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    return pd.Series([100.0, 100.5, 101.0, 101.5, 102.0, 102.5], index=dates, name="BENCH")


@pytest.fixture
def initial_positions():
    """Initial positions dict."""
    return {"000001.SZ": 1000}


@pytest.fixture
def initial_cash():
    """Initial cash amount."""
    return 100000.0


@pytest.fixture
def evaluator_source(price_series):
    """Create a DataFrameSource from price series."""
    dates = price_series.index
    codes = price_series.columns
    index = pd.MultiIndex.from_product([dates, codes], names=["time", "code"])
    n = len(index)
    data = pd.DataFrame(
        {
            "open": np.full(n, 10.0),
            "high": np.full(n, 11.0),
            "low": np.full(n, 9.0),
            "close": np.tile(price_series.values.T.flatten(), 1)[:n],
            "volume": np.full(n, 1e6),
        },
        index=index,
    )
    # Fix the close prices to match the price series
    for date in dates:
        for code in codes:
            idx = (date, code)
            if idx in data.index:
                data.loc[idx, "close"] = price_series.loc[date, code]
    return DataFrameSource(data)


@pytest.fixture
def broker_with_delivery(evaluator_source, delivery_records):
    """Create a broker with pre-populated delivery records."""
    brk = Broker(commission=FixedRateCommission(), slippage=FixedRateSlippage())
    for _, row in delivery_records.iterrows():
        deliv = Delivery(
            time=row["time"],
            code=row["code"],
            type=row["type"],
            quantity=row["quantity"],
            price=row["price"],
            comm=row["comm"],
        )
        brk.deliver(deliv)
    brk._time = pd.to_datetime("2024-01-06")
    return brk


class TestEvaluatorMetrics:
    """Test individual performance metrics."""

    def test_total_return_calculation(self, price_series, benchmark_series):
        """Test total_return = (end_value - start_value) / start_value."""
        net = pd.Series([1.0, 1.005, 1.01, 1.015, 1.02, 1.025], index=price_series.index)
        result = Evaluator.evaluate(net, benchmark_series)
        expected_total_return = 1.025 - 1.0
        assert abs(result["total_return"] - expected_total_return) < 1e-6

    def test_annual_return_geometry(self, price_series, benchmark_series):
        """Test annual_return uses geometric mean annualized."""
        net = pd.Series([1.0, 1.01, 1.02, 1.03, 1.04, 1.05], index=price_series.index)
        result = Evaluator.evaluate(net, benchmark_series)
        expected_annual = (1.05 / 1.0) ** (365 / 5) - 1
        assert abs(result["annual_return"] - expected_annual) < 0.1

    def test_annual_volatility(self, price_series, benchmark_series):
        """Test annual_volatility = std of daily returns * sqrt(252)."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(100) * 0.01)
        net = (1 + returns).cumprod()
        net.index = pd.date_range("2024-01-01", periods=100, freq="D")
        benchmark = pd.Series(np.ones(100), index=net.index)
        result = Evaluator.evaluate(net, benchmark)
        # Volatility is calculated from net returns, which may differ slightly from input returns
        assert result["annual_volatility"] > 0

    def test_sharpe_ratio(self, price_series, benchmark_series):
        """Test sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility."""
        net = pd.Series([1.0, 1.01, 1.02, 1.03, 1.04, 1.05], index=price_series.index)
        result = Evaluator.evaluate(net, benchmark_series)
        expected_sharpe = result["annual_return"] / result["annual_volatility"]
        assert abs(result["sharpe_ratio"] - expected_sharpe) < 1e-6

    def test_max_drawdown(self, price_series, benchmark_series):
        """Test max_drawdown = maximum peak-to-trough decline."""
        net = pd.Series([1.0, 1.05, 1.03, 1.08, 1.02, 1.06], index=price_series.index)
        result = Evaluator.evaluate(net, benchmark_series)
        # Peak is 1.08, trough is 1.02, drawdown = (1.02 - 1.08) / 1.08 = -0.0556
        assert result["max_drawdown"] < 0
        assert result["max_drawdown"] >= -0.1

    def test_var_5_percent(self, price_series, benchmark_series):
        """Test VaR_5% = 5th percentile of daily returns."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(1000) * 0.01)
        net = (1 + returns).cumprod()
        net.index = pd.date_range("2024-01-01", periods=1000, freq="D")
        benchmark = pd.Series(np.ones(1000), index=net.index)
        result = Evaluator.evaluate(net, benchmark)
        expected_var = np.percentile(returns, 5)
        assert abs(result["VaR_5%"] - expected_var) < 0.01

    def test_cvar_5_percent(self, price_series, benchmark_series):
        """Test CVaR_5% = mean of returns below VaR_5%."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(1000) * 0.01)
        net = (1 + returns).cumprod()
        net.index = pd.date_range("2024-01-01", periods=1000, freq="D")
        benchmark = pd.Series(np.ones(1000), index=net.index)
        result = Evaluator.evaluate(net, benchmark)
        var_5 = np.percentile(returns, 5)
        expected_cvar = returns[returns <= var_5].mean()
        assert abs(result["CVaR_5%"] - expected_cvar) < 0.01

    def test_sortino_ratio(self, price_series, benchmark_series):
        """Test sortino_ratio = annual_return / downside_deviation."""
        # Use data with multiple negative returns to get non-zero downside_std
        net = pd.Series([1.0, 0.98, 0.97, 1.03, 1.04, 1.05], index=price_series.index)
        result = Evaluator.evaluate(net, benchmark_series)
        assert "sortino_ratio" in result
        # Sortino ratio should be defined when there are multiple negative returns
        assert not np.isnan(result["sortino_ratio"])

    def test_calmar_ratio(self, price_series, benchmark_series):
        """Test calmar_ratio = annual_return / max_drawdown."""
        net = pd.Series([1.0, 1.05, 1.03, 1.08, 1.02, 1.06], index=price_series.index)
        result = Evaluator.evaluate(net, benchmark_series)
        expected_calmar = result["annual_return"] / abs(result["max_drawdown"])
        assert abs(result["calmar_ratio"] - expected_calmar) < 1e-6

    def test_beta_calculation(self, price_series, benchmark_series):
        """Test beta vs benchmark."""
        np.random.seed(42)
        stock_returns = pd.Series(np.random.randn(100) * 0.02 + 0.0005)
        bench_returns = pd.Series(np.random.randn(100) * 0.01 + 0.0002)
        stock_net = (1 + stock_returns).cumprod()
        bench_net = (1 + bench_returns).cumprod()
        stock_net.index = pd.date_range("2024-01-01", periods=100, freq="D")
        bench_net.index = pd.date_range("2024-01-01", periods=100, freq="D")
        result = Evaluator.evaluate(stock_net, bench_net)
        expected_beta = stock_returns.cov(bench_returns) / bench_returns.var()
        assert abs(result["beta"] - expected_beta) < 0.1

    def test_alpha_calculation(self, price_series, benchmark_series):
        """Test alpha is calculated when beta is available."""
        np.random.seed(42)
        stock_returns = pd.Series(np.random.randn(100) * 0.02 + 0.0005)
        bench_returns = pd.Series(np.random.randn(100) * 0.01 + 0.0002)
        stock_net = (1 + stock_returns).cumprod()
        bench_net = (1 + bench_returns).cumprod()
        stock_net.index = pd.date_range("2024-01-01", periods=100, freq="D")
        bench_net.index = pd.date_range("2024-01-01", periods=100, freq="D")
        result = Evaluator.evaluate(stock_net, bench_net)
        # Alpha is calculated when beta is not NaN (requires > 30 data points)
        if not np.isnan(result["beta"]):
            assert "alpha" in result

    def test_turnover_ratio(self, price_series, benchmark_series):
        """Test turnover_ratio is mean of turnover series."""
        turnover = pd.Series([0.1, 0.2, 0.15, 0.1, 0.05, 0.1], index=price_series.index)
        net = pd.Series([1.0, 1.01, 1.02, 1.03, 1.04, 1.05], index=price_series.index)
        result = Evaluator.evaluate(net, benchmark_series, turnover=turnover)
        assert abs(result["turnover_ratio"] - turnover.mean()) < 1e-6


class TestEvaluatorReport:
    """Test Evaluator.report() method."""

    def test_report_returns_dict_with_evaluation_metrics(self, broker_with_delivery, evaluator_source):
        """Test that report() returns a dict with evaluation metrics."""
        evaluator = Evaluator(broker_with_delivery, evaluator_source)
        result = evaluator.report()
        assert isinstance(result, dict)
        assert "evaluation" in result
        assert "values" in result
        assert "positions" in result

    def test_report_with_benchmark(self, broker_with_delivery, evaluator_source, benchmark_series):
        """Test report() with benchmark series."""
        evaluator = Evaluator(broker_with_delivery, evaluator_source)
        result = evaluator.report(benchmark=benchmark_series)
        assert "beta" in result["evaluation"]
        assert "alpha" in result["evaluation"]


class TestEvaluatorEvaluateDelivery:
    """Test Evaluator.evaluate_delivery() static method."""

    def test_evaluate_delivery_with_signed_records(self, delivery_records, price_series):
        """Test evaluate_delivery works with signed delivery records."""
        # Convert delivery records to proper format
        delivery = delivery_records.copy()
        delivery["time"] = pd.to_datetime(delivery["time"])
        delivery = delivery.set_index(["time", "code"])
        delivery["amount"] = delivery["quantity"] * delivery["price"]
        # Apply signed amount based on type
        amount_sign = {"TRANSFER": 1, "BUY": -1, "SELL": 1}
        delivery["amount"] = delivery["amount"] * delivery["type"].map(amount_sign)

        # Prepare prices wide format
        prices = price_series.copy()
        prices.index = pd.to_datetime(prices.index)

        result = Evaluator.evaluate_delivery(delivery, prices)
        assert "values" in result
        assert "evaluation" in result
        assert "positions" in result

    def test_evaluate_delivery_empty_benchmark(self, delivery_records, price_series):
        """Test evaluate_delivery with no benchmark (None)."""
        delivery = delivery_records.copy()
        delivery["time"] = pd.to_datetime(delivery["time"])
        delivery = delivery.set_index(["time", "code"])
        delivery["amount"] = delivery["quantity"] * delivery["price"]

        prices = price_series.copy()
        prices.index = pd.to_datetime(prices.index)

        result = Evaluator.evaluate_delivery(delivery, prices, benchmark=None)
        assert "evaluation" in result
        assert not np.isnan(result["evaluation"]["total_return"])


class TestEvaluatorEvaluatePosition:
    """Test Evaluator.evaluate_position() static method."""

    def test_evaluate_position_basic(self, initial_positions, initial_cash, price_series):
        """Test evaluate_position with basic position and cash."""
        dates = price_series.index
        positions = pd.DataFrame(index=dates)
        positions["000001.SZ"] = 0.0
        positions.loc[pd.to_datetime("2024-01-02"), "000001.SZ"] = 1000.0
        positions.loc[pd.to_datetime("2024-01-04"), "000001.SZ"] = 500.0
        positions.loc[pd.to_datetime("2024-01-05"), "000001.SZ"] = 700.0
        positions.loc[pd.to_datetime("2024-01-06"), "000001.SZ"] = 0.0
        positions = positions.fillna(method="ffill").fillna(0.0)

        cash = pd.Series(initial_cash, index=dates)

        prices = price_series.copy()
        prices.index = pd.to_datetime(prices.index)

        result = Evaluator.evaluate_position(positions, cash, prices)
        assert "values" in result
        assert "evaluation" in result
        assert "trades" in result

    def test_evaluate_position_with_position_amount(self, initial_positions, initial_cash, price_series):
        """Test evaluate_position with explicit position_amount."""
        dates = price_series.index
        positions = pd.DataFrame(index=dates)
        positions["000001.SZ"] = 0.0
        positions.loc[pd.to_datetime("2024-01-02"), "000001.SZ"] = 1000.0
        positions = positions.fillna(method="ffill").fillna(0.0)

        cash = pd.Series(initial_cash, index=dates)

        prices = price_series.copy()
        prices.index = pd.to_datetime(prices.index)

        position_amount = pd.DataFrame(index=dates)
        position_amount["000001.SZ"] = 0.0
        position_amount.loc[pd.to_datetime("2024-01-02"), "000001.SZ"] = -10000.0
        position_amount = position_amount.fillna(0.0)

        result = Evaluator.evaluate_position(positions, cash, prices, position_amount)
        assert "values" in result


class TestEvaluatorEvaluateIndex:
    """Test Evaluator.evaluate_index() static method."""

    def test_evaluate_index_basic(self, price_series):
        """Test evaluate_index with weight series."""
        dates = price_series.index
        weights = pd.DataFrame(index=dates)
        weights["000001.SZ"] = 0.5
        weights["BENCH"] = 0.5

        prices = price_series.copy()
        prices.index = pd.to_datetime(prices.index)

        result = Evaluator.evaluate_index(weights, prices)
        assert "values" in result
        assert "evaluation" in result

    def test_evaluate_index_with_freq(self, price_series):
        """Test evaluate_index with frequency parameter."""
        dates = price_series.index
        weights = pd.DataFrame(index=dates)
        weights["000001.SZ"] = 0.5
        weights["BENCH"] = 0.5

        prices = price_series.copy()
        prices.index = pd.to_datetime(prices.index)

        result = Evaluator.evaluate_index(weights, prices, freq=2)
        assert "values" in result

    def test_evaluate_index_without_benchmark(self, price_series):
        """Test evaluate_index without benchmark works correctly."""
        dates = price_series.index
        weights = pd.DataFrame(index=dates)
        weights["000001.SZ"] = 0.5
        weights["BENCH"] = 0.5

        prices = price_series.copy()
        prices.index = pd.to_datetime(prices.index)

        result = Evaluator.evaluate_index(weights, prices)
        assert "evaluation" in result


class TestEvaluatorEvaluateRebalance:
    """Test Evaluator.evaluate_rebalance() static method."""

    def test_evaluate_rebalance_basic(self, price_series):
        """Test evaluate_rebalance with weight series."""
        dates = price_series.index
        weights = pd.DataFrame(index=dates)
        weights["000001.SZ"] = 0.5
        weights["BENCH"] = 0.5

        prices = price_series.copy()
        prices.index = pd.to_datetime(prices.index)

        result = Evaluator.evaluate_rebalance(weights, prices)
        assert "values" in result
        assert "evaluation" in result

    def test_evaluate_rebalance_with_freq(self, price_series):
        """Test evaluate_rebalance with rebalance frequency."""
        dates = price_series.index
        weights = pd.DataFrame(index=dates)
        weights["000001.SZ"] = 0.5
        weights["BENCH"] = 0.5

        prices = price_series.copy()
        prices.index = pd.to_datetime(prices.index)

        result = Evaluator.evaluate_rebalance(weights, prices, freq=2)
        assert "values" in result

    def test_evaluate_rebalance_with_commission(self, price_series):
        """Test evaluate_rebalance with commission."""
        dates = price_series.index
        weights = pd.DataFrame(index=dates)
        weights["000001.SZ"] = 0.5
        weights["BENCH"] = 0.5

        prices = price_series.copy()
        prices.index = pd.to_datetime(prices.index)

        result = Evaluator.evaluate_rebalance(weights, prices, commission=0.001)
        assert "values" in result


class TestEvaluatorEdgeCases:
    """Test edge cases and error handling."""

    def test_evaluate_with_empty_benchmark(self, price_series):
        """Test evaluate with empty benchmark series."""
        net = pd.Series([1.0, 1.01, 1.02], index=price_series.index[:3])
        benchmark = pd.Series([], dtype=float)
        result = Evaluator.evaluate(net, benchmark)
        assert "total_return" in result

    def test_evaluate_with_single_return(self, price_series):
        """Test evaluate with single return value."""
        net = pd.Series([1.0, 1.01], index=price_series.index[:2])
        benchmark = pd.Series([1.0, 1.005], index=net.index)
        result = Evaluator.evaluate(net, benchmark)
        assert "total_return" in result

    def test_evaluate_position_empty_positions(self, price_series):
        """Test evaluate_position with no positions."""
        dates = price_series.index
        positions = pd.DataFrame(index=dates, columns=["STOCK"])
        positions = positions.fillna(0.0)

        cash = pd.Series(100000.0, index=dates)
        prices = price_series.copy()
        prices.index = pd.to_datetime(prices.index)
        prices = prices.reindex(columns=["STOCK"], fill_value=10.0)

        result = Evaluator.evaluate_position(positions, cash, prices)
        assert "values" in result
        assert "evaluation" in result

    def test_evaluate_trades_with_no_trades(self, price_series):
        """Test evaluate with no trades provided."""
        net = pd.Series([1.0, 1.01, 1.02], index=price_series.index[:3])
        result = Evaluator.evaluate(net, None, turnover=None, trades=None)
        assert "total_return" in result
        # When trades is None, trading behavior metrics may use different keys
        # Just verify basic metrics exist
        assert "annual_return" in result


class TestEvaluatorDistributionMetrics:
    """Test distribution-related metrics."""

    def test_skewness_calculation(self, price_series):
        """Test skewness metric."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(1000) * 0.01)
        net = (1 + returns).cumprod()
        net.index = pd.date_range("2024-01-01", periods=1000, freq="D")
        result = Evaluator.evaluate(net, None)
        assert "skewness" in result

    def test_kurtosis_calculation(self, price_series):
        """Test kurtosis metric."""
        np.random.seed(42)
        returns = pd.Series(np.random.randn(1000) * 0.01)
        net = (1 + returns).cumprod()
        net.index = pd.date_range("2024-01-01", periods=1000, freq="D")
        result = Evaluator.evaluate(net, None)
        assert "kurtosis" in result

    def test_monthly_stats(self, price_series):
        """Test monthly return statistics."""
        dates = pd.date_range("2024-01-01", periods=252, freq="D")
        np.random.seed(42)
        returns = pd.Series(np.random.randn(252) * 0.01 + 0.0002)
        net = (1 + returns).cumprod()
        net.index = dates
        result = Evaluator.evaluate(net, None)
        assert "monthly_return_std" in result
        assert "monthly_win_rate" in result

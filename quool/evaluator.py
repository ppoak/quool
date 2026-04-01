import numpy as np
import pandas as pd
from .order import Delivery
from .broker import Broker
from .source import Source


class Evaluator:
    """Portfolio and strategy performance evaluator.

    Evaluator aggregates broker deliveries and market data from a Source to
    compute portfolio values, positions, turnover, trade summaries, and a
    comprehensive set of performance metrics. It supports evaluation from raw
    delivery records, index-weight simulations, and periodic rebalancing.

    Attributes:
      broker (Broker): The broker providing deliveries, orders, and pending orders.
      source (Source): The market data provider.

    Notes:
      - The broker is expected to implement get_delivery(), get_orders(), and
        get_pendings(), returning pandas DataFrames.
      - The source is expected to provide market data including close prices
        in a structure usable for evaluation (see report()).
    """

    def __init__(self, broker: Broker, source: Source):
        """Initialize the evaluator with broker and source.

        Args:
          broker (Broker): Broker instance providing executions (deliveries) and order history.
          source (Source): Market data source with time series prices for instruments.

        Returns:
          None
        """
        self.broker = broker
        self.source = source

    def report(self, benchmark: pd.Series = None):
        """Produce a full evaluation report based on deliveries and market prices.

        This method:
          - Retrieves deliveries from the broker.
          - Transforms delivery amounts and quantities to signed values using
            Delivery.AMOUNT_SIGN and Delivery.QUANTITY_SIGN.
          - Builds a price table (wide format) for instruments.
          - Computes values, positions, trades, and evaluation metrics via evaluate_delivery().
          - Appends broker orders, pendings, and delivery logs.

        Args:
          benchmark (pandas.Series, optional): Benchmark net value series for comparison.
            If None or empty, a flat benchmark is used.

        Returns:
          dict: A dictionary containing:
            - values (pandas.DataFrame): Columns ['net', 'total', 'market', 'cash', 'turnover'] indexed by time.
            - positions (pandas.DataFrame): Position quantities per instrument over time.
            - trades (pandas.DataFrame): Trade-level summary with open/close timing and returns.
            - evaluation (pandas.Series): Performance metrics (see evaluate()).
            - orders (pandas.DataFrame): Historical orders from the broker.
            - pendings (pandas.DataFrame): Currently pending orders from the broker.
            - delivery (pandas.DataFrame): Delivery log from the broker.

        Raises:
          KeyError: If expected columns or indices are missing in delivery or price data.
          ValueError: If input series or DataFrames are not aligned in time.
        """
        delivery = self.broker.get_delivery()
        data = self.source.datas
        delivery = delivery.set_index(["time", "code"]).sort_index()
        delivery["amount"] *= delivery["type"].map(Delivery.AMOUNT_SIGN)
        delivery["quantity"] *= delivery["type"].map(Delivery.QUANTITY_SIGN)
        prices = data["close"].unstack("code")
        return {
            **self.evaluate_delivery(
                delivery=delivery, prices=prices, benchmark=benchmark
            ),
            "orders": self.broker.get_orders(),
            "pendings": self.broker.get_pendings(),
            "delivery": self.broker.get_delivery(),
        }

    @staticmethod
    def evaluate_delivery(
        delivery: pd.DataFrame, prices: pd.DataFrame, benchmark: pd.Series = None
    ):
        """Evaluate portfolio based on signed delivery records and price table.

        Computes cumulative cash, positions, and position cash flows from delivery
        data and delegates to evaluate_position() for value and metrics.

        Args:
          delivery (pandas.DataFrame): Delivery records indexed by ['time', 'code'].
            Must include columns:
              - 'type' (str): Delivery type (e.g., BUY/SELL/TRANSFER).
              - 'amount' (float): Notional cash amount (signed).
              - 'quantity' (float): Executed quantity (signed).
          prices (pandas.DataFrame): Wide table of prices with rows indexed by time
            and columns by instrument code, containing close prices.
          benchmark (pandas.Series, optional): Benchmark net value series for comparison.

        Returns:
          dict: See evaluate_position() return for structure.

        Raises:
          KeyError: If required columns are missing from delivery or prices.
        """
        # cash, position, total_value, market_value calculation
        fund = delivery[delivery["type"] == "TRANSFER"]["amount"].droplevel("code")
        cash = delivery.groupby("time")["amount"].sum().cumsum()
        positions = (
            delivery[delivery["type"] != "TRANSFER"]
            .groupby(["time", "code"])["quantity"]
            .sum()
            .unstack()
            .fillna(0)
            .cumsum()
        )
        position_amount = (
            delivery[delivery["type"] != "TRANSFER"]
            .groupby(["time", "code"])["amount"]
            .sum()
            .unstack()
            .fillna(0)
        )

        return Evaluator.evaluate_position(
            positions, cash, prices, position_amount, benchmark, fund
        )

    @staticmethod
    def evaluate_position(
        positions: pd.DataFrame,
        cash: pd.Series,
        prices: pd.DataFrame,
        position_amount: pd.DataFrame = None,
        benchmark: pd.Series = None,
        fund: pd.Series = None,
    ):
        """Evaluate portfolio values, turnover, trades, and performance metrics.

        Steps:
          1. Aligns and forward-fills positions, cash, and prices over the union of time indices.
          2. Computes market value (positions * prices) and total portfolio value (cash + market).
          3. Derives turnover from position changes (delta * prices) scaled by previous total.
          4. Constructs a synthetic fund (capital base) from transfers or initial cash to compute net value.
          5. Builds trade summaries (open/close times, amounts, duration, returns) by tracking cumulative
             position changes per instrument.
          6. Computes evaluation metrics via evaluate().

        Args:
          positions (pandas.DataFrame): Position quantities per instrument indexed by time.
          cash (pandas.Series): Cash series indexed by time.
          prices (pandas.DataFrame): Price table (same instruments as positions), indexed by time.
          position_amount (pandas.DataFrame, optional): Signed cash flows tied to position changes
            per instrument and time. If None, inferred from position deltas and prices.
          benchmark (pandas.Series, optional): Benchmark net value series for comparison.
          fund (pandas.Series, optional): Capital base over time (e.g., transfers). If None,
            inferred from initial cash.

        Returns:
          dict: A dictionary containing:
            - values (pandas.DataFrame): Columns ['net', 'total', 'market', 'cash', 'turnover'].
            - positions (pandas.DataFrame): Positions per instrument over time.
            - trades (pandas.DataFrame): Trade summary with columns
              ['open_at', 'open_amount', 'close_at', 'close_amount', 'duration', 'return'] and
              a multi-instrument index flattened to rows.
            - evaluation (pandas.Series): Performance metrics (see evaluate()).

        Raises:
          ValueError: If input series/dataframes have incompatible indices or missing data.
        """
        times = prices.index.union(cash.index).union(positions.index)
        cash = cash.reindex(times).ffill()
        prices = prices.reindex(times).ffill()
        positions = positions.reindex(times).ffill().fillna(0)
        market = (positions * prices).sum(axis=1)
        total = (cash + market).bfill()
        delta = positions.diff()
        delta.iloc[0] = positions.iloc[0]
        turnover = (delta * prices).abs().sum(axis=1) / total.shift(1).fillna(
            cash.iloc[0]
        )
        fund = (
            fund
            if isinstance(fund, pd.Series) and not fund.empty
            else cash.iloc[:1].reindex(times).fillna(0)
        )
        prod = fund / (total - fund)
        prod.iloc[0] = 0
        fund = (1 + prod).fillna(1).cumprod() * fund.iloc[0]
        net = total / fund

        position_amount = (
            position_amount
            if position_amount is not None
            else (-delta * prices).dropna(axis=1, how="all").fillna(0)
        )
        position_cumsum = delta.cumsum()
        position_trade_mark = (position_cumsum <= 1e-8) & (delta != 0)
        position_trade_num = position_trade_mark.shift(1).astype("bool").cumsum()
        trades = []
        for code in positions.columns:
            trades.append(
                position_cumsum[code]
                .groupby(position_trade_num[code], group_keys=False)
                .apply(
                    lambda x: pd.Series(
                        {
                            "open_at": (
                                (x.index[x > 0][0]) if x.index[x > 0].size else np.nan
                            ),
                            "open_amount": (
                                -position_amount[code][position_amount[code] < 0]
                                .loc[x.index.min() : x.index.max()]
                                .sum()
                                if x.index[x > 0].size
                                else np.nan
                            ),
                            "close_at": (
                                (x.index[x < 1e-8][-1])
                                if x.index[x < 1e-8].max() > x.index[x > 0].min()
                                else np.nan
                            ),
                            "close_amount": (
                                position_amount[code][position_amount[code] > 0]
                                .loc[x.index.min() : x.index.max()]
                                .sum()
                                if x.index[x < 1e-8].max() > x.index[x > 0].min()
                                else np.nan
                            ),
                            "duration": (
                                x.index.get_indexer_for([x.index[x < 1e-8][-1]])[0]
                                - x.index.get_indexer_for([x.index[x > 0][0]])[0]
                                if x.index[x > 0].size
                                and x.index[x < 1e-8].max() > x.index[x > 0].min()
                                else np.nan
                            ),
                        }
                    )
                    .to_frame(f"{code}#{x.name}")
                    .T
                )
            )

        if not trades:
            trades = pd.DataFrame(columns=["open_at", "close_at", "open_amount", "close_amount", "duration", "return"])
        else:
            trades = pd.concat(trades).dropna(subset="open_at")
            trades["return"] = trades["close_amount"] / trades["open_amount"] - 1
        return {
            "values": pd.concat(
                [net, total, market, cash, turnover],
                axis=1,
                keys=["net", "total", "market", "cash", "turnover"],
            ),
            "positions": positions,
            "trades": trades.reset_index(),
            "evaluation": Evaluator.evaluate(net, benchmark, turnover, trades),
        }

    @staticmethod
    def evaluate_index(
        weights: pd.DataFrame,
        prices: pd.DataFrame,
        freq: int = 1,
        benchmark: pd.Series = None,
        commission: float = 0.0005,
    ):
        """Simulate an index strategy with periodic allocation updating and transaction costs.

        Splits the evaluation into freq sub-schedules to approximate intra-period
        allocation. For each sub-schedule:
          - Computes cash residual (1 - sum(weights)).
          - Converts weights into shares at current prices.
          - Accumulates share holdings, applying commission on re-allocation.
        Aggregates across sub-schedules and evaluates via evaluate_position().

        Args:
          weights (pandas.DataFrame): Target weights per instrument over time (rows=time, columns=code).
            Row sums should be <= 1 (remaining goes to cash).
          prices (pandas.DataFrame): Price table indexed by time with instrument columns.
          freq (int, optional): Number of sub-schedules per period. Defaults to 1.
          benchmark (pandas.Series, optional): Benchmark net value series. Defaults to None.
          commission (float, optional): Proportional commission applied to re-allocations. Defaults to 0.0005.

        Returns:
          dict: Same structure as evaluate_position().

        Raises:
          ValueError: If weights and prices cannot be aligned or contain invalid values.
        """
        share = pd.DataFrame(
            np.zeros_like(weights), index=weights.index, columns=weights.columns
        )
        cash = pd.Series(
            np.zeros_like(weights.index), index=weights.index, dtype="float"
        )
        for f in range(freq):
            _weight = weights.iloc[f::freq]
            _cash = 1 - _weight.sum(axis=1)
            _share = (_weight / prices).dropna(axis=0, how="all").fillna(0)
            for i, t in enumerate(_share.index[1:], start=1):
                _share.iloc[i] = (
                    ((_share.iloc[i - 1] * prices.loc[t]).sum() + _cash.iloc[i - 1])
                    * _share.iloc[i]
                    / (1 + commission)
                )
            _share = _share.reindex(prices.index, method="ffill").fillna(0)
            _cash = _cash.reindex(prices.index, method="ffill").fillna(1)
            share = share + _share
            cash = cash + _cash
        return Evaluator.evaluate_position(
            share,
            cash,
            prices,
            benchmark,
        )

    @staticmethod
    def evaluate_rebalance(
        weights: pd.DataFrame,
        prices: pd.DataFrame,
        freq: int = 1,
        benchmark: pd.Series = None,
        commission: float = 0.0005,
    ):
        """Evaluate periodic rebalancing strategy returns and performance.

        Assumes rebalancing occurs every 'freq' periods:
          - Downsamples weights to the rebalance schedule.
          - Computes turnover from weight changes and applies commission.
          - Estimates future returns based on shifted prices over the rebalance interval.
          - Calculates cumulative value of the strategy.

        Args:
          weights (pandas.DataFrame): Target weights per instrument over time (rows=time, columns=code).
          prices (pandas.DataFrame): Price table indexed by time with instrument columns.
          freq (int, optional): Rebalance frequency in periods. Defaults to 1.
          benchmark (pandas.Series, optional): Benchmark net value series. Defaults to None.
          commission (float, optional): Proportional commission applied to turnover. Defaults to 0.0005.

        Returns:
          dict: A dictionary containing:
            - values (pandas.Series): Cumulative value of the strategy.
            - evaluation (pandas.Series): Performance metrics (see evaluate()).

        Raises:
          ValueError: If weights or prices are not compatible for rebalancing evaluation.
        """
        weights = weights.iloc[::freq]
        delta = weights.diff()
        delta.iloc[0] = weights.iloc[0]
        turnover = delta.abs().sum(axis=1)

        shifted = prices.shift(-1)
        future = shifted.shift(-freq) / shifted - 1
        future = future.loc[future.index.intersection(weights.index)]

        returns = (weights * future).sum(axis=1) - turnover * commission
        value = (1 + returns).cumprod()
        return {
            "values": value,
            "evaluation": Evaluator.evaluate(value, benchmark, turnover, None),
        }

    @staticmethod
    def evaluate(
        net: pd.Series,
        benchmark: pd.Series = None,
        turnover: pd.Series = None,
        trades: pd.DataFrame = None,
    ):
        """Compute performance, risk, and distribution metrics from a net value series.

        Metrics computed:
          - Basic performance: total_return, annual_return, annual_volatility, sharpe_ratio,
            calmar_ratio, sortino_ratio.
          - Drawdown: max_drawdown, max_drawdown_period.
          - Risk: VaR_5%, CVaR_5%.
          - Activity: turnover_ratio (if turnover provided).
          - Benchmark-relative: beta, alpha, excess_return, excess_volatility, information_ratio.
          - Trading behavior (if trades provided): position_duration, trade_win_rate, trade_return.
          - Distribution: skewness, kurtosis, day_return_win_rate.
          - Monthly stats: monthly_return_std, monthly_win_rate.

        Args:
          net (pandas.Series): Net value series indexed by time (starting near 1.0).
          benchmark (pandas.Series, optional): Benchmark value series. If None or empty,
            a flat benchmark is assumed.
          turnover (pandas.Series, optional): Turnover ratio per period. Defaults to None.
          trades (pandas.DataFrame, optional): Trade summary with open/close amounts and durations.
            If provided, trading behavior metrics are computed. Defaults to None.

        Returns:
          pandas.Series: Labeled performance metrics.

        Raises:
          ValueError: If net has an invalid index or insufficient data for metrics.
        """
        returns = net.pct_change(fill_method=None).fillna(0)
        drawdown = net / net.cummax() - 1
        max_drawdown = drawdown.min()
        max_drawdown_end = drawdown.idxmin()
        max_drawdown_start = drawdown.loc[:max_drawdown_end][
            drawdown.loc[:max_drawdown_end] == 0
        ]
        if max_drawdown_start.empty:
            max_drawdown_start = max_drawdown_end
        else:
            max_drawdown_start = max_drawdown_start.index[-1]

        # Benchmark Comparison Metrics
        if benchmark is None or benchmark.empty:
            benchmark = pd.Series(np.ones_like(net), index=net.index)
        benchmark_returns = benchmark.pct_change(fill_method=None).fillna(0)
        excess_returns = returns - benchmark_returns
        excess_value = (1 + excess_returns).cumprod()

        evaluation = pd.Series(name="evaluation")
        # Basic Performance Metrics
        evaluation["total_return"] = net.iloc[-1] - 1
        evaluation["annual_return"] = (
            (
                (1 + evaluation["total_return"])
                ** (365 / (net.index[-1] - net.index[0]).days)
                - 1
            )
            if (net.index[-1] - net.index[0]).days != 0
            else np.nan
        )
        evaluation["annual_volatility"] = returns.std() * np.sqrt(252)
        evaluation["sharpe_ratio"] = (
            evaluation["annual_return"] / evaluation["annual_volatility"]
            if evaluation["annual_volatility"] != 0
            else np.nan
        )
        evaluation["calmar_ratio"] = (
            evaluation["annual_return"] / abs(max_drawdown)
            if max_drawdown != 0
            else np.nan
        )
        downside_std = returns[returns < 0].std()
        evaluation["sortino_ratio"] = (
            evaluation["annual_return"] / (downside_std * np.sqrt(252))
            if downside_std != 0
            else np.nan
        )

        # Risk Metrics
        evaluation["max_drawdown"] = max_drawdown
        evaluation["max_drawdown_period"] = (
            drawdown.index.get_indexer_for([max_drawdown_end])[0]
            - drawdown.index.get_indexer_for([max_drawdown_start])[0]
        )
        var_95 = np.percentile(returns, 5)
        evaluation["VaR_5%"] = var_95
        cvar_95 = returns[returns <= var_95].mean()
        evaluation["CVaR_5%"] = cvar_95

        # Turnover Ratio
        if turnover is not None:
            evaluation["turnover_ratio"] = turnover.mean()
        else:
            evaluation["turnover_ratio"] = np.nan

        # Alpha and Beta, Benchmark related
        if returns.count() > 30:
            beta = (
                returns.cov(benchmark_returns) / benchmark_returns.var()
                if benchmark_returns.var() != 0
                else np.nan
            )
        else:
            beta = np.nan
        evaluation["beta"] = beta
        evaluation["alpha"] = (
            (returns.mean() - beta * benchmark_returns.mean()) * 252
            if beta is not np.nan
            else np.nan
        )
        evaluation["excess_return"] = excess_value.dropna().iloc[-1] - 1
        evaluation["excess_volatility"] = excess_returns.std() * np.sqrt(252)
        tracking_error = (returns - benchmark_returns).std() * np.sqrt(252)
        evaluation["information_ratio"] = (
            evaluation["excess_return"] / tracking_error
            if tracking_error != 0
            else np.nan
        )

        # Trading behavior
        if trades is not None and not trades.empty:
            evaluation["position_duration"] = trades["duration"].mean()
            profit = trades["close_amount"] - trades["open_amount"]
            evaluation["trade_win_rate"] = (
                profit[profit > 0].count() / profit.count()
                if profit.count() != 0
                else np.nan
            )
            evaluation["trade_return"] = profit.sum() / trades["open_amount"].sum()
        else:
            evaluation["position_duration(days)"] = np.nan
            evaluation["trade_win_rate"] = np.nan
            evaluation["trade_return"] = np.nan

        # Distribution Metrics
        evaluation["skewness"] = returns.skew()
        evaluation["kurtosis"] = returns.kurtosis()
        positive_returns = returns[
            returns.gt(0 if benchmark is None else benchmark_returns)
        ].count()
        evaluation["day_return_win_rate"] = positive_returns / returns.count()
        monthly_returns = net.resample("ME").last().pct_change(fill_method=None).fillna(0)
        evaluation["monthly_return_std"] = monthly_returns.std()
        evaluation["monthly_win_rate"] = (monthly_returns > 0).sum() / len(
            monthly_returns
        )
        return evaluation

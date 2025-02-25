import numpy as np
import pandas as pd


def report(delivery: pd.DataFrame, data: pd.DataFrame):
    delivery = delivery.set_index(["time", "code"]).sort_index()
    prices = data["close"].unstack("code")

    # cash, position, trades, total_value, market_value calculation
    cash = delivery.groupby("time")["amount"].sum().cumsum()
    positions = (
        delivery.drop(index="CASH", level=1)
        .groupby(["time", "code"])["quantity"]
        .sum()
        .unstack()
        .fillna(0)
        .cumsum()
    )
    timepoints = prices.index.union(cash.index).union(positions.index)
    cash = cash.reindex(timepoints).ffill()
    positions = positions.reindex(timepoints).ffill().fillna(0)
    market = (positions * prices).sum(axis=1)
    total = cash + market
    delta = positions.diff()
    delta.iloc[0] = positions.iloc[0]
    turnover = (delta * prices).abs().sum(axis=1) / total.shift(1).fillna(cash.iloc[0])

    delivery = delivery.drop(index="CASH", level=1)
    delivery["stock_cumsum"] = delivery.groupby("code")["quantity"].cumsum()
    delivery["trade_mark"] = delivery["stock_cumsum"] == 0
    delivery["trade_num"] = (
        delivery.groupby("code")["trade_mark"]
        .shift(1)
        .astype("bool")
        .groupby("code")
        .cumsum()
    )
    trades = delivery.groupby(["code", "trade_num"]).apply(
        lambda x: pd.Series(
            {
                "open_amount": -x[x["quantity"] > 0]["amount"].sum(),
                "open_at": x[x["quantity"] > 0].index.get_level_values("time")[0],
                "close_amount": (
                    x[x["quantity"] < 0]["amount"].sum()
                    if x["quantity"].sum() == 0
                    else np.nan
                ),
                "close_at": (
                    x[x["quantity"] < 0].index.get_level_values("time")[-1]
                    if x["quantity"].sum() == 0
                    else np.nan
                ),
            }
        )
    )
    if not trades.empty:
        trades["duration"] = pd.to_datetime(trades["close_at"]) - pd.to_datetime(
            trades["open_at"]
        )
        trades["return"] = (trades["close_amount"] - trades["open_amount"]) / trades[
            "open_amount"
        ]
    else:
        trades = pd.DataFrame(
            columns=[
                "open_amount",
                "open_at",
                "close_amount",
                "close_at",
                "duration",
                "return",
            ]
        )
    return {
        "values": pd.concat(
            [total, market, cash, turnover],
            axis=1,
            keys=["total", "market", "cash", "turnover"],
        ),
        "positions": positions,
        "trades": trades,
    }


def evaluate(
    value: pd.Series,
    benchmark: pd.Series = None,
    turnover: pd.Series = None,
    trades: pd.DataFrame = None,
):
    """Evaluate the performance of a trading strategy.

    Args:
        value (pd.Series): The value of the portfolio.
        benchmark (pd.Series, optional): The value of the benchmark. Defaults to None.
        turnover (pd.Series, optional): The turnover rate. Defaults to None.
        trades (pd.Series): The trading records.

    Returns:
        dict: A dictionary containing the evaluation results.
    """
    net_value = value / value.iloc[0]
    returns = net_value.pct_change(fill_method=None).fillna(0)
    drawdown = net_value / net_value.cummax() - 1
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
    if benchmark is None:
        benchmark = pd.Series(np.ones_like(net_value), index=net_value.index)
    benchmark_returns = benchmark.pct_change().fillna(0)
    excess_returns = returns - benchmark_returns

    evaluation = pd.Series(name="evaluation")
    # Basic Performance Metrics
    evaluation["total_return(%)"] = (net_value.iloc[-1] - 1) * 100
    evaluation["annual_return(%)"] = (
        (
            (1 + evaluation["total_return(%)"] / 100)
            ** (365 / (net_value.index[-1] - net_value.index[0]).days)
            - 1
        )
        * 100
        if (net_value.index[-1] - net_value.index[0]).days != 0
        else np.nan
    )
    evaluation["annual_volatility(%)"] = (returns.std() * np.sqrt(252)) * 100
    evaluation["sharpe_ratio"] = (
        evaluation["annual_return(%)"] / evaluation["annual_volatility(%)"]
        if evaluation["annual_volatility(%)"] != 0
        else np.nan
    )
    evaluation["calmar_ratio"] = (
        evaluation["annual_return(%)"] / abs(max_drawdown * 100)
        if max_drawdown != 0
        else np.nan
    )
    downside_std = returns[returns < 0].std()
    evaluation["sortino_ratio(%)"] = (
        evaluation["annual_return(%)"] / (downside_std * np.sqrt(252))
        if downside_std != 0
        else np.nan
    )

    # Risk Metrics
    evaluation["max_drawdown(%)"] = max_drawdown * 100
    evaluation["max_drawdown_period"] = max_drawdown_end - max_drawdown_start
    var_95 = np.percentile(returns, 5) * 100
    evaluation["VaR_5%(%)"] = var_95
    cvar_95 = returns[returns <= var_95 / 100].mean() * 100
    evaluation["CVaR_5%(%)"] = cvar_95

    # Turnover Ratio
    if turnover is not None:
        evaluation["turnover_ratio(%)"] = turnover.mean() * 100
    else:
        evaluation["turnover_ratio(%)"] = np.nan

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
    evaluation["alpha(%)"] = (
        (returns.mean() - beta * benchmark_returns.mean()) * 252 * 100
        if beta is not np.nan
        else np.nan
    )
    evaluation["excess_return(%)"] = excess_returns.mean() * 252 * 100
    evaluation["excess_volatility(%)"] = excess_returns.std() * np.sqrt(252) * 100
    tracking_error = (returns - benchmark_returns).std() * np.sqrt(252)
    evaluation["information_ratio"] = (
        evaluation["excess_return(%)"] / tracking_error
        if tracking_error != 0
        else np.nan
    )

    # Trading behavior
    if trades is not None and not trades.empty:
        evaluation["position_duration(days)"] = trades["duration"].mean()
        profit = trades["close_amount"] - trades["open_amount"]
        evaluation["trade_win_rate(%)"] = (
            profit[profit > 0].count() / profit.count() * 100
            if profit.count() != 0
            else np.nan
        )
        evaluation["trade_return(%)"] = profit.sum() / trades["open_amount"].sum() * 100
    else:
        evaluation["position_duration(days)"] = np.nan
        evaluation["trade_win_rate(%)"] = np.nan
        evaluation["trade_return(%)"] = np.nan

    # Distribution Metrics
    evaluation["skewness"] = returns.skew()
    evaluation["kurtosis"] = returns.kurtosis()
    positive_returns = returns[
        returns.gt(0 if benchmark is None else benchmark_returns)
    ].count()
    evaluation["day_return_win_rate(%)"] = (positive_returns / returns.count()) * 100
    monthly_returns = net_value.resample("ME").last().pct_change().fillna(0)
    evaluation["monthly_return_std(%)"] = monthly_returns.std() * 100
    evaluation["monthly_win_rate(%)"] = (
        (monthly_returns > 0).sum() / len(monthly_returns) * 100
    )
    return evaluation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from .table import ItemTable


class TradeRecorder(ItemTable):

    def __init__(
        self, 
        uri: str | Path, 
        principle: float = None, 
        start_date: str | pd.Timestamp = None,
    ):
        super().__init__(uri, True)
        if not self.fragments:
            if principle is None or start_date is None:
                raise ValueError("principle and start_date must be specified when initiating")
            self.add(pd.DataFrame([{
                "datetime": pd.to_datetime(start_date), 
                "code": "cash", "size": float(principle), 
                "price": 1.0, "amount": float(principle), "commission": 0.0
            }], index=[pd.to_datetime('now')]))
        
        if not self.columns.isin(["datetime", "code", "size", "price", "amount", "commission"]).all():
            raise ValueError("The table must have columns datetime, code, size, price, amount, commission")
    
    @property
    def spliter(self):
        return pd.Grouper(key='datetime', freq='M')

    @property
    def namer(self):
        return lambda x: x['datetime'].iloc[0].strftime('%Y%m')
    
    def add_trade(
        self, 
        date: str | pd.Timestamp,
        code: str,
        size: float = None,
        price: float = None,
        amount: float = None,
        commission: float = 0,
        **kwargs,
    ):
        if size is None and price is None and amount is None:
            raise ValueError("two of size, price or amount must be specified")
        size = size or (amount / price)
        price = price or (amount / size)
        amount = amount or (size * price)

        trade = pd.DataFrame([{
            "datetime": pd.to_datetime(date),
            "code": code, "size": size,
            "price": price, "amount": amount,
            "commission": commission, **kwargs
        }], index=[pd.to_datetime('now')])
        if code != "cash":
            cash = pd.DataFrame([{
                "datetime": pd.to_datetime(date),
                "code": "cash", "size": -size * price - commission,
                "price": 1, "commission": 0,
                "amount": -size * price - commission, **kwargs
            }], index=[pd.to_datetime('now')])
            trade = pd.concat([trade, cash], axis=0)
        
        self.update(trade)

    def peek(self, date: str | pd.Timestamp = None) -> pd.Series:
        df = self.read(filters=[("datetime", "<=", pd.to_datetime(date))])
        return df.groupby("code")[["size", "amount", "commission"]].sum()
        
    def report(
        self, 
        price: pd.DataFrame, 
        code_level: int | str = 0,
        date_level: int | str = 1,
    ) -> pd.DataFrame:
        if price.index.nlevels == 1:
            code_level = code_level if not isinstance(code_level, int) else "code"
            date_level = date_level if not isinstance(date_level, int) else "datetime"
            price = price.stack().sort_index().to_frame("price")
            price.index.names = [date_level, code_level]
        price = price.reorder_levels([code_level, date_level]).sort_index()
        dates = price.index.get_level_values(date_level).unique()

        data = self.read(["datetime", "code", "size", "amount", "commission"],
            filters=[("datetime", "<=", dates.max())])
        data = data.groupby(["code", "datetime"]).sum()
        data = data.groupby("code").apply(lambda x: x.droplevel('code').reindex(
            dates.union(x.index.get_level_values('datetime').unique().sort_values())
        ).fillna(0).cumsum())

        cash = data.loc["cash", "amount"]
        noncash = data.drop(labels="cash", axis=0)
        noncash.index.names = [price.index.names[code_level], price.index.names[date_level]]
        market = price.mul(noncash['size'], axis=0).groupby(level=date_level).sum()
        market = market.reindex(cash.index).fillna(0)

        data = pd.concat([market.add_suffix("_market"), cash.to_frame("cash")], axis=1)
        data = pd.concat([data, market.add_suffix("_value").add(cash, axis=0)], axis=1)
        return data

    @staticmethod
    def evaluate(
        value: pd.Series, 
        cash: pd.Series = None,
        benchmark: pd.Series = None,
        image: str = None,
        result: str = None,
    ):
        cash = cash if isinstance(cash, (pd.Series, pd.DataFrame)) else \
            pd.Series(np.zeros(value.shape[0]), index=value.index)
        market = value - cash
        returns = value.pct_change(fill_method=None).fillna(0)
        if benchmark is not None:
            benchmark = benchmark.squeeze()
            benchmark_returns = benchmark.pct_change(fill_method=None).fillna(0)
        drawdown = value / value.cummax() - 1
        turnover = (market.diff() / value.shift(1)).fillna(0)
        
        # evaluation indicators
        evaluation = pd.Series(name='evaluation')
        evaluation['total_return(%)'] = (value.iloc[-1] / value.iloc[0] - 1) * 100
        evaluation['annual_return(%)'] = ((evaluation['total_return(%)'] / 100 + 1) ** (
            365 / (value.index.max() - value.index.min()).days) - 1) * 100
        evaluation['annual_volatility(%)'] = (returns.std() * np.sqrt(252)) * 100
        down_volatility = (returns[returns < 0].std() * np.sqrt(252)) * 100
        maxdate = drawdown.idxmax()
        startdate = drawdown.loc[:maxdate][drawdown.loc[:maxdate] == 0].index[-1]
        evaluation['max_drawdown(%)'] = (drawdown.max()) * 100
        evaluation['max_drawdown_period(days)'] = maxdate - startdate
        evaluation['max_drawdown_start'] = startdate
        evaluation['max_drawdown_stop'] = maxdate
        evaluation['daily_turnover(%)'] = turnover.mean() * 100
        evaluation['sharpe_ratio'] = evaluation['annual_return(%)'] / evaluation['annual_volatility(%)'] \
            if evaluation['annual_volatility(%)'] != 0 else np.nan
        evaluation['sortino_ratio'] = evaluation['annual_return(%)'] / down_volatility \
            if down_volatility != 0 else np.nan
        evaluation['calmar_ratio'] = evaluation['annual_return(%)'] / evaluation['max_drawdown(%)'] \
            if evaluation['max_drawdown(%)'] != 0 else np.nan
        if benchmark is not None:
            exreturns = returns - benchmark_returns
            benchmark_volatility = (benchmark_returns.std() * np.sqrt(252)) * 100
            exvalue = (1 + exreturns).cumprod()
            evaluation['total_exreturn(%)'] = (exvalue.iloc[-1] - exvalue.iloc[0]) * 100
            evaluation['annual_exreturn(%)'] = ((evaluation['total_exreturn(%)'] / 100 + 1
                ) ** (365 / (exvalue.index.max() - exvalue.index.min()).days) - 1) * 100
            evaluation['annual_exvolatility(%)'] = (exreturns.std() * np.sqrt(252)) * 100
            evaluation['beta'] = returns.cov(benchmark_returns) / benchmark_returns.var()
            evaluation['alpha(%)'] = (returns.mean() - (evaluation['beta'] * (benchmark_returns.mean()))) * 100
            evaluation['treynor_ratio(%)'] = (evaluation['annual_exreturn(%)'] / evaluation['beta'])
            evaluation['information_ratio'] = evaluation['annual_exreturn(%)'] / benchmark_volatility \
                if benchmark_volatility != 0 else np.nan

        data = pd.concat([value, cash, returns, drawdown, turnover], 
            axis=1, keys=['value', 'cash', 'returns', 'drawdown', 'turnover'])
        if benchmark is not None:
            data = pd.concat([data, exreturns.to_frame('exreturns'),
                exvalue.to_frame('exvalue')], axis=1)
        
        if result is not None:
            data.to_excel(result, sheet_name="performances")
        
        if image is not None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
            data[["value", "cash"]].plot(ax=ax, title="Portfolio", color=['black', 'red'])
            data[["returns", "drawdown", "turnover"]].plot(ax=ax, alpha=0.7, 
                secondary_y=True, label=['returns', "drawdown", 'turnover'],
                color=["blue", "purple", "yellow"])
            if isinstance(image, (str, Path)):
                fig.savefig(image)

        return data

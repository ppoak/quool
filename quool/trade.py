import numpy as np
import pandas as pd
from pathlib import Path
from .core import Table
from .util import parse_commastr


def strategy(
    weight: pd.DataFrame, 
    price: pd.DataFrame, 
    delay: int = 1,
    side: str = 'both',
    commission: float = 0.005,
) -> dict[str, pd.Series]:
    # compute turnover and commission
    delta = weight - weight.shift(1).fillna(0)
    if side == "both":
        turnover = delta.abs().sum(axis=1) / 2
    elif side == "long":
        turnover = delta.where(delta > 0).abs().sum(axis=1)
    elif side == "short":
        turnover = delta.where(delta < 0).abs().sum(axis=1)
    commission *= turnover

    # compute the daily return
    price = price.shift(-delay).loc[weight.index]
    returns = price.shift(-1) / price - 1
    returns = (weight * returns).sum(axis=1)
    returns -= commission
    value = (1 + returns).cumprod()
    value.name = 'value'
    turnover.name = 'turnover'

    return {'value': value, 'turnover': turnover}

def evaluate(
    value: pd.Series, 
    turnover: pd.Series = None,    
    benchmark: pd.Series = None,
) -> pd.Series:
    returns = value.pct_change().fillna(0)
    if benchmark is not None:
        benchmark = benchmark.squeeze()
        benchmark_returns = benchmark.pct_change(fill_method=None).fillna(0)
    
    # evaluation indicators
    evaluation = pd.Series(name='evaluation')
    evaluation['total_return(%)'] = (value.iloc[-1] / value.iloc[0] - 1) * 100
    evaluation['annual_return(%)'] = ((evaluation['total_return(%)'] / 100 + 1) ** (252 / value.shape[0]) - 1) * 100
    evaluation['annual_volatility(%)'] = (returns.std() * np.sqrt(252)) * 100
    down_volatility = (returns[returns < 0].std() * np.sqrt(252)) * 100
    cumdrawdown = -(value / value.cummax() - 1)
    maxdate = cumdrawdown.idxmax()
    startdate = cumdrawdown.loc[:maxdate][cumdrawdown.loc[:maxdate] == 0].index[-1]
    evaluation['max_drawdown(%)'] = (cumdrawdown.max()) * 100
    evaluation['max_drawdown_period(days)'] = maxdate - startdate
    evaluation['max_drawdown_start'] = startdate
    evaluation['max_drawdown_stop'] = maxdate
    evaluation['daily_turnover(%)'] = turnover.mean() * 100 if turnover is not None else np.nan
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
        evaluation['annual_exreturn(%)'] = ((evaluation['total_exreturn(%)'] / 100 + 1) ** (252 / exvalue.shape[0]) - 1) * 100
        evaluation['annual_exvolatility(%)'] = (exreturns.std() * np.sqrt(252)) * 100
        evaluation['beta'] = returns.cov(benchmark_returns) / benchmark_returns.var()
        evaluation['alpha(%)'] = (returns.mean() - (evaluation['beta'] * (benchmark_returns.mean()))) * 100
        evaluation['treynor_ratio(%)'] = (evaluation['annual_exreturn(%)'] / evaluation['beta'])
        evaluation['information_ratio'] = evaluation['annual_exreturn(%)'] / benchmark_volatility \
            if benchmark_volatility != 0 else np.nan
    
    return evaluation


class TradeTable(Table):

    def __init__(
        self, 
        uri: str | Path, 
        principle: float = None, 
        start_date: str | pd.Timestamp = None,
    ):
        if not Path(uri).exists():
            if principle is None or start_date is None:
                raise ValueError("principle and start_date must be specified when initiating")
            pd.DataFrame([{
                "datetime": pd.to_datetime(start_date), 
                "code": "cash", "size": principle, 
                "price": 1, "amount": principle, "commission": 0
            }]).to_parquet(uri)
        super().__init__(uri, False)
        if not self.columns.isin(["datetime", "code", "size", "price", "amount", "commission"]).all():
            raise ValueError("The table must have columns datetime, code, size, price, amount, commission")
            
    @property
    def fragments(self):
        return [self.path]
    
    @property
    def minfrag(self):
        return self.path
    
    @property
    def spliter(self):
        return super().spliter

    @property
    def namer(self):
        return super().namer
    
    def __str__(self):
        return str(self.read())
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def read(self, date: str | pd.Timestamp = None):
        if date is not None:
            date = [("datetime", "<=", pd.to_datetime(date))]
        return super().read(None, date)
    
    def update(
        self, 
        date: str | pd.Timestamp,
        code: str,
        size: int = None,
        price: float = None,
        amount: float = None,
        commission: float = 0,
        **kwargs,
    ):
        pd.concat([self.read(), pd.DataFrame([{
            "datetime": pd.to_datetime(date),
            "code": code, "size": size,
            "price": price, "amount": price * size,
            "commission": commission, **kwargs
        }])] + ([] if code == "cash" else [pd.DataFrame([{
            "datetime": pd.to_datetime(date),
            "code": "cash", "size": -size * price - commission,
            "price": 1, "commission": 0,
            "amount": -size * price - commission, **kwargs
        }])]), axis=0, ignore_index=True).sort_index().to_parquet(self.path)

    def peek(self, date: str | pd.Timestamp = None) -> pd.Series:
        df = self.read(date)
        return df.groupby("code")[["size", "amount", "commission"]].sum()
        
    def report(
        self, 
        price: pd.DataFrame, 
        benchmark: pd.Series = None
    ) -> pd.DataFrame:
        values = []
        cashes = []
        for date in price.index:
            pr = price.loc[date]
            ps = self.peek(date)
            val = pr.loc[pr.index.intersection(ps.index)].mul(ps["size"], axis=0)
            val.loc["cash"] = ps.loc["cash", "size"]
            values.append(val.sum())
            cashes.append(val.loc["cash"])
        
        value = pd.Series(values, index=price.index)
        cash = pd.Series(cashes, index=price.index)
        drawdown = value / value.cummax() - 1
        turnover_rate = (value.diff() / value.shift(1)).abs()
        info = pd.concat(
            [value, cash, drawdown, turnover_rate], axis=1, 
            keys=["value", "cash", "drawdown", "turnover_rate"]
        )
        evaluation = evaluate(value, turnover_rate, benchmark=benchmark)
        return info, evaluation
            

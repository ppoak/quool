import numpy as np
import pandas as pd
from .order import Order
from .source import Source
from .broker import Broker
from .util import evaluate
from joblib import Parallel, delayed


class Strategy:

    def __init__(
        self, 
        id: str,
        source: Source,
        broker: Broker,
    ):
        self.id = id
        self.source = source
        self.broker = broker
    
    def init(self, **kwargs):
        pass

    def update(self, **kwargs):
        raise NotImplementedError("`update` method must be implemented")
    
    def stop(self, **kwargs):
        pass

    def run(self, **kwargs):
        data = self.source.update()
        if data is None:
            return False
        self.broker.update(time=self.source.time, data=data)
        self.update(**kwargs)
        return True

    def backtest(self, **kwargs):
        self.init(**kwargs)
        while True:
            if not self.run(**kwargs):
                break
        self.stop(**kwargs)

    def __call__(
        self, 
        params: dict | list[dict], 
        since: pd.Timestamp = None, 
        n_jobs: int = -1
    ):
        Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(self.backtest)(param, path, since) for path, param in params.items()
        )
    
    def __str__(self) -> str:
        return (
            f"{self.__class__}({self.id})@{self.status}\n"
            f"Broker:\n{self.broker}\n"
            f"Source:\n{self.source}\n"
        )

    def __repr__(self):
        return self.__str__()

    def log(self, message: str, level: str = "DEBUG"):
        if self.logger is not None:
            self.logger.log(level, f"[{self.manager.time}]: {message}")
        else:
            print(f"[{self.manager.time}]: {message}")

    def get_value(self):
        return self.broker.get_value(self.source.data)
    
    def get_positions(self):
        return self.broker.get_positions()

    def buy(
        self,
        code: str,
        quantity: int,
        exectype: str,
        limit: float = None,
        trigger: float = None,
        id: str = None,
        valid: str = None,
    ) -> Order:
        return self.broker.create(
            side=self.broker.order_type.BUY,
            code=code,
            quantity=quantity,
            exectype=exectype,
            limit=limit,
            trigger=trigger,
            id=id,
            valid=valid,
        )

    def sell(
        self,
        code: str,
        quantity: int,
        exectype: str,
        limit: float = None,
        trigger: float = None,
        id: str = None,
        valid: str = None,
    ) -> Order:
        return self.broker.create(
            side=self.broker.order_type.SELL,
            code=code,
            quantity=quantity,
            exectype=exectype,
            limit=limit,
            trigger=trigger,
            id=id,
            valid=valid,
        )
    
    def order_target_value(
        self,
        code: str,
        value: float,
        exectype: str,
        limit: float = None,
        trigger: float = None,
        id: str = None,
        valid: str = None,
    ) -> Order:
        quantity = (value / self.source.data.loc[code, "close"] // 100) * 100
        if quantity > 0:
            side = self.broker.order_type.BUY
        else:
            side = self.broker.order_type.SELL
        return self.broker.create(
            side=side,
            code=code,
            quantity=abs(quantity),
            exectype=exectype,
            limit=limit,
            trigger=trigger,
            id=id,
            valid=valid,
        )

    def order_target_percent(
        self,
        code: str,
        value: float,
        exectype: str,
        limit: float = None,
        trigger: float = None,
        id: str = None,
        valid: str = None,
    ) -> Order:
        quantity = (value / self.source.data.loc[code, "close"] // 100) * 100
        if quantity > 0:
            side = self.broker.order_type.BUY
        else:
            side = self.broker.order_type.SELL
        return self.broker.create(
            side=side,
            code=code,
            quantity=abs(quantity),
            exectype=exectype,
            limit=limit,
            trigger=trigger,
            id=id,
            valid=valid,
        )

    def close(
        self,
        code: str,
        exectype: str,
        limit: float = None,
        trigger: float = None,
        id: str = None,
        valid: str = None,
    ) -> Order:
        return self.broker.create(
            side=self.broker.order_type.SELL,
            code=code,
            quantity=self.broker.positions.get(code),
            exectype=exectype,
            limit=limit,
            trigger=trigger,
            id=id,
            valid=valid,
        )

    def evaluate(self, benchmark: pd.Series = None):
        ledger = self.broker.ledger.set_index(["time", "code"]).sort_index()
        prices = self.source.datas["close"].unstack("code")

        # cash, position, trades, total_value, market_value calculation 
        cash = ledger.groupby("time")[["amount", "commission"]].sum()
        cash = (cash["amount"] - cash["commission"]).cumsum()
        positions = ledger.drop(index="CASH", level=1).groupby(["time", "code"])["unit"].sum().unstack().fillna(0).cumsum()
        timepoints = prices.index.union(cash.index).union(positions.index)
        cash = cash.reindex(timepoints).ffill()
        positions = positions.reindex(timepoints).ffill().fillna(0)
        market = (positions * prices).sum(axis=1)
        total = cash + market
        delta = positions.diff()
        delta.iloc[0] = positions.iloc[0]
        turnover = (delta * prices).abs().sum(axis=1) / total.shift(1).fillna(cash.iloc[0])
        
        ledger = ledger.drop(index="CASH", level=1)
        ledger["stock_cumsum"] = ledger.groupby("code")["unit"].cumsum()
        ledger["trade_mark"] = ledger["stock_cumsum"] == 0
        ledger["trade_num"] = ledger.groupby("code")["trade_mark"].shift(1).astype("bool").groupby("code").cumsum()
        trades = ledger.groupby(["code", "trade_num"]).apply(
            lambda x: pd.Series({
                "open_amount": -x[x["unit"] > 0]["amount"].sum(),
                "open_at": x[x["unit"] > 0].index.get_level_values("time")[0],
                "close_amount": x[x["unit"] < 0]["amount"].sum() if x["unit"].sum() == 0 else np.nan,
                "close_at": x[x["unit"] < 0].index.get_level_values("time")[-1] if x["unit"].sum() == 0 else np.nan,
            })
        )
        if not trades.empty:
            trades["duration"] = pd.to_datetime(trades["close_at"]) - pd.to_datetime(trades["open_at"])
            trades["return"] = (trades["close_amount"] - trades["open_amount"]) / trades["open_amount"]
        else:
            trades = pd.DataFrame(columns=["open_amount", "open_at", "close_amount", "close_at", "duration", "return"])
        return {
            "evaluation": evaluate(total, benchmark=benchmark, turnover=turnover, trades=trades),
            "values": pd.concat(
                [total, market, cash, turnover], 
                axis=1, keys=["total", "market", "cash", "turnover"]
            ),
            "positions": positions,
            "trades": trades,
        }
        
    def cancel(self, order_or_id: str | Order) -> Order:
        return self.broker.cancel(order_or_id)

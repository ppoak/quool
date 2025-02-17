import numpy as np
import pandas as pd
from uuid import uuid4
from typing import Type
from collections import deque
from abc import ABC, abstractmethod
from joblib import Parallel, delayed


class OrderBase(ABC):

    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOPLIMIT = "STOPLIMIT"

    BUY = "BUY"
    SELL = "SELL"

    def __init__(
        self,
        broker: 'BrokerBase',
        time: str,
        side: str,
        code: str,
        quantity: int,
        exectype: str = MARKET,
        limit: float = None,
        trigger: float = None,
        id: str = None,
        valid: str = None,
    ):
        self.id = id or str(uuid4())
        self.broker = broker
        self.creatime = pd.to_datetime(time)
        self.side = side
        self.code = code
        self.quantity = quantity
        self.exectype = exectype
        self.limit = limit
        self.trigger = trigger
        self.status = self.CREATED
        self.filled = 0
        self.value = 0
        self.exectime = None
        self.execprice = None
        self.commission = 0
        self.valid = pd.to_datetime(valid) if valid else None

    def execute(self, price: float, quantity: int) -> None:
        quantity = min(quantity, self.quantity - self.filled)

        self.execprice = price
        self.filled += quantity
        value = quantity * price
        self.value += value

        if self.filled == self.quantity:
            self.status = self.FILLED
            self.exectime = self.broker.time
        else:
            self.status = self.PARTIAL
            self.exectime = self.broker.time

    def cancel(self) -> None:
        if self.status in {self.CREATED, self.PARTIAL, self.SUBMITTED}:
            self.status = self.CANCELED

    def is_alive(self) -> bool:
        if self.status in {self.CREATED, self.SUBMITTED, self.PARTIAL}:
            if self.valid and self.broker.time > self.valid:
                self.status = self.EXPIRED
                return False
            return True
        return False

    def dump(self) -> dict:
        return {
            "id": self.id,
            "creatime": self.creatime.isoformat(),
            "side": self.side,
            "code": self.code,
            "quantity": self.quantity,
            "exectype": self.exectype,
            "limit": self.limit,
            "trigger": self.trigger,
            "execprice": self.execprice,
            "status": self.status,
            "filled": self.filled,
            "value": self.value,
            "exectime": self.exectime.isoformat() if self.exectime else None,
            "commission": self.commission,
            "valid": self.valid.isoformat() if self.valid else None,
        }

    @classmethod
    def load(cls, data: dict, broker: 'BrokerBase') -> 'OrderBase':
        order = cls(
            broker=broker,
            time=pd.to_datetime(data["creatime"]),
            side=data["side"],
            code=data["code"],
            quantity=data["quantity"],
            exectype=data.get("exectype", cls.MARKET),
            limit=data.get("limit", None),
            trigger=data.get("trigger", None),
            id=data["id"],
            valid=data.get("valid"),
        )
        # Load additional attributes
        order.status = data.get("status", cls.CREATED)
        order.filled = data.get("filled", 0)
        order.value = data.get("value", 0)
        order.exectime = pd.to_datetime(data["exetime"]) if data.get("exetime") else None
        order.execprice = data.get("exeprice")
        order.commission = data.get("commission", 0)
        
        return order

    def __str__(self) -> str:
        latest_date = self.creatime if self.exectime is None else self.exectime
        latest_price = (
            self.value / (self.filled or np.nan)
            if self.status in {self.FILLED, self.PARTIAL}
            else self.limit
        )
        latest_price = latest_price if latest_price else 0
        return (
            f"{self.__class__.__name__}({self.id[:5]})@{latest_date} [{self.status}]\n"
            f"{self.exectype} {self.side} {self.code} {self.quantity:.2f}x${latest_price:.2f})"
        )

    def __repr__(self) -> str:
        return self.__str__()


class CommissionBase(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, order: OrderBase, price: float, quantity: float) -> float:
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    def __repr__(self):
        return super().__repr__()


class SlippageBase(ABC):

    @abstractmethod
    def __init__(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, order: OrderBase, quantity: float, data: pd.DataFrame) -> float:
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError
    
    def __repr__(self):
        return super().__repr__()

class BrokerBase(ABC):

    def __init__(
        self,
        id: str,
        order_type: Type[OrderBase],
        commission: CommissionBase,
        slippage: SlippageBase,
    ):
        self.id = id or str(uuid4())
        self.order_type = order_type
        self.commission = commission
        self.slippage = slippage
        self._time = None
        self._balance = 0
        self._positions = {}
        self._pendings = deque()
        self._ledger = [] # Key parameter to restore the state of the broker
        self._orders = []  # History of processed orders
        self._ordict = {}

    @property
    def time(self) -> pd.Timestamp:
        return self._time

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def positions(self) -> dict:
        return self._positions

    @property
    def pendings(self) -> deque[OrderBase]:
        return self._pendings
    
    @property
    def orders(self) -> list[OrderBase]:
        return self._orders
    
    @property
    def ledger(self) -> list:
        return self._ledger

    def submit(self, order: OrderBase) -> None:
        self._pendings.append(order)
        self._ordict[order.id] = order
        order.status = order.SUBMITTED

    def create(
        self,
        side: str,
        code: str,
        quantity: int,
        exectype: str,
        limit: float = None,
        trigger: float = None,
        id: str = None,
        valid: str = None,
    ) -> OrderBase:
        if self._time is None:
            raise ValueError("broker must be initialized with a time")
        order = self.order_type(
            broker=self,
            time=self._time,
            side=side,
            code=code,
            quantity=quantity,
            limit=limit,
            trigger=trigger,
            ordtype=exectype,
            id=id,
            valid=valid,
        )
        self.submit(order)
        return order
    
    def cancel(self, order_or_id: str | OrderBase) -> OrderBase:
        if isinstance(order_or_id, str):
            order_or_id = self.get_order(order_or_id)
        order_or_id.cancel()
        return order_or_id

    @abstractmethod
    def update(self, time: str | pd.Timestamp, data: pd.DataFrame) -> None:
        raise NotImplementedError

    def get_value(self, data: pd.DataFrame) -> float:
        return (self.get_positions() * data["close"]).sum()
        
    def get_order(self, id: str) -> OrderBase:
        return self._orders.get[id]
    
    def get_ledger(self) -> pd.DataFrame:
        return pd.DataFrame(self.ledger)
    
    def get_pendings(self) -> pd.DataFrame:
        return pd.DataFrame([order.dump() for order in self.pendings])
    
    def get_orders(self) -> pd.DataFrame:
        return pd.DataFrame([order.dump() for order in self.orders]), 
    
    def get_positions(self) -> pd.Series:
        return pd.Series(self._positions, name="positions")
    
    def dump(self, since: pd.Timestamp) -> dict:
        since = pd.to_datetime(since or 0)
        ledger = self.get_ledger()
        orders = self.get_orders()
        pendings = self.get_pendings()

        if not self.ledger.empty:
            ledger["time"] = ledger["time"].dt.strftime('%Y-%m-%dT%H:%M:%S')
        
        return {
            "id": self.id,
            "balance": self.balance,
            "positions": self.positions,
            "ledger": (
                ledger[pd.to_datetime(ledger["time"]) >= since].replace(np.nan, None).to_dict(orient="records")
                if not ledger.empty else []
            ),
            "orders": (
                orders[pd.to_datetime(orders["creatime"]) >= since].replace(np.nan, None).to_dict(orient="records")
                if not orders.empty else []
            ),
            "pendings": pendings.replace(np.nan, None).to_dict(orient="records"),
            "commission": self.commission,
            "time": self._time.isoformat() if self._time else None,
        }
    
    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.id})@{self.time}\n"
            f"balance: ${self.balance:.2f}\n"
            f"commission: {self.commission}\n"
            f"slippage: {self.slippage}\n"
            f"#pending: {len(self.pendings)} & #over: {len(self.orders)}\n"
            f"position: {self.positions}\n"
        )
    
    def __repr__(self):
        return self.__str__()


class StrategyBase(ABC):

    # Status
    INIT = "INIT"
    RUNNING = "RUNNING"
    STOPPED = "STOPPED"

    def __init__(
        self, 
        id: str,
        source: SourceBase,
        broker: BrokerBase,
    ):
        self.id = id
        self.source = source
        self.broker = broker
        self.status = self.INIT
    
    def init(self, **kwargs):
        pass
    
    @abstractmethod
    def update(self, **kwargs):
        raise NotImplementedError("`update` method must be implemented")

    def stop(self, **kwargs):
        pass

    def run(self, **kwargs):
        data = self.source.update()
        if data is None:
            self.status = self.STOPPED
        
        self.broker.update(time=self.source.time, data=data)
        if self.status == self.INIT:
            self.init(**kwargs)
            self.status = self.RUNNING
        elif self.status == self.RUNNING:
            self.update(**kwargs)
        elif self.status == self.STOPPED:
            self.stop(**kwargs)

    def backtest(self, **kwargs):
        while True:
            if self.status == self.STOPPED:
                break
            self.run(**kwargs)

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

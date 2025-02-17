import numpy as np
import pandas as pd
from uuid import uuid4


class Order:

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

    def execute(self, time: str | pd.Timestamp, price: float, quantity: int) -> None:
        quantity = min(quantity, self.quantity - self.filled)

        self.execprice = price
        self.filled += quantity
        value = quantity * price
        self.value += value

        if self.filled == self.quantity:
            self.status = self.FILLED
        else:
            self.status = self.PARTIAL
        self.exectime = pd.to_datetime(time)

    def cancel(self) -> None:
        if self.status in {self.CREATED, self.PARTIAL, self.SUBMITTED}:
            self.status = self.CANCELED

    def is_alive(self, time: pd.Timestamp) -> bool:
        if self.status in {self.CREATED, self.SUBMITTED, self.PARTIAL}:
            if self.valid and pd.to_datetime(time) > self.valid:
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
    def load(cls, data: dict) -> 'Order':
        order = cls(
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

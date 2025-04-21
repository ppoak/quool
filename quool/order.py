import pandas as pd
from uuid import uuid4


class Delivery:

    AMOUNT_SIGN = {
        "TRANSFER": 1,
        "WITHDRAW": -1,
        "BUY": -1,
        "SELL": 1,
        "DIVIDEND": 1,
    }
    QUANTITY_SIGN = {
        "TRANSFER": 1,
        "WITHDRAW": -1,
        "BUY": 1,
        "SELL": -1,
        "SPLIT": 1,
    }

    def __init__(
        self,
        time: str,
        code: str,
        type: str,
        quantity: float,
        price: float,
        comm: float,
        id: str = None,
    ):
        self.id = id or str(uuid4())
        self.time = pd.to_datetime(time)
        self.type = type
        self.code = code
        self.quantity = quantity
        self.price = price
        self.comm = comm
        self.amount = quantity * price + self.QUANTITY_SIGN[type] * comm

    def dump(self) -> dict:
        return {
            "id": self.id,
            "time": self.time.isoformat(),
            "type": self.type,
            "code": self.code,
            "quantity": self.quantity,
            "price": self.price,
            "amount": self.amount,
            "commission": self.comm,
        }

    @classmethod
    def load(cls, data: dict) -> "Delivery":
        return cls(
            time=data["time"],
            type=data["type"],
            code=data["code"],
            quantity=data["quantity"],
            price=data["price"],
            comm=data["commission"],
            id=data["id"],
        )

    def __str__(self):
        return (
            f"{self.__class__.__name__}(#{self.id[:5]}@{self.time} "
            f"{self.type} {self.code} {self.quantity:.2f}x${self.price:.2f}-${self.comm:.2f}=${self.amount:.2f})"
        )

    def __repr__(self):
        return self.__str__()


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
        code: str,
        type: str,
        quantity: int,
        exectype: str = MARKET,
        limit: float = None,
        trigger: float = None,
        id: str = None,
        valid: str = None,
    ):
        self.create = pd.to_datetime(time)
        self.time = pd.to_datetime(time)
        self.code = code
        self.type = type
        self.quantity = quantity
        self.exectype = exectype
        self.limit = limit
        self.trigger = trigger
        self.price = 0
        self.status = self.CREATED
        self.filled = 0
        self.id = id or str(uuid4())
        self.valid = pd.to_datetime(valid) if valid else None
        self.delivery = []

    def __add__(self, delivery: Delivery) -> "Order":
        # disabling extra quantity execution
        if delivery.quantity > self.quantity - self.filled:
            raise ValueError("Delivery quantity exceeds order remaining")

        self.delivery.append(delivery)

        self.filled += delivery.quantity
        self.price = (
            self.price * (self.filled - delivery.quantity)
            + delivery.amount / self.filled
        )

        if self.filled == self.quantity:
            self.status = self.FILLED
        else:
            self.status = self.PARTIAL

        # latest time
        self.time = pd.to_datetime(delivery.time)
        return self

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

    def dump(self, delivery: bool = True) -> dict:
        data = {
            "id": self.id,
            "create": self.create.isoformat(),
            "time": self.time.isoformat(),
            "code": self.code,
            "type": self.type,
            "quantity": self.quantity,
            "exectype": self.exectype,
            "limit": self.limit,
            "trigger": self.trigger,
            "valid": self.valid.isoformat() if self.valid else None,
            "status": self.status,
            "filled": self.filled,
        }
        # with delivery
        if delivery:
            data["delivery"] = [deliv.dump() for deliv in self.delivery]
        return data

    @classmethod
    def load(cls, data: dict) -> "Order":
        order = cls(
            time=pd.to_datetime(data["creatime"]),
            code=data["code"],
            type=data["type"],
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
        if data.get("delivery") is not None:
            order.delivery = [Delivery.load(deliv) for deliv in data["delivery"]]

        return order

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(#{self.id[:5]}@{self.time} "
            f"{self.exectype} {self.type} {self.code} {self.quantity:.2f}x${self.price:.2f} [{self.status}])"
        )

    def __repr__(self) -> str:
        return self.__str__()

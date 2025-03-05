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
        amount: float,
        id: str = None,
    ):
        self.id = id or str(uuid4())
        self.time = pd.to_datetime(time)
        self.type = type
        self.code = code
        self.quantity = quantity
        self.amount = amount
        self.price = price

    def dump(self) -> dict:
        return {
            "id": self.id,
            "time": self.time.isoformat(),
            "code": self.code,
            "type": self.type,
            "quantity": self.quantity,
            "amount": self.amount,
            "price": self.price,
        }

    @classmethod
    def load(cls, data: dict) -> "Delivery":
        return cls(
            time=data["time"],
            code=data["code"],
            type=data["type"],
            quantity=data["quantity"],
            amount=data["amount"],
            price=data["price"],
            id=data["id"],
        )

    def __str__(self):
        return (
            f"{self.__class__.__name__}(#{self.id[:5]}@{self.time}"
            f"{self.type} {self.code} {self.quantity:.2f}x${self.price:.2f})"
        )

    def __repr__(self):
        return self.__str__()


class Order(Delivery):

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
        super().__init__(
            time=time, code=code, type=type, quantity=quantity, price=0, amount=0, id=id
        )
        self.creatime = self.time
        self.exectype = exectype
        self.limit = limit
        self.trigger = trigger
        self.status = self.CREATED
        self.filled = 0
        self.valid = pd.to_datetime(valid) if valid else None
        self.delivery = []

    def __add__(self, delivery: Delivery) -> "Order":
        # disabling extra quantity execution
        if delivery.quantity > self.quantity - self.filled:
            raise ValueError("Delivery quantity exceeds order remaining")

        self.delivery.append(delivery)
        self.amount += delivery.amount
        # latest price
        self.price = self.amount / (self.filled + delivery.quantity)
        self.filled += delivery.quantity

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
            "creatime": self.creatime.isoformat(),
            "time": self.time.isoformat(),
            "code": self.code,
            "type": self.type,
            "quantity": self.quantity,
            "exectype": self.exectype,
            "price": self.price,
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
        order.price = data.get("price", 0)
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

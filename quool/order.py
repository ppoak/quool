import pandas as pd
from uuid import uuid4


class Delivery:
    """Represents a single transaction record (execution fill or cash/position movement).

    A Delivery encapsulates one atomic event that affects holdings and/or cash,
    such as a BUY/SELL execution, TRANSFER, WITHDRAWAL, DIVIDEND, or SPLIT. It
    stores the executed quantity, price, commission, and computes the net cash
    amount associated with the event.

    Two sign maps are provided for convenience:
        - AMOUNT_SIGN: conventional cash-flow sign by event type.
        - QUANTITY_SIGN: sign used in quantity-related calculations, including the
            commission sign when computing the net amount.

    Attributes:
        id (str): Unique identifier for the delivery.
        time (pandas.Timestamp): Event timestamp parsed from the provided string.
        type (str): Delivery type (e.g., 'BUY', 'SELL', 'TRANSFER', 'WITHDRAW', 'DIVIDEND', 'SPLIT').
        code (str): Instrument identifier (e.g., ticker).
        quantity (float): Executed quantity (units or shares).
        price (float): Execution price per unit.
        comm (float): Commission associated with the event.
        amount (float): Net cash amount computed as quantity * price + QUANTITY_SIGN[type] * comm.
    """

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
        """Initialize a Delivery.

        Parses the provided time string into a pandas.Timestamp and computes the
        net cash amount as:
            amount = quantity * price + QUANTITY_SIGN[type] * comm

        Args:
            time (str): Event timestamp in a pandas-parsable format (e.g., '2024-03-01 10:00:00').
            code (str): Instrument identifier (e.g., ticker).
            type (str): Delivery type. Must be a key in QUANTITY_SIGN (e.g., 'BUY', 'SELL', 'TRANSFER', 'WITHDRAW', 'DIVIDEND', 'SPLIT').
            quantity (float): Executed quantity (units or shares).
            price (float): Execution price per unit.
            comm (float): Commission for the event.
            id (str, optional): Unique identifier. If not provided, a UUID4 string is generated.

        Raises:
            KeyError: If type is not present in QUANTITY_SIGN.
            ValueError: If time cannot be parsed into a timestamp.
        """
        self.id = id or str(uuid4())
        self.time = pd.to_datetime(time)
        self.type = type
        self.code = code
        self.quantity = quantity
        self.price = price
        self.comm = comm
        self.amount = quantity * price + self.QUANTITY_SIGN[type] * comm

    def dump(self) -> dict:
        """Serialize the Delivery to a JSON-friendly dictionary.

        Converts time to ISO-8601 string and returns a dictionary with standardized keys.

        Returns:
            dict: A dictionary with the following keys:
                - id (str)
                - time (str): ISO-8601 timestamp.
                - type (str)
                - code (str)
                - quantity (float)
                - price (float)
                - amount (float)
                - commission (float)
        """
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
        """Deserialize a Delivery from a dictionary.

        Constructs a Delivery from a dict with keys consistent with dump() output.

        Args:
            data (dict): Input dictionary with keys:
                - id (str)
                - time (str): ISO-8601 or pandas-parsable timestamp string.
                - type (str)
                - code (str)
                - quantity (float)
                - price (float)
                - commission (float)

        Returns:
            Delivery: The reconstructed Delivery instance.

        Raises:
            KeyError: If required fields are missing.
        """
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
    """Represents an order and its lifecycle, including partial fills and status transitions.

    An Order tracks creation time, instrument, side (BUY/SELL), quantity, execution type
    (MARKET/LIMIT/STOP/STOPLIMIT), price triggers/limits (if applicable), commissions, and
    delivery fills. It supports partial fills via Delivery objects and maintains effective
    prices and status transitions.

    Status constants:
        - CREATED
        - SUBMITTED
        - PARTIAL
        - FILLED
        - CANCELED
        - EXPIRED
        - REJECTED

    Execution type constants:
        - MARKET
        - LIMIT
        - STOP
        - STOPLIMIT

    Order side constants:
        - BUY
        - SELL

    Attributes:
        create (pandas.Timestamp): Creation timestamp of the order.
        time (pandas.Timestamp): Last updated timestamp (e.g., upon latest fill).
        code (str): Instrument identifier.
        type (str): Side of the order, 'BUY' or 'SELL'.
        quantity (int): Total requested quantity.
        exectype (str): Execution type (MARKET, LIMIT, STOP, STOPLIMIT).
        limit (float or None): Limit price for LIMIT/STOPLIMIT orders.
        trigger (float or None): Trigger price for STOP/STOPLIMIT orders.
        price_slip (float): Volume-weighted average execution price of fills (excluding commissions).
        price_eff (float): Effective average price of fills including commissions.
        comm (float): Total commissions accrued from fills.
        status (str): Current status of the order.
        filled (int): Total filled quantity so far.
        amount (float): Accumulated notional from fills (sum of fill.price * fill.quantity).
        id (str): Unique identifier for the order.
        valid (pandas.Timestamp or None): Good-till timestamp; order expires after this time.
        delivery (list[Delivery]): List of associated Delivery fills.
    """

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
        """Initialize an Order.

        Creates an order with the specified properties. The order starts in CREATED
        status with zero filled quantity, zero commissions, and empty delivery list.

        Args:
            time (str): Creation timestamp in a pandas-parsable format.
            code (str): Instrument identifier (e.g., ticker).
            type (str): Order side, either Order.BUY or Order.SELL.
            quantity (int): Total requested quantity.
            exectype (str, optional): Execution type (Order.MARKET, Order.LIMIT, Order.STOP, Order.STOPLIMIT). Defaults to Order.MARKET.
            limit (float, optional): Limit price for LIMIT or STOPLIMIT orders.
            trigger (float, optional): Trigger price for STOP or STOPLIMIT orders.
            id (str, optional): Unique identifier. If not provided, a UUID4 string is generated.
            valid (str, optional): Good-till timestamp in a pandas-parsable format. If None, the order does not expire.
        """
        self.create = pd.to_datetime(time)
        self.time = pd.to_datetime(time)
        self.code = code
        self.type = type
        self.quantity = quantity
        self.exectype = exectype
        self.limit = limit
        self.trigger = trigger
        self.price_slip = 0
        self.price_eff = 0
        self.comm = 0
        self.status = self.CREATED
        self.filled = 0
        self.amount = 0
        self.id = id or str(uuid4())
        self.valid = pd.to_datetime(valid) if valid else None
        self.delivery = []

    def __add__(self, delivery: Delivery) -> "Order":
        """Attach a Delivery fill to the order and update state.

        This operator allows using the + syntax to register a fill:
            order + delivery

        The method mutates the order in-place and returns self. It updates filled
        quantity, total amount, commissions, average prices (price_slip, price_eff),
        status (PARTIAL or FILLED), and the last update time.

        Args:
            delivery (Delivery): The delivery to apply. Its quantity must not exceed the order's remaining quantity.

        Returns:
            Order: The same order instance after mutation.

        Raises:
            ValueError: If delivery.quantity exceeds the remaining unfilled quantity.
        """
        # disabling extra quantity execution
        if delivery.quantity > self.quantity - self.filled:
            raise ValueError("Delivery quantity exceeds order remaining")

        self.delivery.append(delivery)

        self.filled += delivery.quantity
        self.amount += delivery.price * delivery.quantity
        self.comm += delivery.comm

        self.price_slip = self.amount / self.filled
        self.price_eff = (self.amount + self.comm) / self.filled

        if self.filled == self.quantity:
            self.status = self.FILLED
        else:
            self.status = self.PARTIAL

        # latest time
        self.time = pd.to_datetime(delivery.time)
        return self

    def cancel(self) -> None:
        """Cancel the order if it is still open.

        If the current status is CREATED, SUBMITTED, or PARTIAL, the status is set to
        CANCELED. Otherwise, the call has no effect.

        Returns:
            None
        """
        if self.status in {self.CREATED, self.PARTIAL, self.SUBMITTED}:
            self.status = self.CANCELED

    def is_alive(self, time: pd.Timestamp) -> bool:
        """Check whether the order is still active at a given time.

        An order is considered alive if its status is one of CREATED, SUBMITTED, or
        PARTIAL, and it has not expired. If a 'valid' timestamp is set and the provided
        time is later than 'valid', the order is marked EXPIRED and considered not alive.

        Args:
            time (pandas.Timestamp): The reference time to test, or any pandas-parsable input that can be converted to a Timestamp.

        Returns:
            bool: True if the order is active (and not expired), False otherwise.
        """
        if self.status in {self.CREATED, self.SUBMITTED, self.PARTIAL}:
            if self.valid and pd.to_datetime(time) > self.valid:
                self.status = self.EXPIRED
                return False
            return True
        return False

    def dump(self, delivery: bool = True) -> dict:
        """Serialize the Order to a JSON-friendly dictionary.

        Converts timestamps to ISO-8601 strings and includes essential fields for
        persistence or logging. Optionally includes serialized deliveries.

        Args:
            delivery (bool, optional): If True, include a 'delivery' key with a list of serialized Delivery objects. Defaults to True.

        Returns:
            dict: A dictionary with keys:
                - id (str)
                - create (str): ISO-8601 creation timestamp.
                - time (str): ISO-8601 last update timestamp.
                - code (str)
                - type (str)
                - quantity (int)
                - exectype (str)
                - limit (float or None)
                - trigger (float or None)
                - valid (str or None): ISO-8601 expiration timestamp if set.
                - status (str)
                - filled (int)
                - delivery (list[dict], optional): Present when delivery=True.
        """
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
        """Deserialize an Order from a dictionary.

        Constructs an Order from a dict, sets status and filled fields, and loads
        associated deliveries if present. Note: The creation time is read from the key
        'creatime'.

        Args:
            data (dict): Input dictionary with keys such as:
                - id (str)
                - creatime (str): Creation time in a pandas-parsable format.
                - code (str)
                - type (str)
                - quantity (int)
                - exectype (str, optional)
                - limit (float, optional)
                - trigger (float, optional)
                - valid (str, optional)
                - status (str, optional)
                - filled (int, optional)
                - delivery (list[dict], optional): List of serialized deliveries.

        Returns:
            Order: The reconstructed Order instance.

        Raises:
            KeyError: If required fields are missing.
        """

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
            f"{self.exectype} {self.type} {self.code} {self.quantity:.2f}x${self.price_eff:.2f}-${self.comm:.2f} [{self.status}])"
        )

    def __repr__(self) -> str:
        return self.__str__()

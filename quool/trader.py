import uuid
import queue
import logging
import numpy as np
import pandas as pd
from .tool import setup_logger
from .manager import ParquetManager


class Order:
    """Represents a financial order with attributes and state management."""

    # Order statuses
    CREATED = "CREATED"
    SUBMITTED = "SUBMITTED"
    PARTIAL = "PARTIAL"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"

    # Order types
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOPLIMIT = "STOPLIMIT"

    # Order sides
    BUY = 1
    SELL = -1

    def __init__(
        self,
        broker: 'Broker',
        code: str,
        quantity: int,
        limit: float = None,
        trigger: float = None,
        ordtype: str = MARKET,
        side: str = BUY,
        time: str = None,
        valid: str = None,
    ):
        """
        Initializes an Order instance.

        Args:
            code (str): The code of the financial instrument.
            quantity (int): The number of units for the order.
            trigger_price (float, optional): The price for triggering a limit order.
                Defaults to None for market orders.
            limit_price (float, optional): The price for limit orders. Defaults to None.
            order_type (str): The type of order ('MARKET' or 'LIMIT'). Defaults to 'MARKET'.
            side (str): The side of the order ('BUY' or 'SELL'). Defaults to 'BUY'.
            time (str, optional): The creation timestamp. Defaults to None.
            valid_before (str, optional): The expiry time for the order. Defaults to None.
        """
        self.ordid = str(uuid.uuid4())
        self.broker = broker
        self.code = code
        self.quantity = quantity
        self.filled = 0
        self.trigger = trigger
        self.limit = limit
        self.ordtype = ordtype
        self.side = side
        self.status = self.CREATED
        self.value = 0
        self.cretime = pd.to_datetime(time) if time else None
        self.exetime = None
        self.exeprice = None
        self.commission = 0
        self.valid = pd.to_datetime(valid) if valid else None

    def execute(self, price: float, quantity: int) -> None:
        """
        Executes the order partially or fully, updating its status and attributes.

        Args:
            time (str): The execution time.
            price (float): The execution price per unit.
            quantity (int): The number of units filled.

        Updates:
            - Fills the order partially or fully, updating cost and status.
        """
        quantity = min(quantity, self.quantity - self.filled)

        self.exeprice = price
        self.filled += quantity
        value = quantity * price
        self.value += value
        self.commission += value * self.broker.commission

        if self.filled == self.quantity:
            self.status = self.FILLED
            self.exetime = self.broker.time
        else:
            self.status = self.PARTIAL

    def cancel(self) -> None:
        """
        Cancels the order if it is not already fully executed.

        Args:
            time (str, optional): The cancellation timestamp. Defaults to None.

        Updates:
            - Sets the status to 'CANCELED'.
            - Sets the executed_at time if provided.
        """
        if self.status in {self.CREATED, self.PARTIAL, self.SUBMITTED}:
            self.status = self.CANCELED
            self.exetime = self.broker.time

    def is_alive(self) -> bool:
        """
        Checks if the order is still valid based on its expiration and status.

        Args:
            time (str, optional): The current time. Defaults to None.

        Returns:
            bool: True if the order is valid, False otherwise.
        """
        if self.status in {self.CREATED, self.SUBMITTED, self.PARTIAL}:
            if self.valid and self.broker.time > self.valid:
                self.status = self.EXPIRED
                self.exetime = self.broker.time
                return False
            return True
        return False

    def to_dict(self) -> dict:
        """
        Converts the order's attributes to a dictionary for structured representation.

        Returns:
            dict: A dictionary containing the order's attributes.
        """
        return {
            "OrdID": self.ordid,
            "Code": self.code,
            "Quantity": self.quantity,
            "Limit": self.limit,
            "Trigger": self.trigger,
            "ExePrice": self.exeprice,
            "OrdType": self.ordtype,
            "Side": self.side,
            "Status": self.status,
            "Filled": self.filled,
            "Value": self.value,
            "CreTime": self.cretime,
            "ExeTime": self.exetime,
            "Commission": self.commission,
            "Valid": self.valid,
        }

    def __str__(self) -> str:
        """
        Provides a string representation of the order for logging or debugging.

        Returns:
            str: A summary of the order details.
        """
        latest_date = self.cretime if self.exetime is None else self.exetime
        latest_price = (
            self.value / (self.filled or np.nan)
            if self.status in {self.FILLED, self.PARTIAL}
            else self.limit
        )
        latest_price = latest_price if latest_price else 0
        return (
            f"Order(#{self.ordid[:5]}@[{latest_date}] <{self.code}> "
            f"{self.side * self.quantity:.2f}x${latest_price:.2f} |{self.status}|)"
        )

    def __repr__(self) -> str:
        """
        Provides a detailed string representation for the object.

        Returns:
            str: The string representation of the order.
        """
        return self.__str__()


class Broker:
    """Abstract base class for a Broker."""

    def __init__(
        self, 
        manager: ParquetManager = None,
        principle: float = 1_000_000, 
        commission: float = 0.001,
        logger: logging.Logger = None,
    ):
        """
        Initializes the base broker with essential attributes.

        Args:
            principle (float): Initial cash balance for the broker. Defaults to 1,000,000.
            commission (float): Commission rate for transactions. Defaults to 0.001 (0.1%).
        """
        self.manager = manager
        self._time = None
        self._balance = principle
        self._positions = {}
        self.commission = commission
        self._pendings = queue.Queue()
        self._orders = []  # History of processed orders
        self._ordict = {}
        self.logger = logger or setup_logger("Broker", level="DEBUG")
    
    @property
    def time(self) -> pd.Timestamp:
        """
        Returns the current time of the broker.

        Returns:
            pd.Timestamp: The current time.
        """
        return self._time

    @property
    def balance(self) -> float:
        """
        Returns the current balance of the broker.

        Returns:
            float: The current balance.
        """
        return self._balance

    @property
    def positions(self) -> dict:
        """
        Returns the current positions held by the broker.

        Returns:
            dict: A dictionary of positions with stock codes as keys and quantities as values.
        """
        return self._positions
    
    @property
    def pendings(self) -> list:
        """
        Returns the list of orders submitted by the broker.

        Returns:
            list: A list of Order objects.
        """
        return self._pendings
    
    @property
    def orders(self) -> list:
        """
        Returns the list of processed orders.

        Returns:
            list: A list of Order objects.
        """
        return self._orders

    def buy(
        self,
        code: str,
        quantity: int,
        limit: float = None,
        trigger: float = None,
        exectype: str = Order.MARKET,
        valid: str = None,
    ) -> Order:
        """
        Creates and submits a buy order.

        Args:
            code (str): The stock code (e.g., "AAPL").
            quantity (int): The number of shares to buy.
            trigger_price (float, optional): The trigger price for the order. Defaults to None.
            limit_price (float, optional): The limit price for the order. Defaults to None.
            exectype (str): The execution type ('MARKET' or 'LIMIT'). Defaults to 'MARKET'.
            valid_before (str, optional): The validity period for the order. Defaults to None.

        Returns:
            Order: The created buy order.
        """
        if not self._time:
            raise ValueError("Current date is not set. Please call update() before placing orders.")

        order = Order(
            broker=self,
            code=code,
            quantity=quantity,
            limit=limit,
            trigger=trigger,
            ordtype=exectype,
            side=Order.BUY,
            time=self._time,
            valid=valid,
        )
        self.submit(order)
        return order

    def sell(
        self,
        code: str,
        quantity: int,
        limit: float = None,
        trigger: float = None,
        exectype: str = Order.MARKET,
        valid: str = None,
    ) -> Order:
        """
        Creates and submits a sell order.

        Args:
            code (str): The stock code (e.g., "AAPL").
            quantity (int): The number of shares to sell.
            trigger_price (float, optional): The trigger price for the order. Defaults to None.
            limit_price (float, optional): The limit price for the order. Defaults to None.
            exectype (str): The execution type ('MARKET' or 'LIMIT'). Defaults to 'MARKET'.
            valid_before (str, optional): The validity period for the order. Defaults to None.

        Returns:
            Order: The created sell order.
        """
        if not self._time:
            raise ValueError("Current date is not set. Please call update() before placing orders.")

        order = Order(
            broker=self,
            code=code,
            quantity=quantity,
            limit=limit,
            trigger=trigger,
            ordtype=exectype,
            side=Order.SELL,
            time=self._time,
            valid=valid,
        )
        self.submit(order)
        return order

    def submit(self, order: Order) -> None:
        """
        Submits an order for processing.

        Args:
            order (Order): The order to be submitted.
        """
        self._pendings.put(order)
        self._ordict[order.ordid] = order
        order.status = order.SUBMITTED
        self.logger.debug(f"Order submitted: {order}")

    def update(self, time: str) -> None:
        """
        Updates the broker's state for a new trading day.

        Args:
            time (str): The current trading day.
        """
        self._time = pd.to_datetime(time)
        self._pendings.put(None)  # Placeholder for end-of-day processing.
        data = self._load()
        order = self._pendings.get()
        while order is not None:
            self._match(order, data)
            if not order.is_alive():
                self._orders.append(order)
            else:
                self._pendings.put(order)
            order = self._pendings.get()

    def _load(self) -> pd.DataFrame:
        """
        Loads market data required for processing orders.

        Returns:
            pd.DataFrame: A DataFrame containing market data (open, high, low, close, volume) indexed by stock codes.
        """
        data = self.manager.read(
            index=["code"],
            date=self._time,
            columns=["open", "high", "low", "close", "volume"],
            code__in=[order.code if order else "" for order in list(self._pendings.queue)],
        )
        return data

    def _match(self, order: Order, data: pd.DataFrame) -> None:
        """
        Matches an order with market data and determines the execution price and quantity.

        Args:
            order (Order): The order to be matched.
            data (pd.DataFrame): The market data containing open, high, low, close, and volume information.
        """
        if order.code not in data.index:
            self.logger.error(f"Lacking data: {order}")
            return
        
        if order.trigger is not None:
            pricetype = "low" if order.side == order.BUY else "high"
            if (
                (order.ordtype == order.STOP or order.ordtype == order.STOPLIMIT) and 
                data.loc[order.code, pricetype] * order.side <= order.trigger * order.side
            ):
                # order triggered
                order.trigger = None
            else:
                return
        
        if order.ordtype == order.MARKET or order.ordtype == order.STOP:
            price = data.loc[order.code, "open"]
            quantity = min(data.loc[order.code, "volume"], order.quantity - order.filled)
        elif order.ordtype == order.LIMIT or order.ordtype == order.STOPLIMIT:
            if order.side == order.BUY and data.loc[order.code, "low"] <= order.limit:
                price = order.limit
                quantity = min(data.loc[order.code, "volume"], order.quantity - order.filled)
            elif order.side == order.SELL and data.loc[order.code, "high"] >= order.limit:
                price = order.limit
                quantity = min(data.loc[order.code, "volume"], order.quantity - order.filled)
            else:
                return
        
        self._execute(order, price, quantity)

    def _execute(self, order: Order, price: float, quantity: int) -> None:
        """
        Executes an order and updates broker's balance and positions.

        Args:
            order (Order): The order to be executed.
            execute_price (float): The price at which the order is executed.
            execute_quantity (int): The quantity of shares executed.
        """
        if order.side == order.BUY:
            cost = price * quantity * (1 + self.commission)
            if cost > self._balance:
                order.status = order.REJECTED
                order.exetime = self._time
                self.logger.warning(f"Insufficient balance: {order}")
            else:
                order.execute(price, quantity)
                self._balance -= cost
                self._positions[order.code] = self._positions.get(order.code, 0) + quantity
                self.logger.info(f"Order executed: {order}")
        elif order.side == order.SELL:
            revenue = price * quantity * (1 - self.commission)
            if self._positions.get(order.code, 0) < quantity:
                order.status = order.REJECTED
                order.exetime = self._time
                self.logger.warning(f"Insufficient position: {order}")
            else:
                order.execute(price, quantity)
                order.commission = price * quantity * self.commission
                self._balance += revenue
                self._positions[order.code] -= quantity
                if self._positions[order.code] == 0:
                    del self._positions[order.code]
                self.logger.info(f"Order executed: {order}")

    def get_order(self, ordid: str) -> Order:
        """
        Retrieves an order by its order ID.

        Args:
            ordid (str): The order ID of the order to retrieve.

        Returns:
            Order: The order with the specified order ID.
        """
        return self._ordict[ordid]

    def get_transactions(self, alive: bool = False) -> pd.DataFrame:
        """
        Compiles a detailed trade log from the order history.

        Returns:
            pd.DataFrame: A DataFrame containing structured trade details for all orders.
        """
        orders = list(self._orders)
        if alive:
            orders += self._orders
        trade_log = [order.to_dict() for order in orders]
        return pd.DataFrame(trade_log)

    def __str__(self) -> str:
        """
        Provides a string representation of the broker's state.

        Returns:
            str: A summary of the broker's balance, positions, and orders.
        """
        return (
            f"Balance: {self._balance}\n"
            f"Positions: {self._positions}\n"
            f"Commssion: {self.commission}\n"
            f"#Orders: {len(self._orders)}\n"
            f"#Pendings: {self._pendings.qsize()}"
        )
    
    def __repr__(self):
        return self.__str__()

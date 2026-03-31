import json
import numpy as np
import pandas as pd
from uuid import uuid4
from collections import deque
from .source import Source
from .order import Order, Delivery
from .friction import FixedRateCommission, FixedRateSlippage


class Broker:
    """Simulated brokerage for order management, execution, and portfolio accounting.

    The Broker wraps execution logic with commission and slippage models, maintains
    cash balance, positions, pending orders, execution deliveries, and processed
    orders. It advances with market data provided by a Source and applies matching
    rules for MARKET, LIMIT, STOP, and STOPLIMIT orders, supporting partial fills.

    Attributes:
      id (str): Unique identifier of the broker.
      commission (FixedRateCommission): Commission model applied to executions.
      slippage (FixedRateSlippage): Slippage model applied to executions.
      time (pandas.Timestamp or None): Broker's current timestamp synced with market data.
      balance (float): Current cash balance.
      positions (dict[str, float]): Current positions per instrument code (quantity).
      pendings (collections.deque[Order]): Queue of pending orders awaiting execution.
      orders (list[Order]): History of processed orders (filled, canceled, rejected, expired).
      delivery (list[Delivery]): All execution or cash movement deliveries recorded.

    Notes:
      - Commission and slippage models must be callable. Commission is called as
        commission(order, price, quantity) and returns a float. Slippage is called
        as slippage(order, row) and returns a tuple (price: float, quantity: int),
        where row is market data for the instrument at the current time.
      - Source must provide:
        - time (pandas.Timestamp): Current market timestamp.
        - data (pandas.DataFrame): Market snapshot with index containing instrument
          codes and columns referenced by Source (e.g., close, high, low).
        - attributes 'close', 'high', 'low' denoting the column names used for pricing.
    """

    order_type = Order

    def __init__(
        self,
        id: str = None,
        commission: FixedRateCommission = None,
        slippage: FixedRateSlippage = None,
    ):
        """Initialize a Broker with commission and slippage models.

        If no id is provided, a UUID4 string is generated. If no commission or slippage
        model is provided, defaults are used.

        Args:
          id (str, optional): Broker identifier. Defaults to a generated UUID4 string.
          commission (FixedRateCommission, optional): Commission model. Defaults to FixedRateCommission().
          slippage (FixedRateSlippage, optional): Slippage model. Defaults to FixedRateSlippage().

        Raises:
          ValueError: Not raised during initialization; see other methods for validation.
        """
        self.id = id or str(uuid4())
        self.commission = commission or FixedRateCommission()
        self.slippage = slippage or FixedRateSlippage()
        self._time = None
        self._balance = 0
        self._positions = {}
        self._pendings = deque()
        self._delivery = []  # Key parameter to restore the state of the broker
        self._orders = []  # History of processed orders
        self._order_dict = {}

    @property
    def time(self) -> pd.Timestamp:
        """Return the broker's current timestamp.

        Returns:
          pandas.Timestamp or None: The current broker time. None if not yet initialized.
        """
        return self._time

    @property
    def balance(self) -> float:
        """Return the current cash balance.

        Returns:
          float: Available cash balance.
        """
        return self._balance

    @property
    def positions(self) -> dict:
        """Return current positions.

        Returns:
          dict[str, float]: Mapping from instrument code to position quantity.
        """
        return self._positions

    @property
    def pendings(self) -> deque[Order]:
        """Return the queue of pending orders.

        Returns:
          collections.deque[Order]: FIFO queue of orders awaiting execution or expiration.
        """
        return self._pendings

    @property
    def orders(self) -> list[Order]:
        """Return the list of processed orders.

        Returns:
          list[Order]: History of orders that are filled, canceled, rejected, or expired.
        """
        return self._orders

    @property
    def delivery(self) -> list:
        """Return all recorded deliveries.

        Returns:
          list[Delivery]: List of execution and cash movement deliveries.
        """
        return self._delivery

    def submit(self, order: Order) -> None:
        """Submit an order to the broker's pending queue.

        The order status is set to SUBMITTED.

        Args:
          order (Order): The order to submit.

        Returns:
          None
        """
        self._pendings.append(order)
        self._order_dict[order.id] = order
        order.status = order.SUBMITTED

    def create(
        self,
        code: str,
        type: str,
        quantity: float,
        exectype: str,
        limit: float = None,
        trigger: float = None,
        id: str = None,
        valid: str = None,
    ) -> Order:
        """Create and submit a new order at the current broker time.

        This is a low-level constructor that validates broker time and forwards to submit().

        Args:
          code (str): Instrument code (e.g., ticker).
          type (str): Order side, typically Order.BUY or Order.SELL.
          quantity (float): Requested quantity.
          exectype (str): Execution type (Order.MARKET, Order.LIMIT, Order.STOP, Order.STOPLIMIT).
          limit (float, optional): Limit price for LIMIT/STOPLIMIT orders. Defaults to None.
          trigger (float, optional): Trigger price for STOP/STOPLIMIT orders. Defaults to None.
          id (str, optional): Order identifier. Defaults to a generated UUID4 string.
          valid (str, optional): Good-till timestamp as a pandas-parsable string. Defaults to None.

        Returns:
          Order: The created order, now in the SUBMITTED status and pending queue.

        Raises:
          ValueError: If the broker has no current time (must be set before creating orders).
        """
        if self._time is None:
            raise ValueError("broker must be initialized with a time")
        order = self.order_type(
            time=self.time,
            code=code,
            type=type,
            quantity=quantity,
            exectype=exectype,
            limit=limit,
            trigger=trigger,
            id=id,
            valid=valid,
        )
        self.submit(order)
        return order

    def cancel(self, order_or_id: str | Order) -> Order:
        """Cancel an existing order by object or identifier.

        If an identifier is provided, the corresponding order is looked up and canceled.

        Args:
          order_or_id (str | Order): The order object or its unique identifier.

        Returns:
          Order: The canceled order (status set to CANCELED if it was open).

        Raises:
          KeyError: If an order id is provided but cannot be found.
        """
        if isinstance(order_or_id, str):
            order_or_id = self.get_order(order_or_id)
        order_or_id.cancel()
        return order_or_id

    def deliver(self, delivery: Delivery):
        """Record a delivery event.

        Appends the given delivery (execution or cash/position movement) to the broker's
        delivery log.

        Args:
          delivery (Delivery): The delivery to record.

        Returns:
          None
        """
        self._delivery.append(delivery)

    def transfer(self, time: pd.Timestamp, amount: float):
        """Adjust cash balance and record a TRANSFER delivery.

        Positive amounts increase balance (e.g., deposit); negative amounts reduce
        balance (e.g., withdrawal). Uses a synthetic instrument code 'CASH' and
        a unit price of 1.

        Args:
          time (pandas.Timestamp): Timestamp of the transfer (or any pandas-parsable input).
          amount (float): Cash amount to transfer. Positive for deposit, negative for withdrawal.

        Returns:
          None

        Raises:
          ValueError: If time cannot be parsed into a pandas.Timestamp.
        """
        self._balance += amount
        self._time = pd.to_datetime(time)
        self.deliver(
            Delivery(
                time=self._time,
                code="CASH",
                type="TRANSFER",
                quantity=amount,
                price=1,
                comm=0,
            )
        )

    def buy(
        self,
        code: str,
        quantity: float,
        exectype: str = Order.MARKET,
        limit: float = None,
        trigger: float = None,
        id: str = None,
        valid: str = None,
    ) -> Order:
        """Create and submit a BUY order.

        Convenience wrapper around create() for BUY side.

        Args:
          code (str): Instrument code.
          quantity (float): Requested quantity.
          exectype (str, optional): Execution type. Defaults to Order.MARKET.
          limit (float, optional): Limit price for LIMIT/STOPLIMIT. Defaults to None.
          trigger (float, optional): Trigger price for STOP/STOPLIMIT. Defaults to None.
          id (str, optional): Order identifier. Defaults to None (auto-generated).
          valid (str, optional): Good-till timestamp as a pandas-parsable string. Defaults to None.

        Returns:
          Order: The submitted BUY order.

        Raises:
          ValueError: If broker time is not initialized.
        """
        return self.create(
            type=self.order_type.BUY,
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
        quantity: float,
        exectype: str = Order.MARKET,
        limit: float = None,
        trigger: float = None,
        id: str = None,
        valid: str = None,
    ) -> Order:
        """Create and submit a SELL order.

        Convenience wrapper around create() for SELL side.

        Args:
          code (str): Instrument code.
          quantity (float): Requested quantity.
          exectype (str, optional): Execution type. Defaults to Order.MARKET.
          limit (float, optional): Limit price for LIMIT/STOPLIMIT. Defaults to None.
          trigger (float, optional): Trigger price for STOP/STOPLIMIT. Defaults to None.
          id (str, optional): Order identifier. Defaults to None (auto-generated).
          valid (str, optional): Good-till timestamp as a pandas-parsable string. Defaults to None.

        Returns:
          Order: The submitted SELL order.

        Raises:
          ValueError: If broker time is not initialized.
        """
        return self.create(
            type=self.order_type.SELL,
            code=code,
            quantity=quantity,
            exectype=exectype,
            limit=limit,
            trigger=trigger,
            id=id,
            valid=valid,
        )

    def update(self, source: Source) -> None:
        """Advance broker state using the latest market data.

        Syncs the broker time to source.time, iterates and attempts to match pending
        orders against source.data. Orders are either updated, kept pending, or finalized
        (FILLED, CANCELED, REJECTED, EXPIRED). Finalized orders are appended to history
        and returned in a notification list.

        Args:
          source (Source): Market data source with attributes:
            - time (pandas.Timestamp)
            - data (pandas.DataFrame) containing instrument rows and price columns
            - high (str): Column name for high price
            - low (str): Column name for low price
            - close (str): Column name for close price

        Returns:
          list[Order]: Orders that transitioned to a terminal status during this update.

        Raises:
          ValueError: If source.time is not a pandas.Timestamp or convertible to one.
        """
        self._time = source.time
        if not isinstance(self._time, pd.Timestamp):
            raise ValueError("time must be a pd.Timestamp or convertible to one")

        notify = []
        self._pendings.append(None)  # Placeholder for end-of-day processing.
        order = self._pendings.popleft()
        while order is not None:
            self._match(order, source)
            if not order.is_alive(self.time):
                self._orders.append(order)
                notify.append(order)
            else:
                self._pendings.append(order)
            order = self._pendings.popleft()
        return notify

    def _match(self, order: Order, source: Source) -> None:
        """Evaluate whether an order matches current market conditions.

        Handles triggers for STOP and STOPLIMIT orders. For MARKET or STOP orders,
        matching is unconditional when the order is triggered. For LIMIT or STOPLIMIT,
        matching requires price to cross the specified limit. If matched, applies
        slippage to determine executable price and quantity; executes if quantity
        is positive, otherwise the order is rejected.

        Args:
          order (Order): The order to evaluate.
          source (Source): Market data source as in update().

        Returns:
          None

        Raises:
          ValueError: If a trigger is set but exectype is not STOP or STOPLIMIT.
        """
        data = source.data
        if order.code not in data.index:
            return

        if order.trigger is not None:
            # STOP and STOPLIMIT orders:
            # Triggered when price higher than trigger for BUY,
            # or lower than trigger for SELL.
            if order.exectype == order.STOP or order.exectype == order.STOPLIMIT:
                pricetype = source.high if order.type == order.BUY else source.low
                if (
                    order.type == order.BUY
                    and data.loc[order.code, pricetype] >= order.trigger
                ) or (
                    order.type == order.SELL
                    and data.loc[order.code, pricetype] <= order.trigger
                ):
                    # order triggered
                    order.trigger = None
                return
            elif order.exectype == order.TARGET or order.exectype == order.TARGETLIMIT:
                pricetype = source.low if order.type == order.BUY else source.high
                if (
                    order.type == order.BUY
                    and data.loc[order.code, pricetype] <= order.trigger
                ) or (
                    order.type == order.SELL
                    and data.loc[order.code, pricetype] >= order.trigger
                ):
                    # order triggered
                    order.trigger = None
                return
            else:
                raise ValueError("Invalid order type for trigger.")

        # if the order type is market or stop order, match condition satisfies
        market_match = (
            order.exectype == order.MARKET
            or order.exectype == order.STOP
            or order.exectype == order.TARGET
        )
        # if the order type is limit or stop limit order, check if the price conditions are met
        limit_order = (
            order.exectype == order.LIMIT
            or order.exectype == order.STOPLIMIT
            or order.exectype == order.TARGETLIMIT
        )
        limit_match = limit_order and (
            (
                order.type == order.BUY
                and data.loc[order.code, source.low] <= order.limit
            )
            or (
                order.type == order.SELL
                and data.loc[order.code, source.high] >= order.limit
            )
        )
        # if match condition is satisfied
        if market_match or limit_match:
            price, quantity = self.slippage(order, data.loc[order.code])
            if quantity > 0:
                self._execute(order, price, quantity)
            else:
                order.status = order.REJECTED

    def _execute(self, order: Order, price: float, quantity: int) -> None:
        """Execute a fill, update cash and positions, and record delivery.

        Computes commission via the commission model. For BUY:
        - Requires sufficient cash; otherwise rejects the order.
        - Deducts cost (price * quantity + commission) from balance and increases position.

        For SELL:
        - Requires sufficient position; otherwise rejects the order.
        - Adds net revenue (price * quantity - commission) to balance and decreases position.

        Args:
          order (Order): The order being filled.
          price (float): Executed price per unit after slippage.
          quantity (int): Executed quantity.

        Returns:
          None
        """
        amount = price * quantity
        commission = self.commission(order, price, quantity)
        if order.type == order.BUY:
            cost = amount + commission
            if cost > self._balance:
                order.status = order.REJECTED
            else:
                delivery = Delivery(
                    time=self.time,
                    type=order.type,
                    code=order.code,
                    quantity=quantity,
                    price=price,
                    comm=commission,
                )
                order += delivery
                self.deliver(delivery)
                self._balance -= cost
                self._positions[order.code] = (
                    self._positions.get(order.code, 0) + quantity
                )
        elif order.type == order.SELL:
            revenue = amount - commission
            if self._positions.get(order.code, 0) < quantity:
                order.status = order.REJECTED
            else:
                delivery = Delivery(
                    time=self.time,
                    type=order.type,
                    code=order.code,
                    quantity=quantity,
                    price=price,
                    comm=commission,
                )
                order += delivery
                self.deliver(delivery)
                self._balance += revenue
                self._positions[order.code] -= quantity
                if self._positions[order.code] == 0:
                    del self._positions[order.code]

    def get_value(self, source: Source) -> float:
        """Compute portfolio market value at the source's close prices.

        Sums positions valued at source.data[source.close] plus cash balance.

        Args:
          source (Source): Market data source providing the close column name and DataFrame.

        Returns:
          float: Total portfolio value (positions mark-to-market + cash).
        """
        return (self.get_positions() * source.data[source.close]).sum() + self.balance

    def get_order(self, id: str) -> Order:
        """Retrieve an order by its unique identifier.

        Args:
          id (str): Order identifier.

        Returns:
          Order: The corresponding order.

        Raises:
          KeyError: If the order is not found.
        """
        return self._order_dict[id]

    def get_delivery(self, parse_dates: bool = True) -> pd.DataFrame:
        """Return deliveries as a DataFrame.

        Optionally parses the 'time' column to pandas.Timestamp.

        Args:
          parse_dates (bool, optional): Whether to parse date fields. Defaults to True.

        Returns:
          pandas.DataFrame: DataFrame with columns:
            - id (str)
            - time (str or pandas.Timestamp)
            - type (str)
            - code (str)
            - quantity (float)
            - price (float)
            - amount (float)
            - commission (float)
        """
        delivery = pd.DataFrame([deliv.dump() for deliv in self.delivery])
        if parse_dates and not delivery.empty:
            delivery["time"] = pd.to_datetime(delivery["time"])
        return delivery

    def get_pendings(
        self, parse_dates: bool = True, delivery: bool = True
    ) -> pd.DataFrame:
        """Return pending orders as a DataFrame.

        Optionally includes serialized deliveries and parses timestamp columns.

        Args:
          parse_dates (bool, optional): Whether to parse timestamps. Defaults to True.
          delivery (bool, optional): Whether to include associated deliveries. Defaults to True.

        Returns:
          pandas.DataFrame: DataFrame of pending orders with fields such as id, create,
          time, code, type, quantity, exectype, limit, trigger, valid, status, filled, and optional delivery list.
        """
        pendings = pd.DataFrame([order.dump(delivery) for order in self.pendings])
        if parse_dates and not pendings.empty:
            pendings["time"] = pd.to_datetime(pendings["time"])
            pendings["create"] = pd.to_datetime(pendings["create"])
            pendings["valid"] = pd.to_datetime(pendings["valid"])
        return pendings

    def get_orders(
        self, parse_dates: bool = True, delivery: bool = False
    ) -> pd.DataFrame:
        """Return processed orders (history) as a DataFrame.

        Optionally includes serialized deliveries and parses timestamp columns.

        Args:
          parse_dates (bool, optional): Whether to parse timestamps. Defaults to True.
          delivery (bool, optional): Whether to include associated deliveries. Defaults to False.

        Returns:
          pandas.DataFrame: DataFrame of processed orders with fields analogous to dump().
        """
        orders = pd.DataFrame([order.dump(delivery) for order in self.orders])
        if parse_dates and not orders.empty:
            orders["time"] = pd.to_datetime(orders["time"])
            orders["create"] = pd.to_datetime(orders["create"])
            orders["valid"] = pd.to_datetime(orders["valid"])
        return orders

    def get_positions(self) -> pd.Series:
        """Return positions as a pandas Series.

        Returns:
          pandas.Series: Series indexed by instrument code with quantities, named 'positions'.
        """
        return pd.Series(self._positions, name="positions")

    def dump(self, history: bool = True) -> dict:
        """Serialize broker state to a JSON-friendly dictionary.

        Includes id, balance, positions, pending orders, and current time. If history
        is True, also includes delivery and processed orders. NaNs are converted to
        None in nested DataFrames.

        Args:
          history (bool, optional): Include deliveries and processed orders. Defaults to True.

        Returns:
          dict: Serialized broker state. Keys:
            - id (str)
            - balance (float)
            - positions (dict[str, float])
            - pendings (list[dict])
            - time (str or None): ISO-8601 string if available
            - delivery (list[dict], optional)
            - orders (list[dict], optional)
        """
        pendings = self.get_pendings(parse_dates=False)
        data = {
            "id": self.id,
            "balance": self.balance,
            "positions": self.positions,
            "pendings": pendings.replace(np.nan, None).to_dict(orient="records"),
            "time": self._time.isoformat() if self._time else None,
        }
        if history:
            data["delivery"] = (
                self.get_delivery(parse_dates=False)
                .replace(np.nan, None)
                .to_dict(orient="records")
            )
            data["orders"] = (
                self.get_orders(parse_dates=False)
                .replace(np.nan, None)
                .to_dict(orient="records")
            )
        return data

    def store(self, path: str, history: bool = True):
        """Persist broker state to a JSON file.

        Args:
          path (str): File path to write.
          history (bool, optional): Include deliveries and processed orders. Defaults to True.

        Returns:
          None

        Raises:
          OSError: If the file cannot be written.
          IOError: If an I/O error occurs during writing.
        """
        data = self.dump(history)
        with open(path, "w") as f:
            json.dump(data, f, indent=4, ensure_ascii=True)

    @classmethod
    def load(
        cls,
        data: dict,
        commission: FixedRateCommission = None,
        slippage: FixedRateSlippage = None,
    ) -> "Broker":
        """Reconstruct a Broker from a serialized dictionary.

        Restores broker core attributes, pending orders, processed orders, and deliveries.
        Commission and slippage models can be supplied to override defaults.

        Args:
          data (dict): Serialized broker state as produced by dump().
          commission (FixedRateCommission, optional): Commission model to use. Defaults to None (use default).
          slippage (FixedRateSlippage, optional): Slippage model to use. Defaults to None (use default).

        Returns:
          Broker: The reconstructed broker.

        Raises:
          KeyError: If required fields (e.g., id, balance, positions, time) are missing.
          ValueError: If time fields cannot be parsed into pandas.Timestamp.
        """
        # Initialize Broker with external market data and commission
        broker = cls(id=data["id"], commission=commission, slippage=slippage)

        # Restore basic attributes
        broker._time = pd.to_datetime(data["time"])
        broker._balance = data["balance"]
        broker._positions = data["positions"]

        # Restore pending orders
        for order_data in data["pendings"]:
            order = Order.load(order_data)
            broker._pendings.append(order)

        # Restore orders
        for order_data in data.get("orders", []):
            order = Order.load(order_data)
            broker._orders.append(order)

        # Restore deliveries
        for delivery_data in data.get("delivery", []):
            delivery = Delivery.load(delivery_data)
            broker._delivery.append(delivery)

        return broker

    @classmethod
    def restore(
        cls,
        path: str,
        commission: FixedRateCommission = None,
        slippage: FixedRateSlippage = None,
    ):
        """Load a Broker from a JSON file.

        Reads the file at the given path and delegates to load().

        Args:
          path (str): Path to the JSON file.
          commission (FixedRateCommission, optional): Commission model to use. Defaults to None.
          slippage (FixedRateSlippage, optional): Slippage model to use. Defaults to None.

        Returns:
          Broker: The reconstructed broker.

        Raises:
          FileNotFoundError: If the file does not exist.
          json.JSONDecodeError: If the file content is not valid JSON.
        """
        with open(path, "r") as f:
            data = json.load(f)
        return cls.load(data, commission, slippage)

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}(#{self.id}@{self.time}\n"
            f"  balance: ${self.balance:.2f}\n"
            f"  commission: {self.commission}\n"
            f"  slippage: {self.slippage}\n"
            f"  #pending: {len(self.pendings)} & #over: {len(self.orders)} & #deliveries: {len(self.delivery)}\n"
            f"  position: {self.positions}\n"
            ")"
        )

    def __repr__(self):
        return self.__str__()

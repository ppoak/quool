import numpy as np
import pandas as pd
from uuid import uuid4
from collections import deque
from .order import Order, Delivery
from .friction import FixedRateCommission, FixedRateSlippage


class Broker:

    order_type = Order

    def __init__(
        self,
        id: str,
        commission: FixedRateCommission = None,
        slippage: FixedRateSlippage = None,
    ):
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
        return self._time

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def positions(self) -> dict:
        return self._positions

    @property
    def pendings(self) -> deque[Order]:
        return self._pendings

    @property
    def orders(self) -> list[Order]:
        return self._orders

    @property
    def delivery(self) -> list:
        return self._delivery

    def submit(self, order: Order) -> None:
        self._pendings.append(order)
        self._order_dict[order.id] = order
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
    ) -> Order:
        if self._time is None:
            raise ValueError("broker must be initialized with a time")
        order = self.order_type(
            time=self._time,
            side=side,
            code=code,
            quantity=quantity,
            limit=limit,
            trigger=trigger,
            exectype=exectype,
            id=id,
            valid=valid,
        )
        self.submit(order)
        return order

    def cancel(self, order_or_id: str | Order) -> Order:
        if isinstance(order_or_id, str):
            order_or_id = self.get_order(order_or_id)
        order_or_id.cancel()
        return order_or_id

    def deliver(self, delivery: Delivery):
        self._delivery.append(delivery)

    def transfer(self, time: pd.Timestamp, amount: float):
        self._balance += amount
        self._time = pd.to_datetime(time)
        self.deliver(
            Delivery(
                time=self._time,
                code="CASH",
                type="TRANSFER",
                quantity=0,
                price=0,
                amount=amount,
            )
        )

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
        """
        Creates and submits a buy order.

        Args:
            code (str): The stock code (e.g., "AAPL").
            quantity (int): The number of shares to buy.
            trigger (float, optional): The trigger price for the order. Defaults to None.
            limit (float, optional): The limit price for the order. Defaults to None.
            exectype (str): The execution type ('MARKET' or 'LIMIT'). Defaults to 'MARKET'.
            valid (str, optional): The validity period for the order. Defaults to None.

        Returns:
            Order: The created buy order.
        """
        return self.create(
            side=self.order_type.BUY,
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
        """
        Creates and submits a sell order.

        Args:
            code (str): The stock code (e.g., "AAPL").
            quantity (int): The number of shares to sell.
            trigger (float, optional): The trigger price for the order. Defaults to None.
            limit (float, optional): The limit price for the order. Defaults to None.
            exectype (str): The execution type ('MARKET' or 'LIMIT'). Defaults to 'MARKET'.
            valid (str, optional): The validity period for the order. Defaults to None.

        Returns:
            Order: The created sell order.
        """
        return self.create(
            side=self.order_type.SELL,
            code=code,
            quantity=quantity,
            exectype=exectype,
            limit=limit,
            trigger=trigger,
            id=id,
            valid=valid,
        )

    def update(self, time: str | pd.Timestamp, data: pd.DataFrame) -> None:
        """
        Updates the broker's state for a new trading day.

        Args:
            time (str): The current trading day.
        """
        if not isinstance(data, pd.DataFrame):
            return

        self._time = pd.to_datetime(time)
        if not isinstance(self._time, pd.Timestamp):
            raise ValueError("time must be a pd.Timestamp or convertible to one")

        self._pendings.append(None)  # Placeholder for end-of-day processing.
        order = self._pendings.popleft()
        while order is not None:
            self._match(order, data)
            if not order.is_alive(self.time):
                self._orders.append(order)
            else:
                self._pendings.append(order)
            order = self._pendings.popleft()

    def _match(self, order: Order, data: pd.DataFrame) -> None:
        """
        Matches an order with market data and determines the execution price and quantity.

        Args:
            order (Order): The order to be matched.
            data (pd.DataFrame): The market data containing open, high, low, close, and volume information.
        """
        if order.code not in data.index:
            return

        if order.trigger is not None:
            # STOP and STOPLIMIT orders:
            # Triggered when price higher than trigger for BUY,
            # or lower than trigger for SELL.
            if order.exectype == order.STOP or order.exectype == order.STOPLIMIT:
                pricetype = "high" if order.type == order.BUY else "low"
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
            else:
                raise ValueError("Invalid order type for trigger.")

        # if the order type is market or stop order, match condition satisfies
        market_order = order.exectype == order.MARKET or order.exectype == order.STOP
        # if the order type is limit or stop limit order, check if the price conditions are met
        limit_order = order.exectype == order.LIMIT or order.exectype == order.STOPLIMIT
        limit_match = (
            order.type == order.BUY and data.loc[order.code, "low"] <= order.limit
        ) or (order.type == order.SELL and data.loc[order.code, "high"] >= order.limit)
        # if match condition is not satisfied
        if market_order or (limit_order and limit_match):
            price, quantity = self.slippage(order, data.loc[order.code], quantity)
            if quantity > 0:
                self._execute(order, price, quantity)

    def _execute(self, order: Order, price: float, quantity: int) -> None:
        """
        Executes an order and updates broker's balance and positions.

        Args:
            order (Order): The order to be executed.
            price (float): The price at which the order is executed.
            quantity (int): The quantity of shares executed.
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
                    code=order.code,
                    type=order.type,
                    quantity=quantity,
                    price=price,
                    amount=cost,
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
                    code=order.code,
                    type=order.type,
                    quantity=quantity,
                    price=price,
                    amount=cost,
                )
                order += delivery
                self.deliver(delivery)
                self._balance += revenue
                self._positions[order.code] -= quantity
                if self._positions[order.code] == 0:
                    del self._positions[order.code]

    def get_value(self, data: pd.DataFrame) -> float:
        return (self.get_positions() * data["close"]).sum() + self.balance

    def get_order(self, id: str) -> Order:
        return self._orders.get[id]

    def get_delivery(self) -> pd.DataFrame:
        return pd.DataFrame([deliv.dump() for deliv in self.delivery])

    def get_pendings(self) -> pd.DataFrame:
        return pd.DataFrame([order.dump() for order in self.pendings])

    def get_orders(self) -> pd.DataFrame:
        return pd.DataFrame([order.dump() for order in self.orders])

    def get_positions(self) -> pd.Series:
        return pd.Series(self._positions, name="positions")

    def dump(self, history: bool = True) -> dict:
        pendings = self.get_pendings()
        data = {
            "id": self.id,
            "balance": self.balance,
            "positions": self.positions,
            "pendings": pendings.replace(np.nan, None).to_dict(orient="records"),
            "time": self._time.isoformat() if self._time else None,
        }
        if history:
            delivery = self.get_delivery()
            orders = self.get_orders()
            if not self.delivery.empty:
                delivery["time"] = delivery["time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
            data["delivery"] = (
                delivery.to_dict(orient="records") if not delivery.empty else []
            )
            data["orders"] = (
                orders.replace(np.nan, None).to_dict(orient="records")
                if not orders.empty
                else []
            )

    @classmethod
    def load(
        cls, data: dict, commission: FixedRateCommission, slippage: FixedRateSlippage
    ) -> "Broker":
        # Initialize Broker with external market data and commission
        broker = cls(id=data["id"], commission=commission, slippage=slippage)

        # Restore basic attributes
        broker._time = pd.to_datetime(data["time"])
        broker._balance = data["balance"]
        broker._positions = data["positions"]

        # Restore pending orders
        for order_data in data["pendings"]:
            order = Order.load(order_data)
            broker._pendings.put(order)

        # Restore orders
        for order_data in data.get("orders", []):
            order = Order.load(order_data)
            broker._orders.append(order)

        # Restore deliveries
        for delivery_data in data.get("delivery", []):
            delivery = Delivery.load(delivery_data)
            broker._delivery.append(delivery)

        return broker

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.id})@{self.time}\n"
            f"balance: ${self.balance:.2f}\n"
            f"commission: {self.commission}\n"
            f"slippage: {self.slippage}\n"
            f"#pending: {len(self.pendings)} & #over: {len(self.orders)} & #deliveries: {len(self.delivery)}\n"
            f"position: {self.positions}\n"
        )

    def __repr__(self):
        return self.__str__()

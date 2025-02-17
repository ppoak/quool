import numpy as np
import pandas as pd
from uuid import uuid4
from .order import Order
from .util import evaluate
from collections import deque
from .friction import FixedRateCommission, FixedRateSlippage


class Broker:

    order_type = Order

    def __init__(
        self,
        id: str,
        commission: FixedRateCommission,
        slippage: FixedRateSlippage,
    ):
        self.id = id or str(uuid4())
        self.commission = commission
        self.slippage = slippage
        self._time = None
        self._balance = 0
        self._positions = {}
        self._pendings = deque()
        self._ledger = [] # Key parameter to restore the state of the broker
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
    def ledger(self) -> list:
        return self._ledger

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
            ordtype=exectype,
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
    
    def _post(
        self,
        time: pd.Timestamp,
        code: str,
        ttype: str,
        unit: float,
        amount: float,
        price: float,
        commission: float,
    ):
        """
        Records a transaction in the broker's ledger.

        Args:
            time (pd.Timestamp): The timestamp of the transaction.
            code (str): The stock code.
            ttype (str): The transaction type ('BUY' or 'SELL').
            unit (float): The number of shares transacted.
            amount (float): The total amount of the transaction.
            price (float): The price per share.
            commission (float): The commission fee.
        """
        self._ledger.append({
            "time": time,
            "code": code,
            "ttype": ttype,
            "unit": unit,
            "amount": amount,
            "price": price,
            "commission": commission,
        })

    def transfer(self, time: pd.Timestamp, amount: float):
        self._balance += amount
        self._time = pd.to_datetime(time)
        self._post(time=self._time, code="CASH", ttype="TRANSFER", unit=0, amount=amount, price=0, commission=0)
    
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
            side = self.order_type.BUY,
            code = code,
            quantity = quantity,
            exectype = exectype,
            limit = limit,
            trigger = trigger,
            id = id,
            valid = valid,
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
            side = self.order_type.SELL,
            code = code,
            quantity = quantity,
            exectype = exectype,
            limit = limit,
            trigger = trigger,
            id = id,
            valid = valid,
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
            if not order.is_alive():
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
                pricetype = "high" if order.side == order.BUY else "low"
                if (
                    (order.side == order.BUY and data.loc[order.code, pricetype] >= order.trigger)
                    or (order.side == order.SELL and data.loc[order.code, pricetype] <= order.trigger)
                ):
                    # order triggered
                    order.trigger = None
                return
            else:
                raise ValueError("Invalid order type for trigger.")

        # If the order type is market or stop order, use the opening price and minimum trading volume
        if order.exectype == order.MARKET or order.exectype == order.STOP:
            quantity = min(data.loc[order.code, "volume"], order.quantity - order.filled)
            price = self.slippage(order, quantity, data.loc[order.code])
        # If the order type is limit or stop limit order, check if the price conditions are met
        elif order.exectype == order.LIMIT or order.exectype == order.STOPLIMIT:
            if order.side == order.BUY and data.loc[order.code, "low"] <= order.limit:
                quantity = min(data.loc[order.code, "volume"], order.quantity - order.filled)
                price = self.slippage(order, quantity, data.loc[order.code])
            elif order.side == order.SELL and data.loc[order.code, "high"] >= order.limit:
                quantity = min(data.loc[order.code, "volume"], order.quantity - order.filled)
                price = self.slippage(order, quantity, data.loc[order.code])
            else:
                return
                
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
        if order.side == order.BUY:
            cost = amount + commission
            if cost > self._balance:
                order.status = order.REJECTED
            else:
                order.execute(self.time, price, quantity)
                order.commission = getattr(order, "commission", 0) + commission
                self._post(
                    time=self.time, code=order.code, ttype=order.side, 
                    unit=quantity, amount=-amount, price=price, commission=commission
                )
                self._balance -= cost
                self._positions[order.code] = self._positions.get(order.code, 0) + quantity
        elif order.side == order.SELL:
            revenue = amount - commission
            if self._positions.get(order.code, 0) < quantity:
                order.status = order.REJECTED
            else:
                order.execute(self.time, price, quantity)
                order.commission = getattr(order, "commission", 0) + commission
                self._post(
                    time=self.time, code=order.code, ttype=order.side, 
                    unit=-quantity, amount=amount, price=price, commission=commission
                )
                self._balance += revenue
                self._positions[order.code] -= quantity
                if self._positions[order.code] == 0:
                    del self._positions[order.code]
    
    def get_value(self, data: pd.DataFrame) -> float:
        return (self.get_positions() * data["close"]).sum()
        
    def get_order(self, id: str) -> Order:
        return self._orders.get[id]
    
    def get_ledger(self) -> pd.DataFrame:
        return pd.DataFrame(self.ledger)
    
    def get_pendings(self) -> pd.DataFrame:
        return pd.DataFrame([order.dump() for order in self.pendings])
    
    def get_orders(self) -> pd.DataFrame:
        return pd.DataFrame([order.dump() for order in self.orders]), 
    
    def get_positions(self) -> pd.Series:
        return pd.Series(self._positions, name="positions")

    def report(self, benchmark: pd.Series = None):
        """
        Generates a report of the broker's performance.

        Returns:
            dict: A dictionary containing the broker's performance metrics.
        """
        ledger = self.ledger.set_index(["time", "code"]).sort_index()
        prices = self.market["close"].unstack("code")

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
        trades["duration"] = pd.to_datetime(trades["close_at"]) - pd.to_datetime(trades["open_at"])
        trades["return"] = (trades["close_amount"] - trades["open_amount"]) / trades["open_amount"]
        return {
            "evaluation": evaluate(value=total, benchmark=benchmark, turnover=turnover, trades=trades),
            "values": pd.concat(
                [total, market, cash, turnover], 
                axis=1, keys=["total", "market", "cash", "turnover"]
            ),
            "positions": positions,
            "trades": trades,
        }

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
    
    @classmethod
    def load(cls, data: dict, commission: FixedRateCommission, slippage: FixedRateSlippage) -> 'Broker':
        """
        Restores a Broker instance from a dictionary.
        Market data must be provided externally.

        Args:
            data (dict): A dictionary containing the serialized Broker state.
            market_data (pd.DataFrame): Market data required for the Broker.

        Returns:
            Broker: The restored Broker instance.
        """
        # Initialize Broker with external market data and commission
        broker = cls(id=data["id"], commission=commission, slippage=slippage)

        # Restore basic attributes
        broker._time = pd.to_datetime(data["time"])
        broker._balance = data["balance"]
        broker._positions = data["positions"]
        ledger = pd.DataFrame(data["ledger"])
        ledger["time"] = pd.to_datetime(ledger["time"])
        broker._ledger = ledger.to_dict(orient="records")

        # Restore orders
        for order_data in data["orders"]:
            order = Order.load(order_data, broker)
            broker._orders.append(order)

        # Restore pending orders
        for order_data in data["pendings"]:
            order = Order.load(order_data, broker)
            broker._pendings.put(order)

        return broker

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

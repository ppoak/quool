import uuid
import json
import queue
import numpy as np
import pandas as pd
from .tool import evaluate
from .manager import ParquetManager


class Order:
    """Represents a financial order with attributes and state management.
    
    Class Attributes:
        CREATED (str): Order status indicating the order has been created.
        SUBMITTED (str): Order status indicating the order has been submitted.
        PARTIAL (str): Order status indicating the order has been partially filled.
        FILLED (str): Order status indicating the order has been fully filled.
        CANCELED (str): Order status indicating the order has been canceled.
        EXPIRED (str): Order status indicating the order has expired.
        REJECTED (str): Order status indicating the order has been rejected.
        MARKET (str): Order type indicating a market order.
        LIMIT (str): Order type indicating a limit order.
        STOP (str): Order type indicating a stop order.
        STOPLIMIT (str): Order type indicating a stop-limit order.
        BUY (str): Order side indicating a buy order.
        SELL (str): Order side indicating a sell order.

    Attributes:
        broker (Broker): The broker instance associated with the order.
        code (str): The code of the financial instrument.
        quantity (int): The number of units for the order.
        trigger (float): The price for triggering a limit order.
        limit (float): The price for limit orders.
        ordtype (str): The type of order ('MARKET' or 'LIMIT' or 'STOP' or 'STOPLIMIT').
        side (str): The side of the order ('BUY' or 'SELL').
        cretime (str): The creation timestamp.
        exetime (str): The execution timestamp.
        status (str): The current status of the order.
        ordid (str): The unique identifier for the order.
    """

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
    BUY = "BUY"
    SELL = "SELL"

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
        ordid: str = None,
    ):
        """
        Initializes an Order instance.

        Args:
            broker (Broker): The broker instance associated with the order.
            code (str): The code of the financial instrument.
            quantity (int): The number of units for the order.
            trigger (float, optional): The price for triggering a limit order.
                Defaults to None for market orders.
            limit (float, optional): The price for limit orders. Defaults to None.
            ordtype (str): The type of order ('MARKET' or 'LIMIT'). Defaults to 'MARKET'.
            side (str): The side of the order ('BUY' or 'SELL'). Defaults to 'BUY'.
            time (str, optional): The creation timestamp. Defaults to None.
            valid (str, optional): The expiry time for the order. Defaults to None.
        """
        self.ordid = ordid or str(uuid.uuid4())
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
            self.exetime = self.broker.time

    def cancel(self) -> None:
        """
        Cancels the order if it is not already fully executed.

        Updates:
            - Sets the status to 'CANCELED'.
        """
        if self.status in {self.CREATED, self.PARTIAL, self.SUBMITTED}:
            self.status = self.CANCELED

    def is_alive(self) -> bool:
        """
        Checks if the order is still valid based on its expiration and status.

        Returns:
            bool: True if the order is valid, False otherwise.
        """
        if self.status in {self.CREATED, self.SUBMITTED, self.PARTIAL}:
            if self.valid and self.broker.time > self.valid:
                self.status = self.EXPIRED
                return False
            return True
        return False

    def dump(self) -> dict:
        """
        Converts the order's attributes to a dictionary for structured representation.

        Returns:
            dict: A dictionary containing the order's attributes.
        """
        return {
            "ordid": self.ordid,
            "code": self.code,
            "quantity": self.quantity,
            "limit": self.limit,
            "trigger": self.trigger,
            "exeprice": self.exeprice,
            "ordtype": self.ordtype,
            "side": self.side,
            "status": self.status,
            "filled": self.filled,
            "value": self.value,
            "cretime": self.cretime.isoformat() if self.cretime else None,
            "exetime": self.exetime.isoformat() if self.exetime else None,
            "commission": self.commission,
            "valid": self.valid.isoformat() if self.valid else None,
        }

    @classmethod
    def load(cls, data: dict, broker: 'Broker') -> 'Order':
        """
        Creates an Order instance from a dictionary.

        Args:
            data (dict): A dictionary containing the order's attributes.
            broker (Broker): The broker instance associated with the order.

        Returns:
            Order: The reconstructed Order instance.
        """
        order = cls(
            broker=broker,
            code=data["code"],
            quantity=data["quantity"],
            limit=data.get("limit"),
            trigger=data.get("trigger"),
            ordtype=data.get("ordtype", cls.MARKET),
            side=data.get("side", cls.BUY),
            time=data.get("cretime"),
            valid=data.get("valid"),
        )
        # Restore additional attributes
        order.ordid = data.get("ordid", order.ordid)  # Use existing ordid or generate a new one
        order.status = data.get("status", cls.CREATED)
        order.filled = data.get("filled", 0)
        order.value = data.get("value", 0)
        order.exetime = pd.to_datetime(data["exetime"]) if data.get("exetime") else None
        order.exeprice = data.get("exeprice")
        order.commission = data.get("commission", 0)
        
        return order

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
            f"Order(#{self.ordid[:5]}@[{latest_date}] {self.ordtype} {self.side} <{self.code}> "
            f"{self.quantity:.2f}x${latest_price:.2f} |{self.status}|)"
        )

    def __repr__(self) -> str:
        """
        Provides a detailed string representation for the object.

        Returns:
            str: The string representation of the order.
        """
        return self.__str__()


class Broker:

    """Broker class is a interface for managing trading operations and market matching.

    Attributes:        
        brokid (str): The unique identifier for the broker instance. If not provided when initializing, uuid will be generated.
        commission (float): The commission rate for transactions. Defaults to 0.001 (0.1%).
        time (pd.Timestamp): The current time of the broker. Indicating the latest time when broker called `update()`.
        balance (float): The current balance of the broker.
        positions (pd.Series): The current positions held by the broker. A pd.Series with code index and share value.
        pendings (pd.DataFrame): The pending orders of the broker. A pd.DataFrame with detailed information of each order.
            For detailed field of the DataFrame, please refer to attributes of `Order` class.
        orders (pd.DataFrame): The history of processed orders. A pd.DataFrame with detailed information of each order.
            For detailed field of the DataFrame, please refer to attributes of `Order` class.
        ledger (pd.DataFrame): The history of balance changes. A pd.DataFrame with detailed information of each balance change.

    Methods:
        update(time: pd.Timestamp, market: pd.DataFrame): Update the broker's state to the given time.
            market (pd.DataFrame): The market data for trading. which contains a standard format:
                1. Index (pd.Index): The stock codes and their corresponding levels.
                2. Columns:
                    - 'open': The opening price of the stock.
                    - 'high': The highest price of the stock.
                    - 'low': The lowest price of the stock.
                    - 'close': The closing price of the stock.
                    - 'volume': The trading volume of the stock.
        buy(code: str, quantity: float, limit: float, trigger: float, exectype: str, valid: pd.Timestamp): Place a buy order.
            limit: The limit price for the order. Only fits for `exectype = "LIMIT"` or `exectype = "STOP_LIMIT"`.
            trigger: The trigger price for the order. Only fits for `exectype = "STOP"` or `exectype = "STOP_LIMIT"`.
            exectype: The execution type of the order. Only fits for `exectype = "MARKET"` or `exectype = "LIMIT"` or `exectype = "STOP"` or `exectype = "STOP_LIMIT"`.
            valid: The validity period of the order. You should provide it in pd.Timestamp format.
        sell(code: str, quantity: float, limit: float, trigger: float, exectype: str, valid: pd.Timestamp): Place a sell order.
            see more information in `buy()`.
        cancel(orderid: str): Cancel the order with the given orderid.
        close(code: str, limit: float, trigger: float, exectype: str, valid: pd.Timestamp): Close the position with the given code.
            see more information in `buy()`.
        get_value(market: pd.DataFrame): Calculate the current value of the broker's portfolio.
            market (pd.DataFrame): The market data for trading. please refer to `update()` for more information.
    """

    def __init__(
        self,
        brokid: str = None,
        commission: float = 0.001,
    ):
        """
        Initializes the broker with essential attributes.

        Args:
            market (pd.DataFrame): The market data for trading.
            principle (float): Initial cash balance for the broker. Defaults to 1,000,000.
            commission (float): Commission rate for transactions. Defaults to 0.001 (0.1%).
        """
        self.brokid = brokid or str(uuid.uuid4())
        self._time = None
        self._balance = 0
        self._positions = {}
        self.commission = commission
        self._pendings = queue.Queue()
        self._ledger = [] # Key parameter to restore the state of the broker
        self._orders = []  # History of processed orders
        self._ordict = {}
        self.container = {}

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
        return pd.Series(self._positions, name="positions")
    
    @property
    def pendings(self) -> list:
        """
        Returns the list of orders submitted by the broker.

        Returns:
            list: A list of Order objects.
        """
        pendings = list(self._pendings.queue)
        if None in pendings:
            return pendings.remove(None)
        return pd.DataFrame([order.dump() for order in pendings])
    
    @property
    def orders(self) -> list:
        """
        Returns the list of processed orders.

        Returns:
            list: A list of Order objects.
        """
        return pd.DataFrame([order.dump() for order in self._orders])

    @property
    def ledger(self) -> pd.DataFrame:
        """
        Returns the DataFrame of transactions.

        Returns:
            pd.DataFrame: A DataFrame containing transaction data.
        """
        return pd.DataFrame(self._ledger)

    def transfer(self, time: pd.Timestamp, amount: float):
        """
        Transfers funds to or from the broker's balance.

        Args:
            amount (float): The amount to transfer.
            code (str, optional): The stock code for the transfer. Defaults to None.
        """
        self._balance += amount
        self._time = pd.to_datetime(time or 'now')
        self._post(time=self._time, code="CASH", ttype="TRANSFER", unit=0, amount=amount, price=0, commission=0)

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
            trigger (float, optional): The trigger price for the order. Defaults to None.
            limit (float, optional): The limit price for the order. Defaults to None.
            exectype (str): The execution type ('MARKET' or 'LIMIT'). Defaults to 'MARKET'.
            valid (str, optional): The validity period for the order. Defaults to None.

        Returns:
            Order: The created buy order.
        """
        order = Order(
            broker=self,
            code=code,
            quantity=quantity,
            limit=limit,
            trigger=trigger,
            ordtype=exectype,
            side=Order.BUY,
            time=self._time or pd.to_datetime('now'),
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
            trigger (float, optional): The trigger price for the order. Defaults to None.
            limit (float, optional): The limit price for the order. Defaults to None.
            exectype (str): The execution type ('MARKET' or 'LIMIT'). Defaults to 'MARKET'.
            valid (str, optional): The validity period for the order. Defaults to None.

        Returns:
            Order: The created sell order.
        """
        order = Order(
            broker=self,
            code=code,
            quantity=quantity,
            limit=limit,
            trigger=trigger,
            ordtype=exectype,
            side=Order.SELL,
            time=self._time or pd.to_datetime('now'),
            valid=valid,
        )
        self.submit(order)
        return order

    def cancel(self, order: Order | str) -> None:
        """
        Cancels an order.

        Args:
            order (Order): The order to be canceled.
        """
        if isinstance(order, str):
            order = self.get_order(order)
        order.cancel()
    
    def close(
        self,
        code: str,
        limit: float = None,
        trigger: float = None,
        exectype: str = Order.MARKET,
        valid: str = None,
    ) -> Order:
        """Close a position by selling all shares of a stock. Refer to sell() for details."""
        quantity = self._positions.get(code, 0)
        return self.sell(code, quantity, limit, trigger, exectype, valid)

    def submit(self, order: Order) -> None:
        """
        Submits an order for processing.

        Args:
            order (Order): The order to be submitted.
        """
        self._pendings.put(order)
        self._ordict[order.ordid] = order
        order.status = order.SUBMITTED

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

    def update(self, time: str, market: pd.DataFrame) -> None:
        """
        Updates the broker's state for a new trading day.

        Args:
            time (str): The current trading day.
        """
        self._time = pd.to_datetime(time or 'now')
        self._pendings.put(None)  # Placeholder for end-of-day processing.
        order = self._pendings.get()
        while order is not None:
            self._match(order, market)
            if not order.is_alive():
                self._orders.append(order)
            else:
                self._pendings.put(order)
            order = self._pendings.get()

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
            if order.ordtype == order.STOP or order.ordtype == order.STOPLIMIT:
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
        if order.ordtype == order.MARKET or order.ordtype == order.STOP:
            price = data.loc[order.code, "open"]
            quantity = min(data.loc[order.code, "volume"], order.quantity - order.filled)
        # If the order type is limit or stop limit order, check if the price conditions are met
        elif order.ordtype == order.LIMIT or order.ordtype == order.STOPLIMIT:
            if order.side == order.BUY and data.loc[order.code, "low"] <= order.limit:
                price = order.limit
                quantity = min(data.loc[order.code, "volume"], order.quantity - order.filled)
            elif order.side == order.SELL and data.loc[order.code, "high"] >= order.limit:
                price = order.limit
                quantity = min(data.loc[order.code, "volume"], order.quantity - order.filled)
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
        if order.side == order.BUY:
            amount = price * quantity
            commission = amount * self.commission
            cost = amount + commission
            if cost > self._balance:
                order.status = order.REJECTED
            else:
                order.execute(price, quantity)
                self._post(
                    time=self.time, code=order.code, ttype=order.side, 
                    unit=quantity, amount=-amount, price=price, commission=commission
                )
                self._balance -= cost
                self._positions[order.code] = self._positions.get(order.code, 0) + quantity
        elif order.side == order.SELL:
            amount = price * quantity
            commission = amount * self.commission
            revenue = amount - commission
            if self._positions.get(order.code, 0) < quantity:
                order.status = order.REJECTED
            else:
                order.execute(price, quantity)
                self._post(
                    time=self.time, code=order.code, ttype=order.side, 
                    unit=-quantity, amount=amount, price=price, commission=commission
                )
                order.commission = price * quantity * self.commission
                self._balance += revenue
                self._positions[order.code] -= quantity
                if self._positions[order.code] == 0:
                    del self._positions[order.code]
    
    def get_value(self, market: pd.DataFrame) -> float:
        """
        Returns the current value of the broker's portfolio.

        Returns:
            float: The current value of the portfolio.
        """
        return self.balance + (self.positions * market["close"]).sum()

    def get_order(self, ordid: str) -> Order:
        """
        Retrieves an order by its order ID.

        Args:
            ordid (str): The order ID of the order to retrieve.

        Returns:
            Order: The order with the specified order ID.
        """
        return self._ordict[ordid]

    def get_orders(self, alive: bool = False) -> pd.DataFrame:
        """
        Compiles a detailed trade log from the order history.

        Returns:
            pd.DataFrame: A DataFrame containing structured trade details for all orders.
        """
        orders = list(self._orders)
        if alive:
            orders += list(self.pendings.queue)
        trade_log = [order.to_dict() for order in orders]
        return pd.DataFrame(trade_log)
    
    def dump(self, since: pd.Timestamp = None) -> dict:
        """
        Serialize the Broker instance to a dictionary for JSON storage.
        """
        since = pd.to_datetime(since or 0)
        ledger = self.ledger
        orders = self.orders
        pendings = self.pendings
        if not self.ledger.empty:
            ledger["time"] = ledger["time"].dt.strftime('%Y-%m-%dT%H:%M:%S')
        return {
            "brokid": self.brokid,
            "balance": self._balance,
            "positions": self._positions,
            "ledger": (
                ledger[pd.to_datetime(ledger["time"]) >= since].replace(np.nan, None).to_dict(orient="records")
                if not ledger.empty else []
            ),
            "orders": (
                orders[pd.to_datetime(orders["cretime"]) >= since].replace(np.nan, None).to_dict(orient="records")
                if not orders.empty else []
            ),
            "pendings": pendings.replace(np.nan, None).to_dict(orient="records"),
            "commission": self.commission,
            "time": self._time.isoformat() if self._time else None,
            "container": self.container,
        }
    
    @classmethod
    def load(cls, data: dict) -> 'Broker':
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
        broker = cls(brokid=data["brokid"], commission=data["commission"])

        # Restore basic attributes
        broker._time = pd.Timestamp(data["time"]) if data["time"] else None
        broker._balance = data["balance"]
        broker._positions = data["positions"]
        ledger = pd.DataFrame(data["ledger"])
        if not ledger.empty:
            ledger["time"] = pd.to_datetime(ledger["time"])
            broker._ledger = ledger.to_dict(orient="records")

        # Restore orders
        for order_data in data["orders"]:
            order = Order.load(order_data, broker)
            broker._orders.append(order)
            broker._ordict[order.ordid] = order

        # Restore pending orders
        for order_data in data["pendings"]:
            order = Order.load(order_data, broker)
            broker._pendings.put(order)
            broker._ordict[order.ordid] = order
        
        # Extra fields
        for key, value in data.items():
            if key not in ["brokid", "commission", "time", "balance", "positions", "ledger", "orders", "pendings"]:
                setattr(broker, key, value)

        return broker

    def store(self, path: str, since: pd.Timestamp = None) -> None:
        """
        Stores the broker's state to a JSON file.

        Args:
            path (str): The file path where the broker's state will be saved.
        """
        with open(path, "w") as f:
            json.dump(self.dump(since=since), f, indent=4, ensure_ascii=False)
        
    @classmethod
    def restore(cls, path: str) -> None:
        """
        Restores the broker's state from a JSON file.

        Args:
            path (str): The file path from which the broker's state will be loaded.
        """
        with open(path, "r") as f:
            data = json.load(f)
            broker = cls.load(data)
        return broker

    def report(self, market: pd.DataFrame):
        """
        Generates a report of the broker's performance.

        Returns:
            dict: A dictionary containing the broker's performance metrics.
        """
        ledger = self.ledger.set_index(["time", "code"]).sort_index()
        prices = market["close"].unstack("code")

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
            "values": pd.concat(
                [total, market, cash, turnover], 
                axis=1, keys=["total", "market", "cash", "turnover"]
            ),
            "positions": positions,
            "trades": trades,
        }

    def evaluate(self, market: pd.DataFrame, benchmark: pd.Series = None):
        """
        Evaluates the broker's performance based on its current state.

        Args:
            benchmark (pd.Series, optional): A benchmark series for comparison. Defaults to None.

        Returns:
            dict: A dictionary containing the broker's performance metrics.
        """
        report = self.report(market)
        return {
            "evaluation": evaluate(
                report["values"]["total"], 
                benchmark=benchmark, 
                turnover=report["values"]["turnover"], 
                trades=report["trades"]
            ),
            "values": report["values"],
            "positions": report["positions"],
            "trades": report["trades"],
        }

    def __str__(self) -> str:
        """
        Provides a string representation of the broker's state.

        Returns:
            str: A summary of the broker's balance, positions, and orders.
        """
        return f"Broker[{self.time}] ${self.balance:.2f}x{self.commission * 1e4:.2f}Bps |#{len(self.pendings)} Pending #{len(self.orders)} Orders| \n{self.positions}\n"
    
    def __repr__(self):
        return self.__str__()


class ManagerBroker(Broker):

    def __init__(
        self, 
        manager: ParquetManager = None,
        principle: float = 1_000_000, 
        commission: float = 0.001,
    ):
        """
        Initializes the base broker with essential attributes.

        Args:
            principle (float): Initial cash balance for the broker. Defaults to 1,000,000.
            commission (float): Commission rate for transactions. Defaults to 0.001 (0.1%).
        """
        super().__init__(
            market=None, 
            principle=principle, 
            commission=commission, 
        )
        self.manager = manager

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

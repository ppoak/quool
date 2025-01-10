import uuid
import json
import queue
import logging
import numpy as np
import pandas as pd
from .manager import ParquetManager
from .tool import setup_logger, evaluate


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

    def __init__(
        self,
        market: pd.DataFrame,
        commission: float = 0.001,
        logger: logging.Logger = None,
    ):
        """
        Initializes the broker with essential attributes.

        Args:
            market (pd.DataFrame): The market data for trading.
            principle (float): Initial cash balance for the broker. Defaults to 1,000,000.
            commission (float): Commission rate for transactions. Defaults to 0.001 (0.1%).
        """
        self.market = market
        self._time = None
        self._balance = 0
        self._positions = {}
        self.commission = commission
        self._pendings = queue.Queue()
        self._ledger = [] # Key parameter to restore the state of the broker
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
        self._post(time=pd.to_datetime(time), code="CASH", ttype="TRANSFER", unit=0, amount=amount, price=0, commission=0)

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
            trigger (float, optional): The trigger price for the order. Defaults to None.
            limit (float, optional): The limit price for the order. Defaults to None.
            exectype (str): The execution type ('MARKET' or 'LIMIT'). Defaults to 'MARKET'.
            valid (str, optional): The validity period for the order. Defaults to None.

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

    def cancel(self, order: Order | str) -> None:
        """
        Cancels an order.

        Args:
            order (Order): The order to be canceled.
        """
        if isinstance(order, str):
            order = self.get_order(order)
        order.cancel()
        self.logger.debug(f"Order canceled: {order}")
    
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
        self.logger.debug(f"Order submitted: {order}")

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

    def _load(self) -> pd.DataFrame:
        """
        Loads market data required for processing orders.

        Returns:
            pd.DataFrame: A DataFrame containing market data (open, high, low, close, volume) indexed by stock codes.
        """
        return self.market.loc[self._time]

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
                    self.logger.debug(f"Order triggered: {order}")
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
                self.logger.warning(f"Insufficient balance: {order}")
            else:
                order.execute(price, quantity)
                self._post(
                    time=self.time, code=order.code, ttype=order.side, 
                    unit=quantity, amount=-amount, price=price, commission=commission
                )
                self._balance -= cost
                self._positions[order.code] = self._positions.get(order.code, 0) + quantity
                self.logger.info(f"Order executed: {order}")
        elif order.side == order.SELL:
            amount = price * quantity
            commission = amount * self.commission
            revenue = amount - commission
            if self._positions.get(order.code, 0) < quantity:
                order.status = order.REJECTED
                self.logger.warning(f"Insufficient position: {order}")
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
    
    def dump(self):
        """
        Serialize the Broker instance to a dictionary for JSON storage.
        """
        ledger = pd.DataFrame(self._ledger)
        ledger["time"] = ledger["time"].dt.strftime('%Y-%m-%dT%H:%M:%S')
        return {
            "balance": self._balance,
            "positions": self._positions,
            "ledger": ledger.to_dict(orient="records"),
            "orders": [order.dump() for order in self._orders],
            "pendings": [order.dump() for order in list(self._pendings.queue)],
            "commission": self.commission,
            "time": self._time.isoformat() if self._time else None,
        }
    
    @classmethod
    def load(cls, data: dict, market: pd.DataFrame, logger = None) -> 'Broker':
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
        broker = cls(market=market, commission=data["commission"], logger=logger)

        # Restore basic attributes
        broker._time = pd.Timestamp(data["time"]) if data["time"] else None
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

    def store(self, path: str) -> None:
        """
        Stores the broker's state to a JSON file.

        Args:
            path (str): The file path where the broker's state will be saved.
        """
        with open(path, "w") as f:
            json.dump(self.dump(), f, indent=4, ensure_ascii=False)
        
    @classmethod
    def restore(cls, path: str, market_data: pd.DataFrame, logger = None) -> None:
        """
        Restores the broker's state from a JSON file.

        Args:
            path (str): The file path from which the broker's state will be loaded.
        """
        with open(path, "r") as f:
            data = json.load(f)
            broker = cls.load(data, market_data, logger)
        return broker

    def report(self):
        """
        Generates a report of the broker's performance.

        Returns:
            dict: A dictionary containing the broker's performance metrics.
        """
        ledger = self.ledger
        ledger = ledger.set_index(["time", "code"]).sort_index()
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
        trades["duration"] = trades["close_at"] - trades["open_at"]
        trades["return"] = (trades["close_amount"] - trades["open_amount"]) / trades["open_amount"]
        return {
            "values": pd.concat(
                [total, market, cash, turnover], 
                axis=1, keys=["total", "market", "cash", "turnover"]
            ),
            "positions": positions,
            "trades": trades,
        }

    def evaluate(self, benchmark: pd.Series = None):
        """
        Evaluates the broker's performance based on its current state.

        Args:
            benchmark (pd.Series, optional): A benchmark series for comparison. Defaults to None.

        Returns:
            dict: A dictionary containing the broker's performance metrics.
        """
        report = self.report()
        return evaluate(report["total_value"], benchmark=benchmark, turnover=report["turnover"], trades=report["trades"])

    def __str__(self) -> str:
        """
        Provides a string representation of the broker's state.

        Returns:
            str: A summary of the broker's balance, positions, and orders.
        """
        return f"Broker[{self.time}] ${self.balance}x{self.commission} |#{self._pendings.qsize()} Pending #{len(self.orders)} Orders| \n{self.positions}\n"
    
    def __repr__(self):
        return self.__str__()


class ManagerBroker(Broker):

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
        super().__init__(
            market=None, 
            principle=principle, 
            commission=commission, 
            logger=logger
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

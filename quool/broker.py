import pandas as pd
from .base import BrokerBase, OrderBase


class Broker(BrokerBase):
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
    ) -> OrderBase:
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
    ) -> OrderBase:
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

    def _match(self, order: OrderBase, data: pd.DataFrame) -> None:
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

    def _execute(self, order: OrderBase, price: float, quantity: int) -> None:
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
                order.execute(price, quantity)
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
                order.execute(price, quantity)
                order.commission = getattr(order, "commission", 0) + commission
                self._post(
                    time=self.time, code=order.code, ttype=order.side, 
                    unit=-quantity, amount=amount, price=price, commission=commission
                )
                self._balance += revenue
                self._positions[order.code] -= quantity
                if self._positions[order.code] == 0:
                    del self._positions[order.code]
    

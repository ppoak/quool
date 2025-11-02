import pandas as pd
from quool import Broker


class XtBroker(Broker):
    """Wrapper around xtquant's XtQuantTrader providing trading and portfolio queries.

    XtBroker integrates with the xtquant trading system to submit stock orders
    and query account assets, orders, and positions. It enforces A-share lot-size
    rules by flooring quantities to multiples of 100 shares for submissions.

    Attributes:
      BUY (int): xtconstant code for stock buy.
      SELL (int): xtconstant code for stock sell.
      MARKET (int): xtconstant code for market (peer price) execution.
      LIMIT (int): xtconstant code for limit price execution.
      trader (xtquant.xttrader.XtQuantTrader): Trader instance.
      account (xtquant.xttype.StockAccount): Account object representing the trading account.

    Notes:
      - Requires the xtquant package and a running XtQuantTrader gateway.
      - Quantities are adjusted to 100-share lots before order submission.
    """

    def __init__(self, account: str, path: str, id: str = None):
        """Initialize the XtBroker and establish a connection to XtQuantTrader.

        Creates the XtQuantTrader instance, sets up the account, starts and connects
        to the trader. Raises if the connection is not established.

        Args:
          account (str): Account identifier recognized by xtquant (e.g., '123456').
          path (str): Working directory path used by XtQuantTrader.
          id (str, optional): Local process id for the trader. Defaults to the current
            timestamp if not provided.

        Raises:
          RuntimeError: If the XtQuantTrader fails to connect.
        """
        from xtquant import xttrader, xttype, xtconstant

        self.BUY = xtconstant.STOCK_BUY
        self.SELL = xtconstant.STOCK_SELL
        self.MARKET = xtconstant.MARKET_PEER_PRICE_FIRST
        self.LIMIT = xtconstant.FIX_PRICE

        id = int(id or pd.Timestamp.now().timestamp())
        self.trader = xttrader.XtQuantTrader(path, id)
        self.account = xttype.StockAccount(account)
        self.trader.start()
        self.trader.connect()
        if not self.trader.connected:
            raise RuntimeError("XtQuantTrader connect failed")

    @property
    def balance(self):
        """Return available cash balance.

        Returns:
          float: Available cash in the trading account.
        """
        return self.trader.query_stock_asset(self.account).cash

    @property
    def frozen(self):
        """Return frozen (reserved) cash balance.

        Returns:
          float: Frozen cash amount.
        """
        return self.trader.query_stock_asset(self.account).frozen_cash

    @property
    def value(self):
        """Return total account asset value.

        Returns:
          float: Total asset value (cash + market value), as reported by xtquant.
        """
        return self.trader.query_stock_asset(self.account).total_asset

    @property
    def market(self):
        """Return current market value of holdings.

        Returns:
          float: Market value of all positions.
        """
        return self.trader.query_stock_asset(self.account).market_value

    @property
    def orders(self):
        """Return all orders (including non-cancelable) from the account.

        Returns:
          list: List of xtquant order objects with fields such as order_id, stock_code,
          order_time, order_volume, traded_volume, price_type, price, order_status, etc.
        """
        return self.trader.query_stock_orders(self.account, cancelable_only=False)

    @property
    def pendings(self):
        """Return cancelable (pending) orders from the account.

        Returns:
          list: List of xtquant order objects that can be canceled.
        """
        return self.trader.query_stock_orders(self.account, cancelable_only=True)

    @property
    def positions(self):
        """Return current positions from the account.

        Returns:
          list: List of xtquant position objects with fields such as stock_code, volume,
          can_use_volume, open_price, market_value, avg_price, etc.
        """
        return self.trader.query_stock_positions(self.account)

    def create(
        self,
        type: int,
        code: str,
        quantity: float,
        exectype: int,
        price: float,
        remark: str,
    ):
        """Submit a stock order through XtQuantTrader.

        Floors quantity to the nearest 100-share lot. If the adjusted quantity is
        positive, submits the order using xtquant APIs.

        Args:
          type (int): Order direction (BUY or SELL), using xtconstant values.
          code (str): Stock code recognized by xtquant (e.g., 'SH600519').
          quantity (float): Requested number of shares; will be floored to 100-share lots.
          exectype (int): Execution type (MARKET or LIMIT), using xtconstant values.
          price (float): Price parameter; for MARKET may be ignored by the system, for LIMIT
            used as the limit price.
          remark (str): Optional order remark or strategy tag.

        Returns:
          Any: Return value from xtquant's order submission (e.g., order_id or status object).
          None: If the adjusted quantity is not positive.

        Raises:
          RuntimeError: If the trader is not connected.
          ValueError: If inputs are invalid for the xtquant submission API.
        """
        quantity = quantity // 100 * 100
        if quantity > 0:
            return self.trader.order_stock(
                self.account, code, type, quantity, exectype, price, order_remark=remark
            )

    def buy(self, code: str, quantity: float, price: float = 0, remark: str = ""):
        """Submit a market buy order.

        Args:
          code (str): Stock code.
          quantity (float): Requested shares; floored to 100-share lots.
          price (float, optional): Price parameter; typically 0 for market. Defaults to 0.
          remark (str, optional): Optional order remark. Defaults to "".

        Returns:
          Any: xtquant order submission result.
          None: If adjusted quantity is not positive.
        """
        return self.create(self.BUY, code, quantity, self.MARKET, price, remark)

    def sell(self, code: str, quantity: float, price: float = 0, remark: str = ""):
        """Submit a market sell order.

        Args:
          code (str): Stock code.
          quantity (float): Requested shares; floored to 100-share lots.
          price (float, optional): Price parameter; typically 0 for market. Defaults to 0.
          remark (str, optional): Optional order remark. Defaults to "".

        Returns:
          Any: xtquant order submission result.
          None: If adjusted quantity is not positive.
        """
        return self.create(self.SELL, code, quantity, self.MARKET, price, remark)

    def close(self, code: str, price: float = 0, remark: str = ""):
        """Close (sell) the available position for the given stock.

        Looks up the current position and sells the usable volume via a market order.
        If no usable volume is available, no order is submitted.

        Args:
          code (str): Stock code to close.
          price (float, optional): Price parameter; typically 0 for market. Defaults to 0.
          remark (str, optional): Optional order remark. Defaults to "".

        Returns:
          Any: xtquant order submission result.
          None: If no usable position volume exists.
        """
        for pos in self.positions:
            if pos.stock_code == code:
                quantity = pos.can_use_volume
                if quantity > 0:
                    return self.create(
                        self.SELL, code, quantity, self.MARKET, price, remark
                    )

    def limit_buy(self, code: str, quantity: float, price: float, remark: str = ""):
        """Submit a limit buy order.

        Args:
          code (str): Stock code.
          quantity (float): Requested shares; floored to 100-share lots.
          price (float): Limit price.
          remark (str, optional): Optional order remark. Defaults to "".

        Returns:
          Any: xtquant order submission result.
          None: If adjusted quantity is not positive.
        """
        return self.create(self.BUY, code, quantity, self.LIMIT, price, remark)

    def limit_sell(self, code: str, quantity: float, price: float, remark: str = ""):
        """Submit a limit sell order.

        Args:
          code (str): Stock code.
          quantity (float): Requested shares; floored to 100-share lots.
          price (float): Limit price.
          remark (str, optional): Optional order remark. Defaults to "".

        Returns:
          Any: xtquant order submission result.
          None: If adjusted quantity is not positive.
        """
        return self.create(self.SELL, code, quantity, self.LIMIT, price, remark)

    def limit_close(self, code: str, price: float, remark: str = ""):
        """Close (sell) the available position using a limit order.

        Sells the usable volume at the specified limit price. If no usable volume
        is available, no order is submitted.

        Args:
          code (str): Stock code to close.
          price (float): Limit price.
          remark (str, optional): Optional order remark. Defaults to "".

        Returns:
          Any: xtquant order submission result.
          None: If no usable position volume exists.
        """
        for pos in self.positions:
            if pos.stock_code == code:
                quantity = pos.can_use_volume
                if quantity > 0:
                    return self.create(
                        self.SELL, code, quantity, self.LIMIT, price, remark
                    )

    def update(self, time: str, data: pd.DataFrame):
        """Optional hook to update broker state with time and market data.

        Provided for API compatibility with other broker interfaces; not implemented.

        Args:
          time (str): Current timestamp or time-label string.
          data (pandas.DataFrame): Market data snapshot.

        Returns:
          None
        """
        pass

    def get_orders(self):
        """Return all orders as a pandas DataFrame.

        Extracts fields from xtquant order objects into a tabular format, including:
        - account_type, account_id, stock_code, order_id, order_sysid, order_time,
        - order_type, order_volume, price_type, price, traded_volume, traded_price,
        - order_status, status_msg, strategy_name, order_remark, direction, offset_flag.

        Returns:
          pandas.DataFrame: Table of orders with the fields listed above.
        """
        orders = []
        for order in self.orders:
            orders.append(
                {
                    "account_type": order.account_type,
                    "account_id": order.account_id,
                    "stock_code": order.stock_code,
                    "order_id": order.order_id,
                    "order_sysid": order.order_sysid,
                    "order_time": order.order_time,
                    "order_type": order.order_type,
                    "order_volume": order.order_volume,
                    "price_type": order.price_type,
                    "price": order.price,
                    "traded_volume": order.traded_volume,
                    "traded_price": order.traded_price,
                    "order_status": order.order_status,
                    "status_msg": order.status_msg,
                    "strategy_name": order.strategy_name,
                    "order_remark": order.order_remark,
                    "direction": order.direction,
                    "offset_flag": order.offset_flag,
                }
            )
        return pd.DataFrame(orders)

    def get_pendings(self):
        """Return cancelable (pending) orders as a pandas DataFrame.

        Extracts fields from xtquant order objects into a tabular format analogous
        to get_orders(), limited to orders that are cancelable.

        Returns:
          pandas.DataFrame: Table of pending orders with the same fields as get_orders().
        """
        pendings = []
        for order in self.pendings:
            pendings.append(
                {
                    "account_type": order.account_type,
                    "account_id": order.account_id,
                    "stock_code": order.stock_code,
                    "order_id": order.order_id,
                    "order_sysid": order.order_sysid,
                    "order_time": order.order_time,
                    "order_type": order.order_type,
                    "order_volume": order.order_volume,
                    "price_type": order.price_type,
                    "price": order.price,
                    "traded_volume": order.traded_volume,
                    "traded_price": order.traded_price,
                    "order_status": order.order_status,
                    "status_msg": order.status_msg,
                    "strategy_name": order.strategy_name,
                    "order_remark": order.order_remark,
                    "direction": order.direction,
                    "offset_flag": order.offset_flag,
                }
            )

        return pd.DataFrame(pendings)

    def get_positions(self):
        """Return current positions as a pandas DataFrame.

        Extracts fields from xtquant position objects, including:
        - account_type, account_id, stock_code, volume, can_use_volume,
        - open_price, market_value, frozen_volume, on_road_volume,
        - yesterday_volume, avg_price, direction.

        Returns:
          pandas.DataFrame: Table of positions with the fields listed above.
        """
        positions = []
        for position in self.positions:
            positions.append(
                {
                    "account_type": position.account_type,
                    "account_id": position.account_id,
                    "stock_code": position.stock_code,
                    "volume": position.volume,
                    "can_use_volume": position.can_use_volume,
                    "open_price": position.open_price,
                    "market_value": position.market_value,
                    "frozen_volume": position.frozen_volume,
                    "on_road_volume": position.on_road_volume,
                    "yesterday_volume": position.yesterday_volume,
                    "avg_price": position.avg_price,
                    "direction": position.direction,
                }
            )

        return pd.DataFrame(positions)

    def get_value(self, data):
        """Compute mark-to-market portfolio value using provided close prices.

        Multiplies position volumes (from get_positions()) by the 'close' column
        of the provided DataFrame and sums across instruments.

        Args:
          data (pandas.DataFrame): Market data with a 'close' column indexed by stock_code.

        Returns:
          float: Total mark-to-market value of current positions.

        Raises:
          KeyError: If 'close' column or required stock codes are missing from data.
        """
        return (
            self.get_positions().set_index("stock_code")["volume"] * data["close"]
        ).sum()

    def __str__(self):
        return (
            f"XtBroker(#{self.account.account_id}\n"
            f"  ${self.value}=${self.market}(market)+${self.balance}(balance)+${self.frozen}(frozen))\n"
            f"  {dict([(position.stock_code, position.volume) for position in self.positions])}\n"
            f")"
        )

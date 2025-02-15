from .base import OrderBase


class Order(OrderBase):
    """A implement for representing a financial trading order.

    This class provides core functionality for order management including execution, cancellation,
    status tracking, and serialization. It defines standard order types, statuses, and directions.

    Class Attributes:
        Order Status Constants:
            CREATED (str): Order created but not yet submitted
            SUBMITTED (str): Order submitted to market but not executed
            PARTIAL (str): Order partially filled
            FILLED (str): Order completely filled
            CANCELED (str): Order canceled by user
            EXPIRED (str): Order expired due to validity period
            REJECTED (str): Order rejected by exchange

        Order Type Constants:
            MARKET (str): Market order type
            LIMIT (str): Limit order type
            STOP (str): Stop order type
            STOPLIMIT (str): Stop-limit order type

        Direction Constants:
            BUY (str): Buy direction
            SELL (str): Sell direction

    Attributes:
        id (str): Unique order identifier (auto-generated if not provided)
        broker (BrokerBase): Associated broker instance handling the order
        creatime (datetime): Order creation time (parsed from ISO string)
        side (str): Trade direction (BUY/SELL)
        code (str): Financial instrument code/symbol
        quantity (int): Total order quantity
        exectype (str): Order execution type (MARKET/LIMIT/STOP/STOPLIMIT)
        limit (float): Limit price for limit orders (None for market orders)
        trigger (float): Trigger price for stop orders
        status (str): Current order status (from status constants)
        filled (int): Number of shares/contracts filled
        value (float): Total executed value (filled * execution price)
        exectime (datetime): Time of last execution (None if not executed)
        execprice (float): Price of last execution (None if not executed)
        commission (float): Accumulated commission fees
        valid (datetime): Order validity expiration time (None for GTC orders)

    Args:
        broker: Broker instance responsible for order execution
        time: Order creation time (ISO 8601 format string)
        side: Trade direction (use class constants BUY/SELL)
        code: Trading instrument identifier
        quantity: Total order quantity (must be > 0)
        exectype: Order type (default: MARKET)
        limit: Limit price for LIMIT/STOPLIMIT orders
        trigger: Trigger price for STOP/STOPLIMIT orders
        id: Custom order ID (auto-generated if None)
        valid: Order validity period (ISO 8601 format string)

    Methods:
        execute(price, quantity): Execute order at specified price/quantity
        cancel(): Cancel the order if cancellable
        is_alive(): Check if order is still active/executable
        dump(): Serialize order to dictionary
        load(data, broker): Classmethod to reconstruct order from dict
    """

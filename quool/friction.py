import pandas as pd
from .order import Order


class FixedRateCommission:
    """Flat-rate commission and stamp duty calculator.

    This model computes transaction costs using a fixed commission rate with a
    minimum fee, and applies an additional stamp duty for SELL orders.

    Attributes:
      commission_rate (float): Proportional commission rate applied to trade notional.
      min_commission (float): Minimum commission fee per transaction.
      stamp_duty_rate (float): Additional duty applied to SELL trade notional.
    """

    def __init__(
        self,
        commission_rate: float = 0.0005,
        stamp_duty_rate: float = 0.001,
        min_commission: float = 5,
    ):
        """Initialize the fixed-rate commission model.

        Args:
          commission_rate (float, optional): Proportional commission rate applied
            to the traded amount (price * quantity). Defaults to 0.0005.
          stamp_duty_rate (float, optional): Stamp duty rate applied to SELL trades
            on the traded amount. Defaults to 0.001.
          min_commission (float, optional): Minimum commission charged per trade.
            Defaults to 5.
        """

        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_duty_rate = stamp_duty_rate

    def __call__(self, order: Order, price: float, quantity: float):
        """Compute commission for a given order fill.

        BUY:
          commission = max(commission_rate * amount, min_commission)

        SELL:
          commission = max(commission_rate * amount, min_commission) + stamp_duty_rate * amount

        where amount = price * quantity.

        Args:
          order (Order): The order being filled. Uses order.type to distinguish BUY vs. SELL.
          price (float): Executed price per unit.
          quantity (float): Executed quantity.

        Returns:
          float: The computed commission cost.
        """
        amount = price * quantity
        if order.type == order.BUY:
            return max(self.commission_rate * amount, self.min_commission)
        else:
            return (
                max(self.commission_rate * amount, self.min_commission)
                + self.stamp_duty_rate * amount
            )

    def __str__(self):
        """Return a human-readable summary of the commission model.

        Returns:
          str: Summary string including rates and minimum commission.
        """
        return f"{self.__class__.__name__}: rate={self.commission_rate}, stamp_duty={self.stamp_duty_rate}, min={self.min_commission}"

    def __repr__(self):
        """Return the official string representation of the commission model.

        This is identical to __str__ for convenience.

        Returns:
          str: The string representation.
        """
        return self.__str__()


class FixedRateSlippage:
    """Fixed-rate slippage model producing executed price and fill quantity.

    The model determines an executable price and quantity given an order and a
    market OHLCV row (kline). The fill quantity is limited by both available
    market volume and the order's remaining quantity.

    For MARKET orders:
      - BUY: price = min(high, open * (1 + slip_rate))
      - SELL: price = max(low, ((low - high) / volume) * quantity * slip_rate + open)

    For LIMIT orders:
      - BUY: price is capped by the kline high and the order's limit price.
      - SELL: price is floored by the kline low and the order's limit price.

    If the computed quantity is zero (no available volume or order already fully
    filled), returns (0, 0).

    Attributes:
      slip_rate (float): Slippage rate influencing the deviation from open/limit prices.
    """

    def __init__(self, slip_rate: float = 0.01):
        """Initialize the fixed-rate slippage model.

        Args:
          slip_rate (float, optional): Slippage rate used to adjust executed prices.
            Defaults to 0.01.
        """
        self.slip_rate = slip_rate

    def __call__(self, order: Order, kline: pd.Series) -> float:
        """Compute executed price and quantity for an order given market data.

        The kline series must include at least the keys: 'open', 'high', 'low', and 'volume'.
        Quantity is determined as:
          quantity = min(kline['volume'], order.quantity - order.filled)

        Pricing logic varies by order side and execution type as described in the
        class docstring. LIMIT pricing uses the order-defined limit along with the
        kline high/low bounds.

        Args:
          order (Order): The order to evaluate; uses side (BUY/SELL), execution type
            (MARKET/LIMIT), total quantity, and already filled quantity.
          kline (pandas.Series): OHLCV snapshot for the instrument with keys
            'open', 'high', 'low', 'volume'.

        Returns:
          tuple[float, int]: A (price, quantity) pair. Returns (0, 0) if no fill can occur.
        """
        quantity = min(kline["volume"], order.quantity - order.filled)
        if quantity == 0:
            return 0, 0
        if order.type == order.BUY:
            if (
                order.exectype == order.MARKET
                or order.exectype == order.STOP
                or order.exectype == order.TARGET
            ):
                return (
                    min(
                        kline["high"],
                        kline["open"] * (1 + self.slip_rate),
                    ),
                    quantity,
                )
            elif (
                order.exectype == order.LIMIT
                or order.exectype == order.STOPLIMIT
                or order.exectype == order.TARGETLIMIT
            ):
                return min(order.limit, kline["high"]), quantity
        else:
            if (
                order.exectype == order.MARKET
                or order.exectype == order.STOP
                or order.exectype == order.TARGET
            ):
                return (
                    max(
                        kline["low"],
                        (kline["low"] - kline["high"])
                        / kline["volume"]
                        * quantity
                        * self.slip_rate
                        + kline["open"],
                    ),
                    quantity,
                )
            elif (
                order.exectype == order.LIMIT
                or order.exectype == order.STOPLIMIT
                or order.exectype == order.TARGETLIMIT
            ):
                return max(order.limit, kline["low"]), quantity

    def __str__(self):
        return f"{self.__class__.__name__}(slip_one_cent_rate={self.slip_rate})"

    def __repr__(self):
        return self.__str__()

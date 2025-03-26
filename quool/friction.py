import pandas as pd
from .order import Order


class FixedRateCommission:

    def __init__(
        self,
        commission_rate: float = 0.0005,
        stamp_duty_rate: float = 0.001,
        min_commission: float = 5,
    ):
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_duty_rate = stamp_duty_rate

    def __call__(self, order: Order, price: float, quantity: float):
        amount = price * quantity
        if order.type == order.BUY:
            return max(self.commission_rate * amount, self.min_commission)
        else:
            return (
                max(self.commission_rate * amount, self.min_commission)
                + self.stamp_duty_rate * amount
            )

    def __str__(self):
        return f"{self.__class__.__name__}: rate={self.commission_rate}, stamp_duty={self.stamp_duty_rate}, min={self.min_commission}"

    def __repr__(self):
        return self.__str__()


class FixedRateSlippage:

    def __init__(self, slip_rate: float = 0.01):
        self.slip_rate = slip_rate

    def __call__(self, order: Order, kline: pd.Series) -> float:
        quantity = min(kline["volume"], order.quantity - order.filled)
        if quantity == 0:
            return 0, 0
        if order.type == order.BUY:
            if order.exectype == order.MARKET:
                return (
                    min(
                        kline["high"],
                        (kline["high"] - kline["low"])
                        / kline["volume"]
                        * quantity
                        * self.slip_rate
                        + kline["open"],
                    ),
                    quantity,
                )
            elif order.exectype == order.LIMIT:
                return min(order.price, kline["high"]), quantity
        else:
            if order.exectype == order.MARKET:
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
            elif order.exectype == order.LIMIT:
                return max(order.price, kline["low"]), quantity

    def __str__(self):
        return f"{self.__class__.__name__}(slip_one_cent_rate={self.slip_rate})"

    def __repr__(self):
        return self.__str__()

import pandas as pd
from quool.order import Order
from quool.friction import CommissionBase, SlippageBase


class FixedRateCommission(CommissionBase):

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
        if order.side == order.BUY:
            return max(self.commission_rate * amount, self.min_commission)
        else:
            return max(self.commission_rate * amount, self.min_commission) + self.stamp_duty_rate * amount
    
    def __str__(self):
        return f"{self.__class__.__name__}: rate={self.commission_rate}, stamp_duty={self.stamp_duty_rate}, min={self.min_commission}"


class FixedRateSlippage(SlippageBase):
    
    def __init__(self, slip_rate: float = 0.01):
        self.slip_rate = slip_rate

    def __call__(self, order: Order, quantity: float, data: pd.Series) -> float:
        if order.side == order.BUY:
            if order.exectype == order.MARKET:
                return min(data["high"], (data["high"] - data["low"]) / data["volume"] * quantity * self.slip_rate + data["open"])
            elif order.exectype == order.LIMIT:
                return min(order.price, data["high"])
        else:
            if order.exectype == order.MARKET:
                return max(data["low"], (data["low"] - data["high"]) / data["volume"] * quantity * self.slip_rate + data["open"])
            elif order.exectype == order.LIMIT:
                return max(order.price, data["low"])

    def __str__(self):
        return f"{self.__class__.__name__}(slip_one_cent_rate={self.slip_rate})"

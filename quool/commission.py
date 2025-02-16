from .base import CommissionBase, OrderBase


class AjustedCommission(CommissionBase):
    """Concrete commission model implementation for equity trading scenarios.

    Inherits from CommissionBase to provide a standard Chinese stock market
    commission structure with percentage-based fees and minimum thresholds.

    Key Features:
        - Percentage-based commission with minimum threshold
        - Stamp duty calculation for sell orders
        - Extensible structure for custom fee models

    Implementation Requirements for Subclasses:
        1. Must override __init__ with explicit parameters (no **kwargs)
        2. Must implement __call__ with (order, price, quantity) signature
        3. Must maintain immutability of fee rates after initialization
        4. Must handle both BUY and SELL order types appropriately

    Args:
        commission_rate (float): Brokerage fee percentage (0.0005 = 0.05%). 
            Must be ≥ 0.
        stamp_duty_rate (float): Government tax percentage on SELL orders.
            Must be ≥ 0. Default: 0.001 (0.1%)
        min_commission (float): Minimum fee per trade. Default: 5 RMB

    Attributes:
        commission_rate (float): Read-only access to brokerage rate
        stamp_duty_rate (float): Read-only access to stamp duty rate 
        min_commission (float): Read-only access to minimum fee

    Methods:
        __call__: Calculate total fees for an order execution

    Raises:
        ValueError: If any rate is negative
        TypeError: If input types are incorrect

    Example Usage:

        # 1. Basic instantiation
        standard_commission = Commission(
            commission_rate=0.0003,
            stamp_duty_rate=0.001,
            min_commission=5
        )

        # 2. Calculate fees for BUY order
        buy_order = Order(side=Order.BUY, quantity=1000, ...)
        buy_fee = standard_commission(
            order=buy_order, 
            price=10.50,  # Execution price
            quantity=1000  # Executed shares
        )
        # buy_fee = max(10.50*1000*0.0003, 5) = max(3.15, 5) = 5.0

        # 3. Calculate fees for SELL order
        sell_order = Order(side=Order.SELL, quantity=500, ...)
        sell_fee = standard_commission(
            order=sell_order,
            price=12.00,
            quantity=500
        )
        # sell_fee = max(12*500*0.0003, 5) + (12*500*0.001)
        #           = max(1.8, 5) + 6 = 5 + 6 = 11.0

    Custom Subclass Example:

        class FixedCommission(CommissionBase):
            \"""Fixed fee + percentage model\"""
            
            def __init__(self, fixed_fee: float, rate: float):
                self.fixed_fee = fixed_fee
                self.rate = rate
            
            def __call__(self, order, price, quantity):
                return self.fixed_fee + (price * quantity * self.rate)

        # Usage:
        fixed_commission = FixedCommission(fixed_fee=10, rate=0.0002)
        fee = fixed_commission(sell_order, 10.0, 1000)
        # fee = 10 + (10*1000*0.0002) = 10 + 2 = 12.0
    """

    def __init__(
        self, 
        commission_rate: float = 0.0005,
        stamp_duty_rate: float = 0.001,
        min_commission: float = 5,
    ):
        self.commission_rate = commission_rate
        self.min_commission = min_commission
        self.stamp_duty_rate = stamp_duty_rate

    def __call__(self, order: OrderBase, price: float, quantity: float):
        amount = price * quantity
        if order.side == order.BUY:
            return max(self.commission_rate * amount, self.min_commission)
        else:
            return max(self.commission_rate * amount, self.min_commission) + self.stamp_duty_rate * amount
    
    def __str__(self):
        return f"{self.__class__.__name__}: rate={self.commission_rate}, stamp_duty={self.stamp_duty_rate}, min={self.min_commission}"

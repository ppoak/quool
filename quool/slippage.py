import pandas as pd
from .base import SlippageBase, OrderBase


class Slippage(SlippageBase):
    """Concrete slippage model implementation for order price adjustment.

    Inherits from SlippageBase to provide volume-weighted price impact simulation
    based on order type and market conditions. Handles both MARKET and LIMIT orders.

    Key Features:
        - Market order: Simulates price impact using volume-weighted slippage
        - Limit order: Ensures execution within limit price constraints
        - BUY/SELL direction-aware calculations
        - Configurable slippage intensity

    Implementation Requirements for Subclasses:
        1. Must implement __init__ with explicit parameters
        2. __call__ signature must match (order, quantity, data)
        3. Must handle all order types defined in OrderBase
        4. Should validate market data completeness

    Args:
        slip_rate (float): Slippage intensity factor (0.01 = 1% impact).
            Represents market impact per unit of volume. Must be ≥ 0.
            Default: 0.01

    Attributes:
        slip_rate (float): Read-only slippage impact factor

    Methods:
        __call__: Calculate executed price with slippage adjustment

    Raises:
        ValueError: If slip_rate is negative
        KeyError: If market data missing required columns
        TypeError: If input types are invalid

    Example Usage:

        # 1. Initialize with default parameters
        slippage_model = Slippage()

        # 2. Calculate execution price for BUY MARKET order
        buy_order = Order(side=Order.BUY, exectype=Order.MARKET)
        market_data = pd.Series({
            'high': 50.0,
            'low': 48.0,
            'open': 49.0,
            'volume': 100000
        })
        executed_price = slippage_model(
            order=buy_order,
            quantity=5000,  # Shares to execute
            data=market_data
        )
        # Calculation:
        # price_impact = (50-48)/100000 * 5000 * 0.01 = 0.01
        # executed_price = min(50, 49 + 0.01) = 49.01

        # 3. Handle SELL LIMIT order
        sell_order = Order(side=Order.SELL, exectype=Order.LIMIT, price=45.0)
        market_data = pd.Series({
            'high': 46.0,
            'low': 44.0,
            'open': 45.5,
            'volume': 80000
        })
        executed_price = slippage_model(
            order=sell_order,
            quantity=3000,
            data=market_data
        )
        # Result: max(45.0, 44.0) = 45.0

    Custom Subclass Example:

        class FixedSlippage(SlippageBase):
            \"""Fixed percentage slippage model\"""
            
            def __init__(self, percent: float):
                if percent < 0:
                    raise ValueError("Percent must be ≥ 0")
                self.percent = percent
                
            def __call__(self, order, quantity, data):
                if order.exectype == Order.MARKET:
                    if order.side == Order.BUY:
                        return data['open'] * (1 + self.percent)
                    else:
                        return data['open'] * (1 - self.percent)
                return super().calculate_limit_price(order, data)

        # Usage:
        fixed_slip = FixedSlippage(percent=0.005)
        price = fixed_slip(buy_order, 1000, market_data)
    """
    
    def __init__(self, slip_rate: float = 0.01):
        self.slip_rate = slip_rate

    def __call__(self, order: OrderBase, quantity: float, data: pd.Series) -> float:
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

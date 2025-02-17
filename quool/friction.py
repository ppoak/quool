import pandas as pd
from .order import Order


class CommissionBase:

    def __init__(self):
        pass

    def __call__(self, order: Order, price: float, quantity: float) -> float:
        return 0

    def __str__(self):
        return f"CommissionBase(no commission)"

    def __repr__(self):
        return super().__repr__()


class SlippageBase:

    def __init__(self):
        pass

    def __call__(self, order: Order, quantity: float, data: pd.Series) -> float:
        return data["open"]

    def __str__(self):
        return f"SlippageBase(no slippage)"
    
    def __repr__(self):
        return super().__repr__()


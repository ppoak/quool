import numpy as np
import pandas as pd
from uuid import uuid4
from collections import deque, defaultdict


class Order:

    def __init__(self, code: str, side: str, time: str, size: float, price: float, otype: str = "Market"):
        self.code = code
        self.side = side
        self.created = time
        self.csize = size
        self.cprice = price
        self.otype = otype
        self.status = "Pending"
        self.executed = pd.to_datetime(np.nan)
        self.esize = np.nan
        self.eprice = np.nan
        self.rsize = self.csize
        self.comm = np.nan
        self.oid = uuid4()
    
    @property
    def alive(self):
        return self.status == "Pending" or self.status == "Partial"
    
    def __str__(self) -> str:
        if self.alive:
            return f"[{self.oid}] <{self.status}> [{self.side}] {self.csize} {self.code} @ {self.cprice}"
        return f"[{self.oid}] <{self.status}> [{self.side}] {self.esize} {self.code} @ {self.eprice}"
    
    def __repr__(self) -> str:
        return self.__str__()
    
class MarketBuyOrder(Order):

    def __init__(self, code: str, time: str, size: float, price: float):
        super().__init__(code, 'Buy', time, size, price, "Market")
    
    def execute(self, time: str, popen: float, phigh: float, plow: float, pclose: float, volume: float):
        if not self.alive:
            return self
        self.executed = pd.to_datetime(time)
        self.eprice = popen # slippage needs to be complemented
        self.esize = min(self.csize, volume)
        self.rsize = self.csize - self.esize
        self.comm = max(self.esize * 0.0002, 5) # commission should be computed more flexibly
        if not self.rsize:
            self.status = "Completed"
        else:
            self.status = "Partial"
        return self


class MarketSellOrder(Order):

    def __init__(self, code: str, time: str, size: float, price: float):
        super().__init__(code, 'Sell', time, -size, price, "Market")

    def execute(self, time: str, popen: float, phigh: float, plow: float, pclose: float, volume: float):
        if not self.alive:
            return self
        self.executed = pd.to_datetime(time)
        self.eprice = popen # slippage needs to be complemented
        self.esize = max(self.csize, -volume)
        self.rsize = self.esize - self.csize
        self.comm = -self.esize * 0.0002 # commission should be computed more flexibly
        if not self.rsize:
            self.status = "Completed"
        else:
            self.status = "Partial"
        return self


class Position:

    def __init__(self):
        self.size = 0
        self.price = 0
    
    def update(self, size: float, price: float):
        self.price = (self.size * self.price + size * price) / (self.size + size)
        self.size += size


class Broker:

    def __init__(self):
        self.pending = deque()
        self.completed = deque()
        self.position = defaultdict(Position)
    
    def marketbuy(self, code: str, time: str, size: float, price: float):
        order = MarketBuyOrder(code, time, size, price, "Market")
        self.pending.append(order)
        return order
    
    def marketsell(self, code: str, time: str, size: float, price: float):
        order = MarketSellOrder(code, time, size, price, "Market")
        self.pending.append(order)
        return order

    def execute(self, time: str, bars: pd.DataFrame):
        self.pending.append(None)
        while True:
            order = self.pending.popleft()
            if order is None:
                break
            order = order.execute(time, **bars.loc[order.code]) # bars should be open, high, low, close, volume
            if order.alive:
                self.pending.append(order)
            else:
                self.completed.append(order)
                self.position[order.code].update(order.esize, order.eprice)


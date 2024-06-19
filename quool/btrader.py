import numpy as np
import pandas as pd
from uuid import uuid4
from .tool import Logger
from collections import deque, defaultdict
from abc import ABC, abstractmethod


class Order(ABC):

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


class MarketSellOrder(Order):

    def __init__(self, code: str, time: str, size: float, price: float):
        super().__init__(code, 'Sell', time, -size, price, "Market")


class Exchange(ABC):

    def __init__(self):
        self.current = pd.to_datetime(np.nan)
        self.cdata = pd.DataFrame()

    def load(self, since: str = None):
        raise NotImplementedError
    
    def match_marketorder(self, market_order: Order):
        market_order.executed = pd.to_datetime(self.current)
        market_order.eprice = self.cdata.loc[market_order.code, "open"] # slippage needs to be complemented
        market_order.esize = min(market_order.csize, self.cdata.loc[market_order.code, "volume"])
        market_order.comm = max(market_order.esize * 0.0002, 5) # here, commission should be handled by broker
        market_order.status = "Completed" # when partial executed, more things need to be done
        return market_order
    
    def match(self, order: Order) -> Order:
        if not order.alive:
            return order
        if order.otype == "Market":
            return self.match_marketorder(order)
        else:
            raise TypeError(f"{order.otype} order is not supported")


class Position:

    def __init__(self):
        self.size = 0
        self.price = 0
    
    def update(self, size: float, price: float):
        self.price = (self.size * self.price + size * price) / (self.size + size)
        self.size += size


class Broker(ABC):
    
    def __init__(self):
        self._pending = deque()
        self._completed = deque()
        self._cash = 0
        self._freecash = 0
        self.position = defaultdict(Position)
        self.current = pd.to_datetime(np.nan)
        self.logger = Logger("Broker", display_name=False)
    
    @property
    def cash(self):
        return self._cash
    
    @property
    def freecash(self):
        return self._freecash

    @property
    def pending(self):
        return self._pending

    @property
    def completed(self):
        return self._completed

    def log(self, text: str, level: int = 10):
        self.logger.log(level=level, msg=text)
    

class BackBroker(Broker):

    def __init__(self):
        super().__init__()
        self.commfee = 0.0002
        self.curve = deque()
    
    def transfer(self, amount: float):
        self._cash += amount
        self._freecash += amount
    
    def marketbuy(self, code: str, time: str, size: float, price: float):
        order = MarketBuyOrder(code, time, size, price, "Market")
        cash = size * price
        if self.freecash < cash:
            order.status = "Rejected"
            self.completed.append(order)
            self.log(f"[{time}] {order}")
            return order
        else:
            self._freecash -= cash
            self.pending.append(order)
            self.log(f"[{time}] {order}")
            return order
    
    def marketsell(self, code: str, time: str, size: float, price: float):
        order = MarketSellOrder(code, time, size, price, "Market")
        if size <= self.position[code].size:
            self.pending.append(order)
            self.log(f"[{time}] {order}")
            return order
        else:
            order.status = "Reject"
            self.completed.append(order)
            self.log(f"[{time}] {order}")
            return order

    def submit(self, exchange: Exchange):
        self.pending.append(None)
        while True:
            order = self.pending.popleft()
            if order is None:
                break
            order = exchange.match(order)
            if order.alive:
                self.pending.append(order)
            else:
                order.comm = max(order.esize * self.commfee, 5)
                self.completed.append(order)
                self.position[order.code].update(order.esize, order.eprice)


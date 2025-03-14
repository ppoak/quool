import pandas as pd
from quool import Broker


class XtBroker(Broker):

    def __init__(self, account: str, path: str, id: str = None):
        from xtquant import xttrader, xttype, xtconstant

        self.BUY = xtconstant.STOCK_BUY
        self.SELL = xtconstant.STOCK_SELL
        self.MARKET = xtconstant.LATEST_PRICE
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
        return self.trader.query_stock_asset(self.account).cash

    @property
    def frozen(self):
        return self.trader.query_stock_asset(self.account).frozen_cash

    @property
    def value(self):
        return self.trader.query_stock_asset(self.account).total_asset

    @property
    def market(self):
        return self.trader.query_stock_asset(self.account).market_value

    @property
    def orders(self):
        return self.trader.query_stock_orders(self.account, cancelable_only=False)

    @property
    def pendings(self):
        return self.trader.query_stock_orders(self.account, cancelable_only=True)

    @property
    def positions(self):
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
        quantity = quantity // 100 * 100
        if quantity > 0:
            return self.trader.order_stock(
                self.account, code, type, quantity, exectype, price, order_remark=remark
            )

    def buy(self, code: str, quantity: float, price: float = -1, remark: str = ""):
        return self.create(self.BUY, code, quantity, self.MARKET, price, remark)

    def sell(self, code: str, quantity: float, price: float = -1, remark: str = ""):
        return self.create(self.SELL, code, quantity, self.MARKET, price, remark)

    def close(self, code: str, price: float = -1, remark: str = ""):
        for pos in self.positions:
            if pos.stock_code == code:
                quantity = pos.can_use_volume
                if quantity > 0:
                    return self.create(self.SELL, code, quantity, self.MARKET, price, remark)

    def limit_buy(self, code: str, quantity: float, price: float, remark: str = ""):
        return self.create(self.BUY, code, quantity, self.LIMIT, price, remark)

    def limit_sell(self, code: str, quantity: float, price: float, remark: str = ""):
        return self.create(self.SELL, code, quantity, self.LIMIT, price, remark)
    
    def limit_close(self, code: str, price: float, remark: str = ""):
        for pos in self.positions:
            if pos.stock_code == code:
                quantity = pos.can_use_volume
                if quantity > 0:
                    return self.create(self.SELL, code, quantity, self.LIMIT, price, remark)

    def update(self, time: str, data: pd.DataFrame):
        pass

    def get_orders(self):
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

    def __str__(self):
        return (
            f"XtBroker(#{self.account.account_id}\n"
            f"  ${self.value}=${self.market}(market)+${self.balance}(balance)+${self.frozen}(frozen))\n"
            f"  {dict([(position.stock_code, position.volume) for position in self.positions])}\n"
            f")"
        )

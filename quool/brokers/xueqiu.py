import pandas as pd
from .util import XueQiu
from quool import Delivery, Order, Broker, FixedRateCommission, FixedRateSlippage


class XueQiuBroker(XueQiu, Broker):

    SUCCESS_CODE = "60000"
    FAILURE_CODE = "70000"
    NETWORK_CODE = "80000"

    BASE_URL_TRADE = "https://tc.xueqiu.com/tc/snowx/MONI/"
    BASE_URL_MARKET = "https://stock.xueqiu.com/v5/stock/screener/quote/list.json"
    BASE_URL_HOT = "https://stock.xueqiu.com/v5/stock/hot_stock/list.json"
    QUOTE_URL = "https://stock.xueqiu.com/v5/stock/batch/quote.json"

    def __init__(
        self,
        token: str,
        name: str,
        reconstruct: bool = False,
        commission: FixedRateCommission = None,
        slippage: FixedRateSlippage = None,
    ):
        Broker.__init__(self, name, commission, slippage)
        XueQiu.__init__(self, token)

        groups = self._list_investment_groups()
        if groups["result_code"] != self.SUCCESS_CODE:
            raise Exception("Failed to list investment groups")
        groups = groups["result_data"]["trans_groups"]
        existing_gids = [group["gid"] for group in groups]
        existing_names = [group["name"] for group in groups]

        if name not in existing_names:
            self._gid = self._add_investment_group(name)["result_data"]["gid"]
        elif not reconstruct:
            self._gid = existing_gids[existing_names.index(name)]
            self._balance = self.get_balance()
            self._positions = {
                code: info["shares"] for code, info in self.get_xueqiu_positions().items()
            }
        else:
            self._gid = existing_gids[existing_names.index(name)]
            self._delete_investment_group(self._gid)
            self._gid = self._add_investment_group(name)["result_data"]["gid"]

    def transfer(self, time: pd.Timestamp, amount: float):
        self._bank_transfer(
            self._gid, 1, str(pd.to_datetime(time).date()), "CHA", amount
        )
        Broker.transfer(self, time=time, amount=amount)

    def get_positions(self):
        return Broker.get_positions(self)

    def get_xueqiu_positions(self):
        portfolio = self._get_portfolio_performance(self._gid)
        if portfolio["result_code"] == self.SUCCESS_CODE:
            markets = portfolio["result_data"]["performances"]
            positions = {
                position.pop("symbol"): position
                for market in markets
                for position in market["list"]
            }
        else:
            raise ValueError(f"Failed to get positions: {portfolio['result_msg']}")
        return positions

    def get_balance(self):
        portfolio = self._get_portfolio_performance(self._gid)
        if portfolio["result_code"] == self.SUCCESS_CODE:
            markets = portfolio["result_data"]["performances"]
            cash = markets[0]["cash"]
        else:
            raise ValueError(f"Failed to get cash: {portfolio['result_msg']}")
        return cash

    def get_all_records(self, row: int = 50):
        transaction_url = (
            f"{self.BASE_URL_TRADE}transaction/list.json?row={row}&gid={self._gid}"
        )
        bank_transfer_url = (
            f"{self.BASE_URL_TRADE}bank_transfer/query.json?row={row}&gid={self._gid}"
        )

        transaction_data = self._get_request(transaction_url)
        bank_transfer_data = self._get_request(bank_transfer_url)

        return {
            "transaction_records": transaction_data,
            "bank_transfer_records": bank_transfer_data,
        }

    def _execute(self, order: Order, price: float, quantity: int):
        amount = price * quantity
        commission = self.commission(order, price, quantity)
        if order.type == order.BUY:
            cost = amount + commission
            if cost > self._balance:
                order.status = order.REJECTED
            else:
                delivery = Delivery(
                    time=self.time,
                    code=order.code,
                    type=order.type,
                    quantity=quantity,
                    price=price,
                    amount=cost,
                )
                result = self._execute_transaction(
                    symbol=order.code,
                    shares=quantity,
                    transaction_type=1,
                    date=self.time,
                    price=price,
                    tax_rate=0,
                    commission_rate=0,
                )
                if result["result_code"] == self.SUCCESS_CODE:
                    order += delivery
                    self.deliver(delivery)
                    self._balance -= cost
                    self._positions[order.code] = (
                        self._positions.get(order.code, 0) + quantity
                    )
                else:
                    order.status = order.REJECTED
                    order.time = self.time
        elif order.type == order.SELL:
            revenue = amount - commission
            if self._positions.get(order.code, 0) < quantity:
                order.status = order.REJECTED
            else:
                delivery = Delivery(
                    time=self.time,
                    code=order.code,
                    type=order.type,
                    quantity=quantity,
                    price=price,
                    amount=revenue,
                )
                result = self._execute_transaction(
                    symbol=order.code,
                    shares=quantity,
                    transaction_type=1,
                    date=self.time,
                    price=price,
                    tax_rate=0,
                    commission_rate=0,
                )
                if result["result_code"] == self.SUCCESS_CODE:
                    order += delivery
                    self.deliver(delivery)
                    self._balance += revenue
                    self._positions[order.code] -= quantity
                    if self._positions[order.code] == 0:
                        del self._positions[order.code]
                else:
                    order.status = order.REJECTED
                    order.time = self.time

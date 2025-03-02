from quool import proxy_request


class XueQiu:

    SUCCESS_CODE = "60000"
    FAILURE_CODE = "70000"
    NETWORK_CODE = "80000"

    BASE_URL_TRADE = "https://tc.xueqiu.com/tc/snowx/MONI/"
    BASE_URL_MARKET = "https://stock.xueqiu.com/v5/stock/screener/quote/list.json"
    BASE_URL_HOT = "https://stock.xueqiu.com/v5/stock/hot_stock/list.json"
    BASE_URL_QUOTE = "https://stock.xueqiu.com/v5/stock/batch/quote.json"

    def __init__(self, token: str):
        self.cookie = f"xq_a_token={token}"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.5359.125 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Cookie": self.cookie,
        }

    def _get_request(self, url: str, params: dict = None):
        response = proxy_request(url, headers=self.headers, params=params)
        if response.status_code == 200:
            return response.json()
        return {
            "result_data": None,
            "msg": f"GET request failed with status code {response.status_code}",
            "result_code": self.NETWORK_CODE,
        }

    def _post_request(self, url: str, data: dict):
        response = proxy_request(url, method="POST", headers=self.headers, data=data)
        if response.status_code == 200:
            return response.json()
        return {
            "result_data": None,
            "msg": f"GET request failed with status code {response.status_code}",
            "result_code": self.NETWORK_CODE,
        }

    def get_hot_stocks(self, time_period: str = "24h", size: int = 100):
        if time_period == "1h":
            _type, type_value = 10, 10
        elif time_period == "24h":
            _type, type_value = 10, 20
        else:
            raise ValueError("Invalid time period. Please use '1h' or '24h'.")

        params = {"size": size, "_type": _type, "type": type_value, "_": 1731392783029}

        response = self._get_request(self.BASE_URL_HOT, params)
        return response.json()

    def get_market_price(self, symbol: str):
        symbol = ",".join([sym for sym in symbol.rstrip(",").split(",")])
        url = f"{self.BASE_URL_QUOTE}?symbol={symbol}&extend=detail"
        response = self._get_request(url)
        if not response.get("error_code", 1):
            items = response.get("data", {}).get("items", [])
            if items:
                return items
        return {
            "result_code": self.FAILURE_CODE,
            "result_data": None,
            "msg": "No market data returned",
        }

    def _add_investment_group(self, name: str):
        url = f"{self.BASE_URL_TRADE}trans_group/add.json"
        data = {"name": name}
        return self._post_request(url, data)

    def _list_investment_groups(self):
        url = f"{self.BASE_URL_TRADE}trans_group/list.json"
        return self._get_request(url)

    def _delete_investment_group(self, gid: str):
        url = f"{self.BASE_URL_TRADE}trans_group/delete.json"
        return self._post_request(url, data={"gid": gid})

    def _bank_transfer(
        self, gid: str, transfer_type: int, date: str, market: str, amount: float
    ):
        url = f"{self.BASE_URL_TRADE}bank_transfer/add.json"
        data = {
            "gid": gid,
            "type": transfer_type,
            "date": date,
            "market": market,
            "amount": amount,
        }
        return self._post_request(url, data)

    def _get_portfolio_performance(self, gid: str):
        url = f"{self.BASE_URL_TRADE}performances.json?gid={gid}"
        return self._get_request(url)

    def _execute_transaction(
        self,
        symbol: str,
        shares: int,
        transaction_type: int,
        date: str,
        price: float = None,
        tax_rate: float = 0.1,
        commission_rate: float = 0.15,
        comment: str = "",
    ):
        if price is None:
            price = self.get_market_price(symbol)[0].get("quote", {}).get("current")
            if isinstance(price, dict) and "error" in price:
                return price

        url = f"{self.BASE_URL_TRADE}transaction/add.json"
        data = {
            "gid": self._gid,
            "symbol": symbol,
            "shares": shares,
            "type": transaction_type,
            "date": date,
            "comment": comment,
            "price": price,
            "tax_rate": tax_rate,
            "commission_rate": commission_rate,
        }
        return self._post_request(url, data)

    def get_positions(self, gid: str):
        portfolio = self._get_portfolio_performance(gid)
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

    def get_balance(self, gid: str):
        portfolio = self._get_portfolio_performance(gid)
        if portfolio["result_code"] == self.SUCCESS_CODE:
            markets = portfolio["result_data"]["performances"]
            cash = markets[0]["cash"]
        else:
            raise ValueError(f"Failed to get cash: {portfolio['result_msg']}")
        return cash

    def get_all_records(self, gid: str, row: int = 50):
        transaction_url = (
            f"{self.BASE_URL_TRADE}transaction/list.json?row={row}&gid={gid}"
        )
        bank_transfer_url = (
            f"{self.BASE_URL_TRADE}bank_transfer/query.json?row={row}&gid={gid}"
        )

        transaction_data = self._get_request(transaction_url)
        bank_transfer_data = self._get_request(bank_transfer_url)

        return {
            "transaction_records": transaction_data,
            "bank_transfer_records": bank_transfer_data,
        }

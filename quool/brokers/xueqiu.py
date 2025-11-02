import pandas as pd
from parquool import proxy_request
from quool import Delivery, Order, Broker, FixedRateCommission, FixedRateSlippage


class XueQiu:
    """Client for XueQiu (Snowball) REST APIs covering hot stocks, quotes, and paper trading.

    This class wraps HTTP GET/POST requests against XueQiu endpoints, using a token
    cookie for authentication. It provides helpers to fetch market data (hot stocks,
    batch quotes), and to manage paper-trading portfolios (groups, transfers, transactions).

    Attributes:
      SUCCESS_CODE (str): API code indicating success ('60000').
      FAILURE_CODE (str): API code indicating failure ('70000').
      NETWORK_CODE (str): Code used locally to indicate network-level failure ('80000').
      BASE_URL_TRADE (str): Base URL for trading endpoints.
      BASE_URL_MARKET (str): Base URL for market screener endpoint.
      BASE_URL_HOT (str): Base URL for hot stock endpoint.
      BASE_URL_QUOTE (str): Base URL for batch quote endpoint.
      cookie (str): Authentication cookie built from the XueQiu token.
      headers (dict): Standard HTTP headers used in requests, including User-Agent and Cookie.
    """

    SUCCESS_CODE = "60000"
    FAILURE_CODE = "70000"
    NETWORK_CODE = "80000"

    BASE_URL_TRADE = "https://tc.xueqiu.com/tc/snowx/MONI/"
    BASE_URL_MARKET = "https://stock.xueqiu.com/v5/stock/screener/quote/list.json"
    BASE_URL_HOT = "https://stock.xueqiu.com/v5/stock/hot_stock/list.json"
    BASE_URL_QUOTE = "https://stock.xueqiu.com/v5/stock/batch/quote.json"

    def __init__(self, token: str):
        """Initialize the XueQiu client with an authentication token.

        Builds the Cookie header 'xq_a_token' used by XueQiu for authentication.

        Args:
          token (str): XueQiu API token string.

        Returns:
          None
        """
        self.cookie = f"xq_a_token={token}"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.5359.125 Safari/537.36",
            "Accept": "application/json, text/plain, */*",
            "Cookie": self.cookie,
        }

    def _get_request(self, url: str, params: dict = None):
        """Perform an HTTP GET request via proxy_request.

        Wraps proxy_request and normalizes the response:
          - On HTTP 200, returns response.json().
          - Otherwise, returns a dict with result_code=NETWORK_CODE, msg, and result_data=None.

        Args:
          url (str): Target URL.
          params (dict, optional): Query string parameters. Defaults to None.

        Returns:
          dict: Parsed JSON response or an error dict with keys:
            - result_code (str)
            - msg (str)
            - result_data (Any)

        Raises:
          Exception: Any exceptions raised by proxy_request may propagate.
        """
        response = proxy_request(url, headers=self.headers, params=params)
        if response.status_code == 200:
            return response.json()
        return {
            "result_data": None,
            "msg": f"GET request failed with status code {response.status_code}",
            "result_code": self.NETWORK_CODE,
        }

    def _post_request(self, url: str, data: dict):
        """Perform an HTTP POST request via proxy_request.

        Wraps proxy_request and normalizes the response:
          - On HTTP 200, returns response.json().
          - Otherwise, returns a dict with result_code=NETWORK_CODE, msg, and result_data=None.

        Args:
          url (str): Target URL.
          data (dict): JSON or form payload to POST.

        Returns:
          dict: Parsed JSON response or an error dict similar to _get_request().

        Raises:
          Exception: Any exceptions raised by proxy_request may propagate.
        """
        response = proxy_request(url, method="POST", headers=self.headers, data=data)
        if response.status_code == 200:
            return response.json()
        return {
            "result_data": None,
            "msg": f"GET request failed with status code {response.status_code}",
            "result_code": self.NETWORK_CODE,
        }

    def get_hot_stocks(self, time_period: str = "24h", size: int = 100):
        """Fetch the list of hot stocks from XueQiu.

        Maps a human-readable time period to the underlying API parameters and
        requests the hot stock list.

        Args:
          time_period (str, optional): Time window ('1h' or '24h'). Defaults to '24h'.
          size (int, optional): Number of stocks to fetch. Defaults to 100.

        Returns:
          dict: Parsed JSON response from the API.

        Raises:
          ValueError: If time_period is not '1h' or '24h'.
        """
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
        """Retrieve batch quotes (including current price) for one or more symbols.

        The input is a comma-separated symbol string, which is sanitized and passed
        to the batch quote endpoint.

        Args:
          symbol (str): Symbols joined by comma (e.g., 'SH600519,SZ000001').

        Returns:
          list[dict] or dict: On success, returns a list of item dicts under 'data.items';
          on failure, returns a dict with keys:
            - result_code (str): FAILURE_CODE or error code
            - result_data (None)
            - msg (str): Error message
        """
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
        """Create a new paper-trading investment group.

        Args:
          name (str): Name of the investment group.

        Returns:
          dict: Parsed JSON response containing group info (e.g., 'result_data.gid').

        Raises:
          Exception: HTTP or network-level errors may propagate.
        """
        url = f"{self.BASE_URL_TRADE}trans_group/add.json"
        data = {"name": name}
        return self._post_request(url, data)

    def _list_investment_groups(self):
        """List all paper-trading investment groups for the authenticated user.

        Returns:
          dict: Parsed JSON response containing 'result_data.trans_groups'.

        Raises:
          Exception: HTTP or network-level errors may propagate.
        """
        url = f"{self.BASE_URL_TRADE}trans_group/list.json"
        return self._get_request(url)

    def _delete_investment_group(self, gid: str):
        """Delete an existing paper-trading investment group.

        Args:
          gid (str): Group identifier.

        Returns:
          dict: Parsed JSON response indicating success or failure.
        """
        url = f"{self.BASE_URL_TRADE}trans_group/delete.json"
        return self._post_request(url, data={"gid": gid})

    def _bank_transfer(
        self, gid: str, transfer_type: int, date: str, market: str, amount: float
    ):
        """Record a cash transfer (deposit/withdrawal) for a paper-trading group.

        Args:
          gid (str): Group identifier.
          transfer_type (int): Transfer type (e.g., 1 for deposit; refer to XueQiu docs).
          date (str): Date string (YYYY-MM-DD).
          market (str): Market code (e.g., 'CHA' for A-shares).
          amount (float): Transfer amount (positive for deposit, negative for withdrawal).

        Returns:
          dict: Parsed JSON response indicating success or failure.
        """
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
        """Fetch performance and holdings for a paper-trading group.

        Args:
          gid (str): Group identifier.

        Returns:
          dict: Parsed JSON response. On success, includes 'result_data.performances'
            with positions and cash.

        Raises:
          Exception: HTTP or network errors may propagate.
        """
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
        """Execute a paper-trading transaction (buy/sell) in XueQiu.

        If price is not provided, attempts to fetch the current price using get_market_price().

        Args:
          symbol (str): Instrument symbol recognized by XueQiu (e.g., 'SH600519').
          shares (int): Number of shares to trade.
          transaction_type (int): Transaction type (e.g., 1=buy, 2=sell; refer to XueQiu docs).
          date (str): Trade date string (YYYY-MM-DD).
          price (float, optional): Execution price. If None, uses current quote. Defaults to None.
          tax_rate (float, optional): Tax rate applied to transaction. Defaults to 0.1.
          commission_rate (float, optional): Commission rate applied. Defaults to 0.15.
          comment (str, optional): Optional trade note. Defaults to "".

        Returns:
          dict: Parsed JSON response indicating success or failure.

        Raises:
          ValueError: If price cannot be determined from market data.
        """
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
        """Return positions for the specified investment group.

        On success, aggregates positions across markets and returns a mapping from
        symbol to its position details.

        Args:
          gid (str): Group identifier.

        Returns:
          dict: Mapping {symbol: position_info_dict}.

        Raises:
          ValueError: If the API returns a non-success result_code or missing data.
        """
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
        """Return cash balance for the specified investment group.

        Args:
          gid (str): Group identifier.

        Returns:
          float: Cash amount.

        Raises:
          ValueError: If the API returns a non-success result_code or missing data.
        """
        portfolio = self._get_portfolio_performance(gid)
        if portfolio["result_code"] == self.SUCCESS_CODE:
            markets = portfolio["result_data"]["performances"]
            cash = markets[0]["cash"]
        else:
            raise ValueError(f"Failed to get cash: {portfolio['result_msg']}")
        return cash

    def get_all_records(self, gid: str, row: int = 50):
        """Fetch recent transaction and bank transfer records for a group.

        Args:
          gid (str): Group identifier.
          row (int, optional): Number of rows to fetch for each endpoint. Defaults to 50.

        Returns:
          dict: A dictionary with keys:
            - transaction_records (dict): Parsed JSON from transaction/list.json.
            - bank_transfer_records (dict): Parsed JSON from bank_transfer/query.json.
        """
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


class XueQiuBroker(XueQiu, Broker):
    """Broker integration with XueQiu paper trading, extending local Broker behavior.

    Combines the local Broker accounting and order lifecycle with remote XueQiu paper
    trading operations. It can reconstruct or initialize a group, synchronize cash and
    positions from XueQiu, and execute orders by posting transactions to XueQiu.

    Attributes:
      SUCCESS_CODE (str): API success code ('60000').
      FAILURE_CODE (str): API failure code ('70000').
      NETWORK_CODE (str): Local network error code ('80000').
      BASE_URL_TRADE (str): Base URL for XueQiu paper-trading endpoints.
      BASE_URL_MARKET (str): Base URL for market screener endpoint.
      BASE_URL_HOT (str): Base URL for hot stock endpoint.
      QUOTE_URL (str): URL for batch quote queries.
      _gid (str): Group identifier used for paper trading.
    """

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
        """Initialize the XueQiuBroker with authentication and group configuration.

        Behavior:
          - Initializes the base Broker (id=name) and XueQiu client.
          - Lists existing investment groups.
          - If 'name' does not exist, creates a new group and sets _gid.
          - If 'name' exists and reconstruct=False, attaches to the existing group,
            synchronizing cash and positions from XueQiu into local Broker state.
          - If reconstruct=True, deletes the existing group and recreates it with the same name.

        Args:
          token (str): XueQiu API token string.
          name (str): Paper-trading group name to use.
          reconstruct (bool, optional): Whether to delete and recreate the group if it exists.
            Defaults to False.
          commission (FixedRateCommission, optional): Commission model. Defaults to FixedRateCommission().
          slippage (FixedRateSlippage, optional): Slippage model. Defaults to FixedRateSlippage().

        Returns:
          None

        Raises:
          Exception: If listing investment groups fails.
          ValueError: If synchronization of cash or positions fails due to API errors.
        """
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
                code: info["shares"]
                for code, info in self.get_xueqiu_positions().items()
            }
        else:
            self._gid = existing_gids[existing_names.index(name)]
            self._delete_investment_group(self._gid)
            self._gid = self._add_investment_group(name)["result_data"]["gid"]

    def transfer(self, time: pd.Timestamp, amount: float):
        """Record a cash transfer both remotely (XueQiu) and locally (Broker).

        Posts a bank transfer to XueQiu and then applies a local Delivery to the
        Broker state.

        Args:
          time (pandas.Timestamp): Transfer timestamp.
          amount (float): Transfer amount (positive deposit, negative withdrawal).

        Returns:
          None

        Raises:
          ValueError: If 'time' cannot be parsed into a timestamp.
          Exception: If the remote bank transfer post fails.
        """
        self._bank_transfer(
            self._gid, 1, str(pd.to_datetime(time).date()), "CHA", amount
        )
        Broker.transfer(self, time=time, amount=amount)

    def get_positions(self):
        """Return local Broker positions.

        Delegates to Broker.get_positions().

        Returns:
          pandas.Series: Series of local positions indexed by instrument code.
        """
        return Broker.get_positions(self)

    def get_xueqiu_positions(self):
        """Fetch positions from XueQiu for the current group.

        Returns:
          dict: Mapping {symbol: position_info_dict}.

        Raises:
          ValueError: If the API returns a non-success result_code or missing data.
        """
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
        """Fetch cash balance from XueQiu for the current group.

        Returns:
          float: Cash amount.

        Raises:
          ValueError: If the API returns a non-success result_code or missing data.
        """
        portfolio = self._get_portfolio_performance(self._gid)
        if portfolio["result_code"] == self.SUCCESS_CODE:
            markets = portfolio["result_data"]["performances"]
            cash = markets[0]["cash"]
        else:
            raise ValueError(f"Failed to get cash: {portfolio['result_msg']}")
        return cash

    def get_all_records(self, row: int = 50):
        """Fetch recent transaction and bank transfer records for the current group.

        Args:
          row (int, optional): Number of rows to fetch. Defaults to 50.

        Returns:
          dict: A dictionary with keys:
            - transaction_records (dict): Parsed JSON from transaction/list.json.
            - bank_transfer_records (dict): Parsed JSON from bank_transfer/query.json.
        """
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
        """Execute an order by posting a transaction to XueQiu and updating local state.

        For BUY:
          - Computes cost = price * quantity + commission.
          - If sufficient local cash, posts transaction to XueQiu.
          - On SUCCESS_CODE, creates a Delivery, updates order and Broker balance/positions.
          - Otherwise, marks order REJECTED and sets order.time.

        For SELL:
          - Validates sufficient local position.
          - Computes revenue = price * quantity - commission.
          - Posts transaction to XueQiu.
          - On SUCCESS_CODE, creates a Delivery, updates order and Broker balance/positions.
          - Otherwise, marks order REJECTED and sets order.time.

        Args:
          order (Order): Order to execute.
          price (float): Execution price per unit.
          quantity (int): Execution quantity.

        Returns:
          None

        Raises:
          Exception: If remote transaction posting fails unexpectedly.
        """
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

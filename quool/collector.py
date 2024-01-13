import re
import abc
import time
import base64
import random
import hashlib
import datetime
import requests
import numpy as np
import pandas as pd
import akshare as ak
from lxml import etree
from pathlib import Path
from tqdm.auto import tqdm
from bs4 import BeautifulSoup
from urllib.parse import quote
from joblib import Parallel, delayed
from .exception import RequestFailedError
from .tool import Logger, strip_stock_code, format_code


class RequestBase(abc.ABC):

    ua = [
        'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/30.0.1599.101',
        'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.122',
        'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.71',
        'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95',
        'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.71',
        'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; QQDownload 732; .NET4.0C; .NET4.0E)',
        'Mozilla/5.0 (Windows NT 5.1; U; en; rv:1.8.1) Gecko/20061208 Firefox/2.0.0 Opera 9.50',
        'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:34.0) Gecko/20100101 Firefox/34.0',
    ]
    basic_headers = {
        'Accept': '*/*',
        'Connection': 'keep-alive',
        'Accept-Language': 'zh-CN,zh;q=0.8'
    }

    def __init__(
        self,
        headers: dict = None,
        proxies: list[dict] = None,
        timeout: float = None,
        retry: int = 1,
        delay: float = 0,
        verbose: bool = False,
    ) -> None:
        self.headers = headers or {}
        if not (self.headers.get('user-agent') or self.headers.get('User-Agent')):
            self.headers['User-Agent'] = random.choice(self.ua)
        if headers:
            self.headers.update(headers)

        self.proxies = proxies or [{}]
        self.timeout = timeout
        self.retry = retry
        self.delay = delay
        self.verbose = verbose
    
    def __request(
        self, 
        url: str, 
        method: str = None,
        **kwargs
    ) -> requests.Response:
        method = method
        method = getattr(requests, method)

        for t in range(1, self.retry + 1):
            try:
                resp = method(
                    url, headers=self.headers, proxies=random.choice(self.proxies),
                    timeout=self.timeout, **kwargs
                )
                resp.raise_for_status()
                if self.verbose:
                    logger = Logger("QuoolRequest")
                    logger.info(f'[+] {url} try {t}')
                return resp
            except Exception as e:
                if self.verbose:
                    logger = Logger("QuoolRequest")
                    logger.warning(f'[-] {e} {url} try {t}')
                time.sleep(self.delay)

    def request(
        self, url: str | list, 
        method: str = 'get',
        n_jobs: int = 1,
        backend: str = 'threading',
        **kwargs
    ):
        self.urls = [url] if not isinstance(url, list) else url
        self.responses = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(self.__request)(url, method, **kwargs) for url in self.urls
        )
        return self
    
    def get(
        self, url: str | list,
        n_jobs: int = 1,
        backend: str = 'threading',
        **kwargs
    ):
        return self.request(url, 'get', n_jobs, backend, **kwargs)

    def post(
        self, url: str | list,
        n_jobs: int = 1,
        backend: str = 'threading',
        **kwargs
    ):
        return self.request(url, 'post', n_jobs, backend, **kwargs)
    
    def callback(self):
        return

    @property
    def json(self):
        return [res.json() if res is not None else None for res in self.responses]

    @property
    def etree(self):
        return [etree.HTML(res.text) if res is not None else None for res in self.responses]
    
    @property
    def html(self):
        return [res.text if res is not None else None for res in self.responses]
    
    @property
    def soup(self):
        return [BeautifulSoup(res.text, 'html.parser')  if res is not None else None for res in self.responses]
    
    @property
    def content(self):
        return [res.content if res is not None else None for res in self.responses]

    def request_callback(
        self, 
        url: str | list, 
        method: str = 'get', 
        n_jobs: int = 1,
        backend: str = 'threading',
        *args, 
        **kwargs
    ):
        """
        Makes requests (parallel or sequential) when the instance is called.

        Args:
            para (bool): If True, makes parallel requests. Otherwise, sequential.

        Returns:
            Any: The result from the callback method.
        """
        return self.request(url, method=method, 
            n_jobs=n_jobs, backend=backend).callback(*args, **kwargs)

    def __str__(self):
        return (
            f"{self.__class__.__name__}\n"
            f"\ttimeout: {self.timeout}; delay: {self.delay}; "
            f"verbose: {self.verbose}; retry: {self.retry}\n"
            f"\tproxy: {self.proxies[:3]}\n"
            f"\theaders: {self.headers['User-Agent']}\n"
        )

    def __repr__(self):
        return self.__str__()


class KaiXin(RequestBase):

    __url_base = "http://www.kxdaili.com/dailiip/2/{i}.html"

    def __init__(self):
        super().__init__()
    
    def request(
        self, 
        page_count: int = 10,
        n_jobs: int = 1, 
        backend: str = 'threading'
    ):
        url = [self.__url_base.format(i) for i in range(1, page_count + 1)]
        return super().request(url, 'get', n_jobs, backend)

    def callback(self):
        results = []
        etrees = self.etree
        for tree in etrees:
            if tree is None:
                continue
            for tr in tree.xpath("//table[@class='active']//tr")[1:]:
                ip = "".join(tr.xpath('./td[1]/text()')).strip()
                port = "".join(tr.xpath('./td[2]/text()')).strip()
                results.append({
                    "http": "http://" + "%s:%s" % (ip, port),
                    "https": "https://" + "%s:%s" % (ip, port)
                })
        return pd.DataFrame(results)


class KuaiDaili(RequestBase):

    __inha_base = 'https://www.kuaidaili.com/free/inha/{page_index}/'
    __intr_base = 'https://www.kuaidaili.com/free/intr/{page_index}/'

    def __init__(self):
        super().__init__(delay=4)
    
    def request(
        self,
        page_count: int = 20,
        n_jobs: int = 1,
        backend: str = 'threading',
    ):
        url = []
        for page_index in range(1, page_count + 1):
            for pattern in [self.__inha_base, self.__intr_base]:
                url.append(pattern.format(page_index))
        super().request(url, 'get', n_jobs, backend)

    def callback(self):
        results = []
        for tree in self.etree:
            if tree is None:
                continue
            proxy_list = tree.xpath('.//table//tr')
            for tr in proxy_list[1:]:
                results.append({
                    "http": "http://" + ':'.join(tr.xpath('./td/text()')[0:2]),
                    "https": "http://" + ':'.join(tr.xpath('./td/text()')[0:2])
                })
        return pd.DataFrame(results)


class Ip3366(RequestBase):

    __type1_base = 'http://www.ip3366.net/free/?stype=1&page={page}' 
    __type2_base = "http://www.ip3366.net/free/?stype=2&page={page}"

    def __init__(self):
        super().__init__()

    def request(
        self,
        page_count: int = 3,
        n_jobs: int = 1,
        backend: str = 'threading',
    ):
        url = []
        for page in range(1, page_count + 1):
            for pattern in [self.__type1_base, self.__type2_base]:
                url.append(pattern.format(page))
        super().request(url, 'get', n_jobs, backend)

    def callback(self):
        results = []
        for text in self.html:
            if text is None:
                continue
            proxies = re.findall(r'<td>(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})</td>[\s\S]*?<td>(\d+)</td>', text)
            for proxy in proxies:
                results.append({"http": "http://" + ":".join(proxy), "https": "http://" + ":".join(proxy)})
        return pd.DataFrame(results)


class Ip98(RequestBase):

    __base_url = "https://www.89ip.cn/index_{page}.html"

    def __init__(self):
        super().__init__()
    
    def request(
        self,
        page_count: int = 20,
        n_jobs: int = 1,
        backend: str = 'threading',
    ):
        url = []
        for page in range(1, page_count + 1):
            url.append(self.__base_url.format(page))
        super().request(url, 'get', n_jobs, backend)
    
    def callback(self):
        results = []
        for text in self.html:
            if text is None:
                continue
            proxies = re.findall(
                r'<td.*?>[\s\S]*?(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})[\s\S]*?</td>[\s\S]*?<td.*?>[\s\S]*?(\d+)[\s\S]*?</td>',
                text
            )
            for proxy in proxies:
                results.append({"http": "http://" + ":".join(proxy), "https": "http://" + ":".join(proxy)})
        return pd.DataFrame(results)


class WeiboSearch:
    '''A search crawler engine for weibo
    ====================================
    sample usage:
    >>> result = WeiboSearch.search("keyword")
    '''

    __base = "https://m.weibo.cn/api/container/getIndex?containerid=100103type%3D1%26q%3D{}&page_type=searchall&page={}"
    logger = Logger("QuoolWeiboSearch")

    @classmethod
    def _get_content(cls, url, headers):

        def _parse(mblog):
            blog = {
                "created_at": mblog["created_at"],
                "text": re.sub(r'<(.*?)>', '', mblog['text']),
                "id": mblog["id"],
                "link": f"https://m.weibo.cn/detail/{mblog['id']}",                    
                "source": mblog["source"],
                "username": mblog["user"]["screen_name"],
                "reposts_count": mblog["reposts_count"],
                "comments_count": mblog["comments_count"],
                "attitudes_count": mblog["attitudes_count"],
                "isLongText": mblog["isLongText"],
            }
            if blog["isLongText"]:
                headers = {
                    "Referer": f"https://m.weibo.cn/detail/{blog['id']}",
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.2 Safari/605.1.15"
                }
                resp = requests.get(f"https://m.weibo.cn/statuses/extend?id={blog['id']}", headers=headers).json()
                blog["full_text"] = resp["data"]["longTextContent"]
            return blog

        # First try to get resources
        res = requests.get(url, headers=headers).json()
        # if it is end
        if res.get("msg"):
            return False

        # if it contains cards
        cards = res["data"]["cards"]
        blogs = []
        for card in cards:
            # find 'mblog' tag and append to result blogs
            mblog = card.get("mblog")
            card_group = card.get("card_group")
            if card.get("mblog"):
                blog = _parse(mblog)
                blogs.append(blog)
            elif card_group:
                for cg in card_group:
                    mblog = cg.get("mblog")
                    if mblog:
                        blog = _parse(mblog)
                        blogs.append(blog)
        return blogs
    
    @classmethod
    def _get_full(cls, keyword: str):
        page = 1
        result = []
        headers = {
            "Referer": f"https://m.weibo.cn/search?containerid=100103type%3D1%26q%3D{quote(keyword, 'utf-8')}",
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
            }
        cls.logger.info(f"Start in keyword: {keyword}")
        while True:
            cls.logger.info(f"Getting {keyword}, currently at page: {page} ... ")
            url = cls.__base.format(keyword, page)
            blogs = cls._get_content(url, headers)
            if not blogs:
                break
            result.extend(blogs)
            page += 1
            time.sleep(random.randint(5, 8))
        cls.logger.info(f"Finished in keyword: {keyword}!")
        return result
    
    @classmethod
    def _get_assigned(cls, keyword: str, pages: int):
        result = []
        cls.logger.info(f"Start in keyword: {keyword}")
        headers = {
            "Referer": f"https://m.weibo.cn/search?containerid=100103type%3D1%26q%3D{quote(keyword, 'utf-8')}",
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
            }
        for page in tqdm(range(1, pages+1)):
            cls.logger.info(f"Getting {keyword}, currently at page: {page} ... ")
            url = cls.__base.format(keyword, page)
            blogs = cls._get_content(url, headers)
            result.extend(blogs)
            time.sleep(random.randint(5, 8))
        cls.logger.info(f"Finished in keyword: {keyword}!")
        return result          
    
    @classmethod
    def search(cls, keyword: str, pages: int = -1):
        """Search for the keyword
        --------------------------
        
        keyword: str, keyword
        pages: int, how many pages you want to get, default -1 to all pages
        """

        keyword = keyword.replace('#', '%23')
        if pages == -1:
            result = cls._get_full(keyword)
        else:
            result = cls._get_assigned(keyword, pages)
        result = pd.DataFrame(result)
        return result


class AkShare:
    """
    AkShare is a class designed to interface with the AkShare API, providing methods to fetch 
    a variety of financial data. It simplifies the process of accessing and retrieving data 
    related to stock markets, ETFs, and other financial instruments.

    Class Attributes:
        - TODAY: A pd.Timestamp object representing today's date.
        - START: A string representing the default start date for fetching historical data.
        - logger: A Logger object for logging messages.

    Class Methods:
        - market_daily: Retrieves daily market prices for a specific stock.
        - stock_quote: Fetches real-time quotes for stocks in the A-share market.
        - plate_quote: Obtains real-time quotes for industry plates.
        - etf_market_daily: Gets daily market prices for a specific ETF.
        - stock_fund_flow: Retrieves fund flow data for a specific stock.
        - stock_fund_rank: Fetches fund flow rankings for stocks.
        - plate_info: Provides information about stocks within a specific plate.
        - balance_sheet: Fetches balance sheet data for a given stock.
        - profit_sheet: Retrieves profit sheet data for a given stock.
        - cashflow_sheet: Obtains cash flow sheet data for a specified stock.
        - index_weight: Fetches index weight data for a given stock index.

    Usage Example:
    --------------
    # Fetching daily market data for a specific stock
    daily_data = AkShare.market_daily('600000', start='20200101', end='20201231')

    # Obtaining real-time quotes for stocks
    stock_data = AkShare.stock_quote()

    # Getting balance sheet data for a stock
    balance_data = AkShare.balance_sheet('600000')
    """
    TODAY = pd.to_datetime(datetime.datetime.today()).normalize()
    START = '20050101'
    logger = Logger("QuoolAkShare")
    
    @classmethod
    def market_daily(cls, code: str, start: str = None, end: str = None):
        """Get market daily prices for one specific stock
        
        code: str, the code of the stock
        start: str, start date in string format
        end: str, end date in string format
        """
        code = strip_stock_code(code)
        start = start or cls.START
        end = end or cls.TODAY.strftime('%Y%m%d')

        price = ak.stock_zh_a_hist(symbol=code, start_date=start, end_date=end, adjust='')
        if not price.empty:
            price = price.set_index('日期')
        else:
            return price
        adjprice = ak.stock_zh_a_hist(symbol=code, start_date=start, end_date=end, adjust='hfq')
        if not adjprice.empty:
            adjprice = adjprice.set_index('日期')
        else:
            return adjprice
        adjfactor = adjprice['收盘'] / price['收盘']
        adjfactor.name = 'adjfactor'
        price = pd.concat([price, adjfactor], axis=1)
        price = price.rename(columns = {
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "涨跌幅": "pctchange",
            "振幅": "vibration",
            "涨跌额": "change",
            "换手率": "turnover",
        }).astype('f')
        price.index = pd.to_datetime(price.index)
        price.index.name = 'datetime'

        return price

    @classmethod
    def stock_quote(cls, code_only: bool = False):
        """Get the realtime quote amoung the a stock share market

        code_only: bool, decide only return codes on the market
        """
        price = ak.stock_zh_a_spot_em()
        price = price.set_index('代码').drop('序号', axis=1)
        if code_only:
            return price.index.to_list()
        return price

    @classmethod
    def plate_quote(cls, name_only: bool = False):
        data = ak.stock_board_industry_name_em()
        data = data.set_index('板块名称')
        if name_only:
            return data.index.to_list()
        return data

    @classmethod
    def etf_market_daily(cls, code: str, start: str = None, end: str = None):
        code = strip_stock_code(code)
        start = start or cls.START
        end = end or cls.TODAY.strftime('%Y%m%d')
        price = ak.fund_etf_fund_info_em(code, start, end).set_index('净值日期')
        price.index = pd.to_datetime(price.index)
        return price
    
    @classmethod
    def stock_fund_flow(cls, code: str):
        code, market = code.split('.')
        if market.isdigit():
            code, market = market, code
        market = market.lower()
        funds = ak.stock_individual_fund_flow(stock=code, market=market)
        funds = funds.set_index('日期')
        funds.index = pd.MultiIndex.from_product([[code], 
            pd.to_datetime(funds.index)], names=['日期', '代码'])
        return funds
    
    @classmethod
    def stock_fund_rank(cls):
        datas = []
        for indi in ['今日', '3日', '5日', '10日']:
            datas.append(ak.stock_individual_fund_flow_rank(indicator=indi
                ).drop('序号', axis=1).set_index('代码').rename(columns={'最新价': f'{indi}最新价'}))
        datas = pd.concat(datas, axis=1)
        datas['简称'] = datas.iloc[:, 0]
        datas = datas.drop('名称', axis=1)
        datas = datas.replace('-', None).apply(pd.to_numeric, errors='ignore')
        datas.index = pd.MultiIndex.from_product([[cls.today], datas.index], names=['日期', '代码'])
        return datas
    
    @classmethod
    def plate_info(cls, plate: str):
        data = ak.stock_board_industry_cons_em(symbol=plate).set_index('代码')
        return data

    @classmethod
    def balance_sheet(cls, code):
        try:
            data = ak.stock_balance_sheet_by_report_em(symbol=code)
            if data.empty:
                return None
            data = data.drop([
                'SECURITY_CODE', 'SECURITY_NAME_ABBR', 'ORG_CODE', 'ORG_TYPE', 'REPORT_TYPE',
                'REPORT_DATE_NAME', 'SECURITY_TYPE_CODE', 'UPDATE_DATE', 'CURRENCY', 'LISTING_STATE'
            ], axis=1)
            data = data.replace({None: np.nan})
            data = data.astype('float32', errors='ignore')
            data[['REPORT_DATE', 'NOTICE_DATE']] = data[['REPORT_DATE', 'NOTICE_DATE']].astype('datetime64[ns]')
            data = data.set_index('REPORT_DATE')
            data = data.reindex(pd.date_range(data.index.min(), data.index.max(), freq='q'))
            data['SECUCODE'] = data['SECUCODE'][~data['SECUCODE'].isna()].iloc[0]
            data = data.set_index(['SECUCODE', 'NOTICE_DATE'], append=True)
            data.index.names = ['report_date', 'secucode', 'notice_date']
            data = data.rename(columns=lambda x: x.lower())
            return data
        except:
            cls.logger.warning(f'{code} get balance sheet failed!, please try again mannually')
            return None

    @classmethod
    def profit_sheet(cls, code):
        try:
            data = ak.stock_profit_sheet_by_report_em(symbol=code)
            if data.empty:
                return None
            data = data.drop([
                'SECURITY_CODE', 'SECURITY_NAME_ABBR', 'ORG_CODE', 'ORG_TYPE', 'REPORT_TYPE',
                'REPORT_DATE_NAME', 'SECURITY_TYPE_CODE', 'UPDATE_DATE', 'CURRENCY'
            ], axis=1)
            data = data.replace({None: np.nan})
            data = data.astype('float32', errors='ignore')
            data[['REPORT_DATE', 'NOTICE_DATE']] = data[['REPORT_DATE', 'NOTICE_DATE']].astype('datetime64[ns]')
            data = data.set_index('REPORT_DATE')
            data = data.reindex(pd.date_range(data.index.min(), data.index.max(), freq='q'))
            data['SECUCODE'] = data['SECUCODE'][~data['SECUCODE'].isna()].iloc[0]
            data = data.set_index(['SECUCODE', 'NOTICE_DATE'], append=True)
            data.index.names = ['report_date', 'secucode', 'notice_date']
            data = data.rename(columns=lambda x: x.lower())
            return data
        except:
            cls.logger.warning(f'{code} get balance sheet failed!, please try again mannually')
            return None

    @classmethod
    def cashflow_sheet(cls, code):
        try:
            data = ak.stock_cash_flow_sheet_by_report_em(symbol=code)
            if data.empty:
                return None
            data = data.drop([
                'SECURITY_CODE', 'SECURITY_NAME_ABBR', 'ORG_CODE', 'ORG_TYPE', 'REPORT_TYPE',
                'REPORT_DATE_NAME', 'SECURITY_TYPE_CODE', 'UPDATE_DATE', 'CURRENCY'
            ], axis=1)
            data = data.replace({None: np.nan})
            data = data.astype('float32', errors='ignore')
            data[['REPORT_DATE', 'NOTICE_DATE']] = data[['REPORT_DATE', 'NOTICE_DATE']].astype('datetime64[ns]')
            data = data.set_index('REPORT_DATE')
            data = data.reindex(pd.date_range(data.index.min(), data.index.max(), freq='q'))
            data['SECUCODE'] = data['SECUCODE'][~data['SECUCODE'].isna()].iloc[0]
            data = data.set_index(['SECUCODE', 'NOTICE_DATE'], append=True)
            data.index.names = ['report_date', 'secucode', 'notice_date']
            data = data.rename(columns=lambda x: x.lower())
            return data
        except:
            cls.logger.warning(f'{code} get balance sheet failed!, please try again mannually')
            return None

        
    @classmethod
    def index_weight(cls, code: str):
        data = ak.index_stock_cons_weight_csindex(code)
        return data


class EastMoney:
    """
    The 'Em' class is designed to interface with East Money (东方财富网) for fetching 
    financial data and analysis. It provides methods to access various types of 
    financial information such as stock market movements and expert analyses.

    Class Attributes:
        - headers: Standard headers used for HTTP requests to East Money.

    Class Methods:
        - look_updown: Fetches real-time rise and fall data for a specific stock.

    Usage Example:
    --------------
    # Fetching rise and fall data for a given stock code
    stock_movement = Em.look_updown('600000')

    Notes:
    ------
    This class primarily targets the Chinese stock market and is useful for investors 
    and analysts focusing on this market.
    """

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.6 Safari/605.1.15",
        "Referer": "http://guba.eastmoney.com/",
        "Host": "gubacdn.dfcfw.com"
    }

    @classmethod
    def look_updown(cls, code: str):
        today = datetime.datetime.today().date()
        code = format_code(code, '{market}{code}')
        url = f"http://gubacdn.dfcfw.com/LookUpAndDown/{code}.js"
        res = requests.get(url, headers=cls.headers)
        res.raise_for_status()
        res = eval(res.text.strip('var LookUpAndDown=').replace('null', f'"{today}"'))
        data = pd.Series(res['Data'])
        data['code'] = code
        return data


class StockUS:
    """
    The 'StockUS' class is tailored for interacting with the stock.us market API. 
    It provides functionalities to fetch stock prices, index prices, and research 
    reports from the US market.

    Class Attributes:
        - __root: The root URL for the stock.us API.
        - headers: Standard headers for API requests.
        - category: Dictionary mapping category IDs to their descriptions.

    Class Methods:
        - index_price: Fetches historical price data for a specified index.
        - cn_price: Retrieves historical price data for a specific Chinese stock.
        - report_list: Lists research reports based on various criteria.
        - report_search: Searches for research reports based on keywords or other filters.

    Usage Example:
    --------------
    # Fetching historical price data for a US index
    index_data = StockUS.index_price('NASDAQ')

    # Searching for research reports in the US stock market
    reports = StockUS.report_search(keyword='technology', period='1m')

    Notes:
    ------
    This class is particularly useful for users interested in the stock.us api, 
    providing easy access to a wide range of financial data.
    """

    __root = "https://api.stock.us/api/v1/"
    headers = {
        "Host": "api.stock.us",
        "Origin": "https://stock.us",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.6 Safari/605.1.15",
        "Accept-Language": "zh-CN",
    }
    category = {
        1: "宏观经济",
        2: "投资策略",
        3: "行业研究",
        4: "晨会早报",
        8: "金工量化",
        9: "债券研究",
        10: "期货研究",
    }
    todaystr = datetime.datetime.today().strftime(r'%Y%m%d')
            
    @classmethod
    def index_price(
        cls, 
        index: str, 
        start: str = None, 
        end: str = None,
    ):
        start = start or '19900101'
        end = end or cls.todaystr
        url = cls.__root + f"index-price?security_code={index}&start={start}&stop={end}"
        res = requests.get(url, headers=cls.headers).json()
        price = pd.DataFrame(res['price'])
        price['date'] = price['date'].astype('datetime64[ns]')
        price = price.set_index('date')
        return price
    
    @classmethod
    def cn_price(
        cls, 
        code: str, 
        start: str = None,
        end: str = None,
    ):
        start = start or '19900101'
        end = end or cls.todaystr
        url = cls.__root + f"cn-price?security_code={code}&start={start}&stop={end}"
        res = requests.get(url, headers=cls.headers).json()
        price = pd.DataFrame(res['price'])
        price['date'] = price['date'].astype('datetime64[ns]')
        price = price.set_index('date')
        return price
    
    @classmethod
    def report_list(
        cls, 
        category: str = 8,
        sub_category: str = 0,
        keyword: str = '', 
        period: str = 'all', 
        org_name: str = '', 
        author: str = '',
        xcf_years: str = '', 
        search_fields: str = 'title',
        page: int = 1, 
        page_size: int = 100
    ):
        '''Get report data in quant block
        ---------------------------------------
        category: str, category to the field, use StockUS.category to see possible choices
        keyword: str, key word to search, default empty string to list recent 100 entries
        period: str, report during this time period
        q: str, search keyword
        org_name: str, search by org_name
        author: str, search by author
        xcf_years: str, search by xcf_years
        search_fields: str, search in fields, support "title", "content", "content_fp"
        page: int, page number
        page_size: int, page size
        '''
        url = cls.__root + 'research/report-list'
        params = (f'?category={category}&dates={period}&q={keyword}&org_name={org_name}'
                  f'&author={author}&xcf_years={xcf_years}&search_fields={search_fields}'
                  f'&page={page}&page_size={page_size}')
        if category != 8:
            params += f'&sub_category={sub_category}'
        headers = {
            "Referer": "https://stock.us/cn/report/quant",
        }
        headers.update(cls.headers)
        url += params
        res = requests.get(url, headers=headers).json()
        data = pd.DataFrame(res['data'])
        data[['pub_date', 'pub_week']] = data[['pub_date', 'pub_week']].astype('datetime64[ns]')
        data.authors = data.authors.map(
            lambda x: ' '.join(list(map(lambda y: y['name'] + ('*' if y['prize'] else ''), x))))
        data = data.set_index('id')
        return data
    
    @classmethod
    def report_search(
        cls, 
        keyword: str = '', 
        period: str = '3m', 
        org_name: str = '', 
        author_name: str = '',
        xcf_years: str = '', 
        search_fields: str = 'title',
        page: int = 1, 
        page_size: int = 100
    ):
        '''Search report in stockus database
        ---------------------------------------
        keyword: str, key word to search, default empty string to list recent 100 entries
        period: str, report during this time period
        org_name: str, search by org_name
        author: str, search by author
        xcf_years: str, search by xcf_years
        search_fields: str, search in fields, support "title", "content", "content_fp"
        page: int, page number
        page_size: int, page size
        '''
        url = cls.__root + 'research/report-search'
        params = (f'?dates={period}&q={keyword}&org_name={org_name}&author_name={author_name}'
                  f'&xcf_years={xcf_years}&search_fields={search_fields}&page={page}'
                  f'&page_size={page_size}')
        url += params
        res = requests.get(url, headers=cls.headers).json()
        data = pd.DataFrame(res['data'])
        data['pub_date'] = data['pub_date'].astype('datetime64[ns]')
        data.authors = data.authors.map(
            lambda x: ' '.join(list(map(lambda y: y['name'] + ('*' if y['prize'] else ''), x)))
            if isinstance(x, list) else '')
        data = data.set_index('id')
        return data


class WeChat(RequestBase):

    __webhook_base = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={key}"
    __qrcode_base = "https://open.weixin.qq.com/connect/qrcode/{uuid}"
    __service_base = "https://open.weixin.qq.com/connect/qrconnect?appid={app_id}&scope=snsapi_login&redirect_uri={redirect_url}"
    __logincheck_base = "https://lp.open.weixin.qq.com/connect/l/qrconnect?uuid={uuid}&_={timestamp}"
    __upload_base = "https://qyapi.weixin.qq.com/cgi-bin/webhook/upload_media?key={key}&type={type}"

    def login(self, appid: str, redirect_url: str):
        service_url = self.__service_base.format(appid=appid, redirect_url=redirect_url)
        soup = super().request(service_url, 'get').soup[0]
        if soup is None:
            raise RequestFailedError("weixin third-party login")
        qrcode_img = soup.find('img', class_='qrcode lightBorder')
        uuid = qrcode_img['src'].split('/')[-1]

        qrcode_url = self.__qrcode_base.format(uuid=uuid)
        img = super().get(qrcode_url).content[0]
        if img is None:
            raise RequestFailedError("weixin third-party login")
        temp_qrcode_path = Path("login.png")
        with open(temp_qrcode_path, "wb") as qrcode_file:
            qrcode_file.write(img)

        def _login_check(_u):
            lp_response = super(WeChat, self).get(_u).responses[0]
            if lp_response is None:
                raise RequestFailedError("weixin third-party login")
            variables = lp_response.text.split(';')[:-1]
            wx_errorcode = variables[0].split('=')[-1]
            wx_code = variables[1].split('=')[-1][1:-1]
            return wx_errorcode, wx_code
        
        wx_errorcode = '408'
        while True:
            timestamp = time.time()
            if wx_errorcode == '405':
                temp_qrcode_path.unlink()
                return wx_code
            
            elif wx_errorcode == '408':
                url = self.__logincheck_base.format(uuid=uuid, timestamp=int(timestamp))
                wx_errorcode, wx_code = _login_check(url)
                
            elif wx_errorcode == '404':
                url = f"{self.__logincheck_base.format(uuid=uuid, timestamp=int(timestamp))}&last=404"
                wx_errorcode, wx_code = _login_check(url)
                
            else:
                raise RequestFailedError("weixin third-party login")
            
    def notify(
        self, 
        key: str, 
        content_or_path: str,
        message_type: str = "text",
        mentions: str | list = None
    ):
        """
        Sends a notification message to WeChat Work using the webhook.

        Parameters:
        - key (str): The webhook key for the WeChat Work API.
        - content_or_path (str): The message content (for text/markdown) or the path to the file (for file/voice).
        - message_type (str): Type of the message ('text', 'markdown', 'image', 'file', 'voice').
        - mentions (str | list): List of users or mobile numbers to mention in the message.

        Returns:
        - dict: The response from the WeChat API.

        Usage Example:
        --------------
        WeiXin.notify('your_webhook_key', 'Hello, WeChat!', 'text')
        """
        notify_url = self.__webhook_base.format(key=key)
        
        mention_mobiles = []
        if mentions is not None:
            if not isinstance(mentions, list):
                mentions = [mentions]
            for i, mention in enumerate(mentions):
                if mention.lstrip('@').isdigit():
                    mention_mobiles.append(mentions.pop(i))
        mentions = mentions or []
        mention_mobiles = mention_mobiles or []
        
        if message_type in ["file", "voice"]:
            upload_url = self.__upload_base.format(key=key, type=message_type)
            if not content_or_path:
                raise ValueError("path is required for file and voice")
            path = Path(content_or_path)
            with path.open('rb') as fp:
                file_info = {"media": (
                    path.name, fp.read(), 
                    "multipart/form-data", 
                    {'Content-Length': str(path.stat().st_size)}
                )}
                resp = super().post(upload_url, files=file_info).json[0]

            if resp is None or resp["errcode"] != 0:
                raise requests.RequestException(resp["errmsg"])
            media_id = resp["media_id"]
            message = {
                "msgtype": message_type,
                message_type: {
                    "media_id": media_id, 
                    "mentioned_list": mentions,
                    "mentioned_mobile_list": mention_mobiles,
                },
            }
            resp = super().post(notify_url, json=message).json[0]
            if resp is None:
                raise RequestFailedError("Failed to upload image")
            return resp

        elif message_type in ["image"]:
            path = Path(content_or_path)
            with path.open('rb') as fp:
                image_orig = fp.read()
                image_base64 = base64.b64encode(image_orig).decode('ascii')
                image_md5 = hashlib.md5(image_orig).hexdigest()
                
            message = {
                "msgtype": message_type,
                message_type: {
                    "base64": image_base64,
                    "md5": image_md5,
                    "mentioned_list": mentions,
                    "mentioned_mobile_list": mention_mobiles,
                }
            }
            resp = super().post(notify_url, json=message).json[0]
            if resp is None:
                raise RequestFailedError("Failed to upload image")
            return resp
    
        elif message_type in ["text", "markdown"]:
            message = {
                "msgtype": message_type,
                message_type: {
                    "content": content_or_path,
                    "mentioned_list": mentions,
                    "mentioned_mobile_list": mention_mobiles,
                }
            }
            resp = super().post(notify_url, json=message).json[0]
            if resp is None:
                raise RequestFailedError("Failed to upload image")
            return resp

        else:
            raise ValueError(f"Unsupported message type: {message_type}")


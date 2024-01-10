import re
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
from .tools import Logger, strip_stock_code, format_code


class Request:
    """
    A custom class for handling HTTP requests with advanced features like retries, delays, and parallel processing.

    Attributes:
        ua (list): A list of user-agent strings for simulating different browsers.
        basic_headers (dict): Basic headers for HTTP requests.

    Methods:
        __init__: Initializes the Request object with URL(s), HTTP method, headers, etc.
        _req: Internal method to make a single HTTP request.
        request: Makes HTTP requests sequentially.
        para_request: Makes HTTP requests in parallel.
        callback: Callback method for handling the response.
        json: Property to get the JSON response from the requests.
        etree: Property to parse the response as lxml etree.
        html: Property to get the HTML response.
        soup: Property to parse the response using BeautifulSoup.
        __call__: Makes requests (parallel or sequential) when the instance is called.
        __str__: String representation of the Request instance.
        __repr__: Representation method for debugging purposes.

    Example:
        req = Request(url="https://example.com", method="get")
        response = req(para=True)  # Makes parallel requests
    """

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
        url: str | list,
        method: str = 'get',
        headers: dict = None,
        proxies: list[dict] = None,
        timeout: float = None,
        retry: int = 1,
        delay: float = 0,
        verbose: bool = False,
        **kwargs
    ) -> None:
        self.url = url if isinstance(url, list) else [url]
        self.method = method

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
        self.kwargs = kwargs
        self.run = False
    
    def _req(
        self, 
        url: str, 
        method: str = None,
        proxies: dict | None = None, 
        headers: dict | None = None,
        timeout: float | None = None,
        retry: int | None = None,
        delay: float | None = None,
        verbose: bool | None = None,
    ) -> requests.Response:
        """
        Internal method to make a single HTTP request with retries and delays.

        Args:
            url (str): URL for the HTTP request.
            method, proxies, headers, timeout, retry, delay, verbose: 
            Additional parameters for the request.

        Returns:
            requests.Response: The response object from the request, or None if failed.
        """
        logger = Logger("QuoolRequest")
        retry = retry or self.retry
        headers = headers or self.headers
        proxies = proxies or self.proxies
        timeout = timeout or self.timeout
        delay = delay or self.delay
        verbose = verbose or self.verbose
        method = method or self.method
        method = getattr(requests, method)

        for t in range(1, retry + 1):
            try:
                resp = method(
                    url, headers=headers, proxies=random.choice(proxies),
                    timeout=timeout, **self.kwargs
                )
                resp.raise_for_status()
                if verbose:
                    logger.info(f'[+] {url} try {t}')
                return resp
            except Exception as e:
                if verbose:
                    logger.warning(f'[-] {e} {url} try {t}')
                time.sleep(delay)

        return None
    
    def request(self) -> list[requests.Response]:
        """
        Makes HTTP requests sequentially for each URL in the instance.

        Returns:
            list[requests.Response]: A list of response objects from the requests.
        """
        responses = []
        for url in tqdm(self.url):
            resp = self._req(url)
            responses.append(resp)
        self.responses = responses
        return self
    
    def para_request(self) -> list[requests.Response]:
        """
        Makes HTTP requests in parallel for each URL in the instance.

        Uses joblib's Parallel and delayed functions for parallel processing.

        Returns:
            list[requests.Response]: A list of response objects from the requests.
        """
        self.responses = Parallel(n_jobs=-1, backend='loky')(
            delayed(self._req)(url) for url in tqdm(self.url)
        )
        return self

    def callback(self, *args, **kwargs):
        """
        Callback method for handling the response.

        Can be overridden in subclasses to provide custom behavior.

        Returns:
            Any: Default implementation returns the list of responses.
        """
        return self.responses

    @property
    def json(self):
        return [res.json() if res is not None else None for res in self.responses]

    @property
    def etree(self):
        return [etree.HTML(res.text) if res is not None else None for res in self.responses]
    
    @property
    def html(self):
        return [res.text  if res is not None else None for res in self.responses]
    
    @property
    def soup(self):
        return [BeautifulSoup(res.text, 'html.parser')  if res is not None else None for res in self.responses]

    def __call__(self, para: bool = True, *args, **kwargs):
        """
        Makes requests (parallel or sequential) when the instance is called.

        Args:
            para (bool): If True, makes parallel requests. Otherwise, sequential.

        Returns:
            Any: The result from the callback method.
        """
        if para:
            self.para_request()
        else:
            self.request()
        self.run = True
        return self.callback(*args, **kwargs)

    def __str__(self):
        return (
            f"{self.__class__.__name__}\n"
            f"\turl: {self.url}\n"
            f"\ttimeout: {self.timeout}; delay: {self.delay}; "
            f"verbose: {self.verbose}; retry: {self.retry}; run: {self.run}\n"
            f"\tmethod: {self.method}\n"
            f"\tproxy: {self.proxies[:3]}\n"
            f"\theaders: {self.headers['User-Agent']}\n"
        )

    def __repr__(self):
        return self.__str__()


class KaiXin(Request):

    def __init__(self, page_count: int = 10):
        url = [f"http://www.kxdaili.com/dailiip/2/{i}.html" for i in range(1, page_count + 1)]
        super().__init__(url = url)
    
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


class KuaiDaili(Request):

    def __init__(self, page_count: int = 20):
        url_pattern = [
            'https://www.kuaidaili.com/free/inha/{}/',
            'https://www.kuaidaili.com/free/intr/{}/'
        ]
        url = []
        for page_index in range(1, page_count + 1):
            for pattern in url_pattern:
                url.append(pattern.format(page_index))
        super().__init__(url=url, delay=4)

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


class Ip3366(Request):

    def __init__(self, page_count: int = 3):
        url = []
        url_pattern = ['http://www.ip3366.net/free/?stype=1&page={}', "http://www.ip3366.net/free/?stype=2&page={}"]
        for page in range(1, page_count + 1):
            for pat in url_pattern:
                url.append(pat.format(page))
        super().__init__(url=url)

    def callback(self, *args, **kwargs):
        results = []
        for text in self.html:
            if text is None:
                continue
            proxies = re.findall(r'<td>(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})</td>[\s\S]*?<td>(\d+)</td>', text)
            for proxy in proxies:
                results.append({"http": "http://" + ":".join(proxy), "https": "http://" + ":".join(proxy)})
        return pd.DataFrame(results)


class Ip98(Request):

    def __init__(self, page_count: int = 20):
        super().__init__(url=[f"https://www.89ip.cn/index_{i}.html" for i in range(1, page_count + 1)])
    
    def callback(self, *args, **kwargs):
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


class Checker(Request):

    def __init__(self, proxies: list[dict], url: str = "http://httpbin.org/ip"):
        super().__init__(
            url = [url] * len(proxies),
            proxies = proxies,
            timeout = 2.0,
            retry = 1,
        )

    def request(self) -> list[requests.Response]:
        responses = []
        for proxy, url in tqdm(zip(self.proxies, self.url)):
            resp = self._req(url=url, proxies=[proxy])
            responses.append(resp)
        self.responses = responses
        return self
    
    def para_request(self) -> list[requests.Response]:
        self.responses = Parallel(n_jobs=-1, backend='threading')(
            delayed(self._req)(url, proxies=[proxy]) for url, proxy in tqdm(list(zip(self.url, self.proxies)))
        )
        return self
    
    def callback(self):
        results = []
        for i, res in enumerate(self.responses):
            if res is None:
                continue
            results.append(self.proxies[i])
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


class Em:

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


class WeiXin:
    """
    This class provides an interface for interacting with WeChat for functionalities like QR code-based login and sending notifications through WeChat Work.

    Usage Example:
    --------------
    # Login example
    wx = WeiXin()
    # or you can
    wx = WeiXin
    app_id = 'your_app_id'
    redirect_url = 'your_redirect_url'
    login_code = wx.login(app_id, redirect_url)
    
    # Notification example
    key = 'your_webhook_key'
    message = 'Hello, WeChat!'
    WeiXin.notify(key, message, message_type='text')

    Class Attributes:
    -----------------
    - webhook_base: The base URL for sending webhook notifications.
    - qrcode_base: The base URL for generating QR codes.
    - service_base: The base URL for the QR code login service.
    - logincheck_base: The base URL for checking QR code login status.
    - upload_base: The base URL for uploading media for notifications.

    """

    webhook_base = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={key}"
    qrcode_base = "https://open.weixin.qq.com/connect/qrcode/{uuid}"
    service_base = "https://open.weixin.qq.com/connect/qrconnect?appid={app_id}&scope=snsapi_login&redirect_uri={redirect_url}"
    logincheck_base = "https://lp.open.weixin.qq.com/connect/l/qrconnect?uuid={uuid}&_={timestamp}"
    upload_base = "https://qyapi.weixin.qq.com/cgi-bin/webhook/upload_media?key={key}&type={type}"

    @classmethod
    def login(cls, appid: str, redirect_url: str):
        """
        Initiates the login process by generating a QR code for WeChat login.
        
        Parameters:
        - appid (str): The app ID for WeChat login.
        - redirect_url (str): The URL to redirect after successful login.

        Returns:
        - str: A login code if the login is successful, otherwise None.

        Usage Example:
        --------------
        wx = WeiXin()
        login_code = wx.login('your_app_id', 'your_redirect_url')
        """
        logger = Logger("WeiXinLogin")
        headers = {"user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.5.2 Safari/605.1.15"}

        service_url = cls.service_base.format(appid=appid, redirect_url=redirect_url)
        serv_resp = requests.get(service_url, headers=headers)
        serv_resp.raise_for_status()
        soup = BeautifulSoup(serv_resp.text, 'html.parser')
        qrcode_img = soup.find('img', class_='qrcode lightBorder')
        uuid = qrcode_img['src'].split('/')[-1]

        qrcode_url = qrcode_url.format(uuid=uuid)
        qrcode_resp = requests.get(qrcode_url, headers=headers)
        temp_qrcode_path = Path("login.png")
        with open(temp_qrcode_path, "wb") as qrcode_file:
            qrcode_file.write(qrcode_resp.content)
        logger.info(f"Please scan the QRCode to Login >>> {temp_qrcode_path}")

        def _login_check(_u):
            lp_response = requests.get(_u, headers=headers)
            lp_response.raise_for_status()
            variables = lp_response.text.split(';')[:-1]
            wx_errorcode = variables[0].split('=')[-1]
            wx_code = variables[1].split('=')[-1][1:-1]
            return wx_errorcode, wx_code
        
        wx_errorcode = '408'
        while True:
            timestamp = time.time()
            if wx_errorcode == '405':
                logger.info("Login Success")
                temp_qrcode_path.unlink()
                return wx_code
            
            elif wx_errorcode == '408':
                url = cls.lp_url.format(uuid=uuid, timestamp=int(timestamp))
                wx_errorcode, wx_code = _login_check(url)
                
            elif wx_errorcode == '404':
                logger.info("Scan Success, Please confirm Login on mobile phone")
                url = f"{cls.lp_url.format(uuid=uuid, timestamp=int(timestamp))}&last=404"
                wx_errorcode, wx_code = _login_check(url)
                
            else:
                logger.critical("Unknown error, please try again")
                return
            
    @classmethod
    def notify(
        cls, 
        key: str, 
        content_or_path: str = "",
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
        notify_url = cls.webhook_base.format(key=key)
        
        mention_mobiles = []
        if mentions is not None:
            if not isinstance(mentions, list):
                mentions = [mentions]
            for i, mention in enumerate(mentions):
                if mention.isdigit():
                    mention_mobiles.append(mentions.pop(i))
        mentions = mentions or []
        mention_mobiles = mention_mobiles or []
        
        if message_type in ["file", "voice"]:
            upload_url = cls.upload_base.format(key=key, type=message_type)
            if not content_or_path:
                raise ValueError("path is required for file and voice")
            path = Path(content_or_path)
            with path.open('rb') as fp:
                file_info = {"media": (
                    path.name, fp.read(), 
                    "multipart/form-data", 
                    {'Content-Length': str(path.stat().st_size)}
                )}
                resp = requests.post(upload_url, files=file_info)

            resp.raise_for_status()
            resp = resp.json()
            if resp["errcode"] != 0:
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
            resp = requests.post(notify_url, json=message)
            return resp.json()

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
            resp = requests.post(notify_url, json=message)
            return resp.json()
    
        elif message_type in ["text", "markdown"]:
            message = {
                "msgtype": message_type,
                message_type: {
                    "content": content_or_path,
                    "mentioned_list": mentions,
                    "mentioned_mobile_list": mention_mobiles,
                }
            }
            resp = requests.post(notify_url, json=message)
            return resp.json()

        else:
            raise ValueError(f"Unsupported message type: {message_type}")


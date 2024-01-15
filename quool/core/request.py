import abc
import time
import random
import requests
from lxml import etree
from bs4 import BeautifulSoup
from joblib import Parallel, delayed
from .util import Logger


class Request(abc.ABC):

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

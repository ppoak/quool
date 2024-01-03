import time
import random
import requests
from lxml import etree
from tqdm.auto import tqdm
from bs4 import BeautifulSoup
from joblib import Parallel, delayed
from .equipment import Logger


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
        if not (self.headers.get('user-agent') and self.headers.get('User-Agent')):
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

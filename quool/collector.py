import time
import base64
import random
import hashlib
import requests
from lxml import etree
from pathlib import Path
from tqdm.auto import tqdm
from .equipment import Logger
from bs4 import BeautifulSoup
from joblib import Parallel, delayed


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
        
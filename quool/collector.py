import abc
import time
import base64
import random
import hashlib
import requests
from lxml import etree
from pathlib import Path
from bs4 import BeautifulSoup
from joblib import Parallel, delayed
from .exception import RequestFailedError


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

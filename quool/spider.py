import re
import time
import base64
import random
import hashlib
import requests
import numpy as np
import pandas as pd
from lxml import etree
from pathlib import Path
from retrying import retry
from .table import ItemTable
from bs4 import BeautifulSoup
from .util import parse_commastr
from joblib import Parallel, delayed


class ProxyManager(ItemTable):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124S Safari/537.36"}

    def __init__(
        self, 
        uri: str | Path, 
        create: bool = False,
    ):
        self._size = 1000000
        super().__init__(uri, create)
        self.forget()
    
    @property
    def spliter(self):
        return pd.Grouper(level=0, freq='D')

    @property
    def namer(self):
        return lambda x: x.index[0].strftime(f"%Y-%m-%d")

    @property
    def picked(self):
        return self._picked
    
    def forget(self):
        self._fraglen = pd.Series(
            np.ones(len(self.fragments)) * self._size,
            index=self.fragments
        )
        self._picked = pd.Series([
            set() for _ in self.fragments
        ], index=self.fragments)
    
    def pick(self, field: str | list = None):
        rand_frag = self._fraglen[self._fraglen.cumsum() / 
            self._fraglen.sum() > random.random()].index
        if rand_frag.size == 0:
            raise ValueError("no more proxies available")
        proxy = self._read_fragment(rand_frag)
        if field is not None:
            proxy = proxy[field]
        index = random.choice(list(set(range(proxy.shape[0])) - self._picked.loc[rand_frag]))
        self._picked.loc[rand_frag].add(index)
        self._fraglen.loc[rand_frag] = proxy.shape[0] - len(self._picked.loc[rand_frag])
        return proxy.to_dict('records')[index]

    def check(self, proxy: dict, timeout: int = 2):
        check_url = "http://httpbin.org/ip"
        try:
            pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
            ip = re.findall(pattern, proxy['http'])[0]
            ips = re.findall(pattern, proxy['https'])[0]
            resp = requests.get(check_url, headers=self.headers, proxies=proxy, timeout=timeout)
            resp = resp.json()
            if resp.get("origin") in [ip, ips]:
                return True
            return False
        except:
            return False

    def add_kxdaili(self, pages: int = 1):
        url_base = "http://www.kxdaili.com/dailiip/2/{i}.html"

        resps = []
        for i in range(1, pages + 1):
            resps.append(requests.get(url_base.format(i=i), headers=self.headers))
        
        results = []
        for resp in resps:
            tree = etree.HTML(resp.text)
            for tr in tree.xpath("//table[@class='active']//tr")[1:]:
                ip = "".join(tr.xpath('./td[1]/text()')).strip()
                port = "".join(tr.xpath('./td[2]/text()')).strip()
                proxy = {
                    "http": "http://" + "%s:%s" % (ip, port),
                    "https": "https://" + "%s:%s" % (ip, port)
                }
                if self.check(proxy):
                    results.append(pd.Series(proxy, name=pd.to_datetime('now')))
        
        if len(results):
            results = pd.concat(results, axis=1).T
            if self.fragments:
                self.update(results)
            else:
                self.add(results)
    
    def add_kuaidaili(self, pages: int = 1):
        inha_base = 'https://www.kuaidaili.com/free/inha/{i}/'
        intr_base = 'https://www.kuaidaili.com/free/intr/{i}/'

        urls = []
        for i in range(1, pages + 1):
            for pattern in [inha_base, intr_base]:
                urls.append(pattern.format(i=i))
            
        resps = []
        for url in urls:
            resps.append(requests.get(url, headers=self.headers))
            time.sleep(1)
        
        results = []
        for resp in resps:
            tree = etree.HTML(resp.text)
            proxy_list = tree.xpath('.//table//tr')
            for tr in proxy_list[1:]:
                proxy = {
                    "http": "http://" + ':'.join(tr.xpath('./td/text()')[0:2]),
                    "https": "http://" + ':'.join(tr.xpath('./td/text()')[0:2])
                }
                if self.check(proxy):
                    results.append(pd.Series(proxy, name=pd.to_datetime('now')))
        
        if len(results):
            results = pd.concat(results, axis=1).T
            if self.fragments:
                self.update(results)
            else:
                self.add(results)
    
    def add_ip3366(self, pages: int = 1):
        base1 = 'http://www.ip3366.net/free/?stype=1&page={i}' 
        base2 = 'http://www.ip3366.net/free/?stype=2&page={i}'

        urls = []
        for i in range(1, pages + 1):
            for pattern in [base1, base2]:
                urls.append(pattern.format(i=i))
            
        resps = []
        for url in urls:
            resps.append(requests.get(url, headers=self.headers))
            time.sleep(1)
        
        results = []
        for resp in resps:
            text = resp.text
            proxies = re.findall(r'<td>(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})</td>[\s\S]*?<td>(\d+)</td>', text)
            for proxy in proxies:
                proxy = {"http": "http://" + ":".join(proxy), "https": "http://" + ":".join(proxy)}
                if self.check(proxy):
                    results.append(pd.Series(proxy, name=pd.to_datetime('now')))
        
        if len(results):
            results = pd.concat(results, axis=1).T
            if self.fragments:
                self.update(results)
            else:
                self.add(results)

    def add_89ip(self, pages: int = 1):
        url_base = "https://www.89ip.cn/index_{i}.html"

        resps = []
        for i in range(1, pages + 1):
            resps.append(requests.get(url_base.format(i=i), headers=self.headers))
            time.sleep(1)
        
        results = []
        for resp in resps:
            proxies = re.findall(
                r'<td.*?>[\s\S]*?(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})[\s\S]*?</td>[\s\S]*?<td.*?>[\s\S]*?(\d+)[\s\S]*?</td>',
                resp.text
            )
            for proxy in proxies:
                proxy = {"http": "http://" + ":".join(proxy), "https": "http://" + ":".join(proxy)}
                if self.check(proxy):
                    results.append(pd.Series(proxy, name=pd.to_datetime('now')))
        
        if len(results):
            results = pd.concat(results, axis=1).T
            if self.fragments:
                self.update(results)
            else:
                self.add(results)

@retry
def get_spot_data(proxy_manager: ProxyManager = None) -> pd.DataFrame:
    url = "http://82.push2.eastmoney.com/api/qt/clist/get"
    params = {
        "pn": "1", "pz": "50000", "po": "1", "np": "1", 
        "ut": "bd1d9ddb04089700cf9c27f6f7426281", "fltt": "2", "invt": "2",
        "fid": "f3", "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
        "fields": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152",
        "_": "1623833739532",
    }
    r = requests.get(url, proxies=proxy_manager.pick('http'), params=params, timeout=2)
    data_json = r.json()
    if not data_json["data"]["diff"]:
        return pd.DataFrame()
    temp_df = pd.DataFrame(data_json["data"]["diff"])
    temp_df.columns = [
        "_", "latest_price", "change_rate", "change_amount", "volume",
        "turnover", "amplitude", "turnover_rate", "pe_ratio_dynamic", 
        "volume_ratio", "five_minute_change", "code", "_", "name", "highest",
        "lowest", "open", "previous_close", "market_cap", "circulating_market_cap", 
        "speed_of_increase", "pb_ratio", "sixty_day_change_rate", 
        "year_to_date_change_rate", "-", "-", "-", "-", "-", "-", "-",
    ]
    
    temp_df = temp_df.dropna(subset=["code"]).set_index("code")
    temp_df = temp_df.drop(["-", "_"], axis=1)
    for col in temp_df.columns:
        if col != 'name':
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
    return temp_df


def wechat_login(appid: str, redirect_url: str):
    service_base = "https://open.weixin.qq.com/connect/qrconnect?appid={app_id}&scope=snsapi_login&redirect_uri={redirect_url}"
    qrcode_base = "https://open.weixin.qq.com/connect/qrcode/{uuid}"
    logincheck_base = "https://lp.open.weixin.qq.com/connect/l/qrconnect?uuid={uuid}&_={timestamp}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.150 Safari/537.36',
    }
    
    service_url = service_base.format(app_id=appid, redirect_url=redirect_url)
    soup = BeautifulSoup(requests.get(service_url, headers=headers).text, 'html.parser')
    if soup is None:
        raise ValueError("weixin third-party login failed")
    qrcode_img = soup.find('img', class_='qrcode lightBorder')
    uuid = qrcode_img['src'].split('/')[-1]

    qrcode_url = qrcode_base.format(uuid=uuid)
    img = requests.get(qrcode_url, headers=headers).content
    if img is None:
        raise ValueError("weixin third-party login failed")
    temp_qrcode_path = Path("login.png")
    with open(temp_qrcode_path, "wb") as qrcode_file:
        qrcode_file.write(img)

    def _login_check(_u):
        lp_response = requests.get(_u, headers=headers)
        if lp_response is None:
            raise ValueError("weixin third-party login failed")
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
            url = logincheck_base.format(uuid=uuid, timestamp=int(timestamp))
            wx_errorcode, wx_code = _login_check(url)
            
        elif wx_errorcode == '404':
            url = f"{logincheck_base.format(uuid=uuid, timestamp=int(timestamp))}&last=404"
            wx_errorcode, wx_code = _login_check(url)
            
        else:
            raise ValueError("weixin third-party login failed")


def ewechat_notify(
    key: str, 
    content_or_path: str,
    message_type: str = "text",
    mentions: str | list = None
):
    webhook_base = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={key}"
    upload_base = "https://qyapi.weixin.qq.com/cgi-bin/webhook/upload_media?key={key}&type={type}"
    notify_url = webhook_base.format(key=key)
    
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
        upload_url = upload_base.format(key=key, type=message_type)
        if not content_or_path:
            raise ValueError("path is required for file and voice")
        path = Path(content_or_path)
        with path.open('rb') as fp:
            file_info = {"media": (
                path.name, fp.read(), 
                "multipart/form-data", 
                {'Content-Length': str(path.stat().st_size)}
            )}
            resp = requests.post(upload_url, files=file_info).json()

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
        resp = requests.post(notify_url, json=message).json()
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
        resp = requests.post(notify_url, json=message).json()
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
        resp = requests.post(notify_url, json=message).json[0]
        return resp

    else:
        raise ValueError(f"Unsupported message type: {message_type}")


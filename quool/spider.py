import time
import base64
import random
import hashlib
import requests
import datetime
import pandas as pd
from pathlib import Path
from retrying import retry
from .table import ItemTable
from bs4 import BeautifulSoup


class ProxyManager(ItemTable):

    def __init__(
        self, 
        uri: str | Path, 
        size: int = 100,
        create: bool = False,
    ):
        self._size = size
        self._invalid = []
        super().__init__(uri, create)
    
    @property
    def spliter(self):
        return lambda x: x // self._size

    @property
    def namer(self):
        return lambda x: f'{x.index[0]}-{x.index[0] + self._size - 1}'

    @property
    def invalid(self):
        return self._invalid
    
    def add_invalid(self, index: int):
        self._invalid.append(index)
    
    def add_proxy(self, proxy: dict | list[dict]):
        max_index = self._read_fragment(self.fragments[-1]).index.max()
        proxy = [proxy] if not isinstance(proxy, list) else proxy
        proxy = pd.DataFrame([proxy], index=range(max_index + 1, max_index + len(proxy) + 1))
        self.update(proxy)
    
    def pick(self, field: str | list = None):
        max_index = self._read_fragment(self.fragments[-1]).index.max()
        index = random.choice(list(set(range(max_index + 1)) - set(self.invalid)))
        return self.read(field, start=[index]).to_dict('records')[0]


@retry()
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


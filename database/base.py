import quool
import time
import base64
import hashlib
import requests
from pathlib import Path
from bs4 import BeautifulSoup


qtd = quool.PanelTable("./data/quotes-day", code_level="order_book_id", date_level="date")
qtm = quool.PanelTable("./data/quotes-min", code_level="order_book_id", date_level="datetime")
fin = quool.PanelTable("./data/financial", code_level="order_book_id", date_level="date")
idxwgt = quool.PanelTable("./data/index-weights", code_level="order_book_id", date_level="date")
idxqtd = quool.PanelTable("./data/index-quotes-day", code_level="order_book_id", date_level="date")
idxqtm = quool.PanelTable("./data/index-quotes-min", code_level="order_book_id", date_level="datetime")
sec = quool.PanelTable("./data/security-margin", code_level="order_book_id", date_level="date")
ids = quool.PanelTable("./data/industry-info", code_level="order_book_id", date_level="date")
con = quool.PanelTable("./data/stock-connect", code_level="order_book_id", date_level="date")
div = quool.PanelTable("./data/dividend-split", code_level="order_book_id", date_level="date")
ins = quool.ItemTable("./data/instruments-info")
prx = quool.ProxyRecorder("./data/proxy")


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

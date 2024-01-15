import time
import base64
import hashlib
import requests
from pathlib import Path
from .core.request import Request
from .core.exception import RequestFailedError


class WeChat(Request):

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

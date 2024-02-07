import time
import base64
import hashlib
import requests
import datetime
import pandas as pd
from pathlib import Path
from .core.request import Request
from .core.util import Logger


class WeChat(Request):

    __webhook_base = "https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key={key}"
    __qrcode_base = "https://open.weixin.qq.com/connect/qrcode/{uuid}"
    __service_base = "https://open.weixin.qq.com/connect/qrconnect?appid={app_id}&scope=snsapi_login&redirect_uri={redirect_url}"
    __logincheck_base = "https://lp.open.weixin.qq.com/connect/l/qrconnect?uuid={uuid}&_={timestamp}"
    __upload_base = "https://qyapi.weixin.qq.com/cgi-bin/webhook/upload_media?key={key}&type={type}"

    def login(self, appid: str, redirect_url: str):
        service_url = self.__service_base.format(app_id=appid, redirect_url=redirect_url)
        soup = super().request(service_url, 'get').soup[0]
        if soup is None:
            raise ValueError("weixin third-party login failed")
        qrcode_img = soup.find('img', class_='qrcode lightBorder')
        uuid = qrcode_img['src'].split('/')[-1]

        qrcode_url = self.__qrcode_base.format(uuid=uuid)
        img = super().get(qrcode_url).content[0]
        if img is None:
            raise ValueError("weixin third-party login failed")
        temp_qrcode_path = Path("login.png")
        with open(temp_qrcode_path, "wb") as qrcode_file:
            qrcode_file.write(img)

        def _login_check(_u):
            lp_response = super(WeChat, self).get(_u).responses[0]
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
                url = self.__logincheck_base.format(uuid=uuid, timestamp=int(timestamp))
                wx_errorcode, wx_code = _login_check(url)
                
            elif wx_errorcode == '404':
                url = f"{self.__logincheck_base.format(uuid=uuid, timestamp=int(timestamp))}&last=404"
                wx_errorcode, wx_code = _login_check(url)
                
            else:
                raise ValueError("weixin third-party login failed")
            
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
                raise ValueError("weixin third-party login failed")
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
                raise ValueError("weixin third-party login failed")
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
                raise ValueError("weixin third-party login failed")
            return resp

        else:
            raise ValueError(f"Unsupported message type: {message_type}")


class SnowBall(Request):

    __group_list = 'https://tc.xueqiu.com/tc/snowx/MONI/trans_group/list.json'
    __group_add = 'https://tc.xueqiu.com/tc/snowx/MONI/trans_group/add.json'
    __group_delete = 'https://tc.xueqiu.com/tc/snowx/MONI/trans_group/delete.json'
    __transaction_list = 'https://tc.xueqiu.com/tc/snowx/MONI/transaction/list.json'
    __transaction_add = 'https://tc.xueqiu.com/tc/snowx/MONI/transaction/add.json'
    __transaction_delete = 'https://tc.xueqiu.com/tc/snowx/MONI/transaction/delete.json'
    __transfer_list = 'https://tc.xueqiu.com/tc/snowx/MONI/bank_transfer/query.json'
    __transfer_add = 'https://tc.xueqiu.com/tc/snowx/MONI/bank_transfer/add.json'
    __transfer_delete = 'https://tc.xueqiu.com/tc/snowx/MONI/bank_transfer/delete.json'
    __performace = 'https://tc.xueqiu.com/tc/snowx//MONI/performances.json'
    __quote = 'https://stock.xueqiu.com/v5/stock/batch/quote.json'

    basic_headers = {
        'Content-Type': 'application/x-www-form-urlencoded',
        'Accept': 'application/json, text/plain, */*',
        'Sec-Fetch-Site': 'same-site',
        'Accept-Language': 'zh-CN,zh-Hans;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Sec-Fetch-Mode': 'cors',
        'Host': 'tc.xueqiu.com',
        'Origin': 'https://xueqiu.com',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2.1 Safari/605.1.15',
        'Connection': 'keep-alive',
        'Sec-Fetch-Dest': 'empty',
    }

    def __init__(
        self, 
        token: str,
        headers: dict = None, 
        proxies: list[dict] = None, 
        timeout: float = None, 
        retry: int = 1, 
        delay: float = 2, 
        loglevel: int = 10,
        logfile: str = None,
    ) -> None:
        super().__init__(headers, proxies, timeout, retry, delay, loglevel, logfile)
        self.headers['Cookie'] = f'xq_a_token={token}'
        self.logger = Logger('SnowBall', display_name=True, level=loglevel)
    
    def group_list(self):
        result = self.get(self.__group_list).json[0]
        if result["success"]:
            return result["result_data"]["trans_groups"]
        return result

    def group_add(self, name: str):
        data = {'name': name}
        return self.post(self.__group_add, data=data).json[0]

    def group_delete(self, gid: str):
        data = {
            'gid': str(gid),
        }
        return self.post(self.__group_delete, data=data).json[0]
    
    def transfer_list(self, gid: str, row: int = 50):
        param = {
            'gid': str(gid),
            'row': str(row),
        }
        return self.get(self.__transfer_list, params=param).json[0]

    def transfer_add(self, gid: str, amount: int, date: str = None):
        date = date or datetime.datetime.now().strftime(r'%Y-%m-%d')
        typ = '1' if amount > 0 else '2'
        data = {
            'gid': str(gid),
            'date': pd.to_datetime(date).strftime(r'%Y-%m-%d'),
            'amount': str(abs(amount)),
            'type': typ,
            'market': 'CHA',
        }
        return self.post(self.__transfer_add, data=data).json[0]

    def transfer_delete(
        self, 
        gid: str, 
        tid: str,
    ):
        data = {
            'gid': str(gid),
            'tid': str(tid),
        }
        return self.post(self.__transfer_delete, data=data).json[0]
    
    def transaction_list(
        self, 
        gid: str, 
        symbol: str = None,
        row: int = 1000,
    ):
        param = {
            'gid': str(gid),
            'row': str(row),
        }
        if symbol:
            param['symbol'] = symbol
        return self.get(self.__transaction_list, params=param).json[0]
    
    def transaction_add(
        self, 
        gid: str, 
        symbol: str, 
        shares: int, 
        price: float = None, 
        date: str = None,
        tax_rate: float = 0.25,
        commission_rate: float = 0,
        comment: str = None,
    ):
        date = date or datetime.datetime.now().strftime(r'%Y-%m-%d')
        typ = '1' if shares > 0 else '2'
        price = price or self.quote(symbol).loc[symbol, 'current']
        data = {
            'type': typ,
            'date': pd.to_datetime(date).strftime(r'%Y-%m-%d'),
            'gid': str(gid),
            'symbol': symbol,
            'price': f'{price:.2f}',
            'shares': f'{abs(shares):.0f}',
            'tax_rate': f'{tax_rate:.2f}',
            'commission_rate': f'{commission_rate:.2f}',
        }
        if comment:
            data.update({"comment": comment})
        return self.post(self.__transaction_add, data=data).json[0]

    def transaction_delete(self, gid: str, tid: str):
        data = {'gid': str(gid), 'tid': str(tid),}
        return self.post(self.__transaction_delete, data=data).json[0]

    def performance(self, gid: str):
        param = {'gid': str(gid)}
        return self.get(self.__performace, params=param).json[0]
    
    def quote(self, symbol: str | list):
        self.headers["Host"] = 'stock.xueqiu.com'
        param = {"symbol": symbol, "extend": "detail"}
        result = self.get(self.__quote, params=param).json[0]
        self.headers["Host"] = 'tc.xueqiu.com'
        if result["error_code"] == 0:
            data = [res["quote"] for res in result["data"]["items"]]
            return pd.DataFrame(data).set_index("symbol")
        return result["error_description"]

    def order_target_percent(
        self, 
        gid: str, 
        weight: pd.Series,
    ):
        perf = self.performance(gid)["result_data"]["performances"]
        value = perf[0]["assets"]
        cash = perf[0]["cash"]
        target = weight * value
        
        # position exists
        if len(perf) > 1:
            position = pd.DataFrame(perf[1]["list"]).set_index('symbol')
            position.index = position.index.str.slice(2)
        # no position at all
        else:
            position = pd.DataFrame()

        holds = position.index.intersection(target.index)
        holds = position.loc[holds, "current"] * position.loc[holds, "shares"]
        adjust = target - holds
        sells = position.index.difference(target.index)
        buys = target.index.difference(position.index)

        self.logger.debug("-" * 10 + "SELLINGS" + "-" * 10)
        for code in sells:
            self.transaction_add(gid, 
                f'SH{code}' if code.startswith('6') else f'SZ{code}', 
                -position.loc[code, "shares"], position.loc[code, "current"])
            self.logger.info(f"{code} closed position {shares:.0f} shares")
            cash += position.loc[code, "current"] * position.loc[code, "shares"]
            time.sleep(self.delay)
        
        self.logger.debug("-" * 10 + "ADJUSTING" + "-" * 10)
        for code in adjust.index:
            if cash < adjust.loc[code]:
                self.logger.warning(f"short in cash abort adjust {code}")
                continue
            shares = (adjust.loc[code] / position.loc[code, "current"] // 100) * 100
            if 0 < shares < 100:
                self.logger.warning(f"not enough share: {shares} to adjust")
                continue
            self.transaction_add(gid, 
                f'SH{code}' if code.startswith('6') else f'SZ{code}', 
                shares, position.loc[code, "current"])
            self.logger.info(f"{code} adjusted position {shares:.0f} shares")
            cash -= shares * position.loc[code, "current"]
            time.sleep(self.delay)
        
        self.logger.debug("-" * 10 + "BUYINGS" + "-" * 10)
        for code in buys:
            if cash < target.loc[code]:
                self.logger.warning(f"short in cash abort {code}")
                continue
            price = self.quote(f'SH{code}' if code.startswith('6') else f'SZ{code}').loc[code, 'current']
            shares = (target.loc[code] / price // 100) * 100
            if shares < 100:
                self.logger.warning(f"current cash {cash} is not enough for 100 share")
                continue
            self.transaction_add(gid, f'SH{code}' if code.startswith('6') else f'SZ{code}', shares, price)
            self.logger.info(f"{code} buy {shares} at {price}")
            cash -= price * shares
            time.sleep(self.delay)

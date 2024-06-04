import re
import time
import random
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lxml import etree
from pathlib import Path
from joblib import Parallel, delayed
from .table import ItemTable, PanelTable
from .util import parse_commastr, evaluate


class Proxy(ItemTable):

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
        rand_frag = rand_frag[0]
        proxy = self._read_fragment(rand_frag)
        if field is not None:
            field = field if isinstance(field, list) else [field]
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
            try:
                resps.append(requests.get(url_base.format(i=i), headers=self.headers))
            except:
                pass
            time.sleep(1)
        
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
            try:
                resps.append(requests.get(url, headers=self.headers))
            except:
                pass
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
            try:
                resps.append(requests.get(url, headers=self.headers))
            except:
                pass
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
            try:
                resps.append(requests.get(url_base.format(i=i), headers=self.headers))
            except:
                pass
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

class Transaction(ItemTable):

    def __init__(
        self, 
        uri: str | Path, 
        principle: float = 1_000_000.00,
        start: str | pd.Timestamp = None,
    ):
        super().__init__(uri, True)
        if not self.fragments:
            if principle is None or start is None:
                raise ValueError("principle and start must be specified when initiating")
            init = pd.DataFrame([{
                "code": "cash",
                "notify_time": pd.to_datetime(start), 
                'type': "transfer",
                'status': "Completed",
                'created_time': start,
                'created_price': 1,
                'created_size': principle,
                'executed_time': start,
                'executed_price': 1,
                'executed_size': principle,
                'price_limit': np.nan,
                'trail_amount': np.nan,
                'trail_percent': np.nan,
                'execute_type': 'Market',
                "commission": 0.0,
                "amount": float(principle), 
            }], index=pd.Index([0], name="reference"))
            self.update(init)
    
    @property
    def spliter(self):
        return pd.Grouper(key='datetime', freq='ME')

    @property
    def namer(self):
        return lambda x: x['datetime'].iloc[0].strftime('%Y%m')

    def read(
        self, 
        column: str | list = None, 
        start: str | list = None, 
        stop: str = None, 
        code: str | list[str] = None,
        filters: list[list[tuple]] = None,
    ):
        filters = filters or []
        if code is not None:
            filters += [("code", "in", parse_commastr(code))]
        return super().read(column, start, stop, "datetime", filters)

    def prune(self):
        for frag in self.fragments:
            self._fragment_path(frag).unlink()
        
    def trade(
        self, 
        date: str | pd.Timestamp,
        code: str,
        ref: int,
        size: float = None,
        price: float = None,
        commission: float = 0,
        **kwargs,
    ):
        size = size
        price = price

        trade = pd.DataFrame([{
            "notify_time": pd.to_datetime(date),
            "code": code, "size": size,
            "price": price, "commission": commission, **kwargs
        }], index=[pd.to_datetime('now')])
        if code != "cash":
            cash = pd.DataFrame([{
                "notify_time": pd.to_datetime(date),
                "code": "cash", "size": -size * price - commission,
                "price": 1, "commission": 0,
            }], index=[pd.to_datetime('now')])
            trade = pd.concat([trade, cash], axis=0)
        
        if kwargs:
            self.add(dict((k, type(v)) for k, v in kwargs.items()))
        self.update(trade)

    def peek(self, date: str | pd.Timestamp = None, price: pd.Series = None) -> pd.Series:
        df = self.read(filters=[("datetime", "<=", pd.to_datetime(date or 'now'))])
        df = df.groupby("code")[["size", "amount", "commission"]].sum()
        df["cost"] = (df["amount"] / df["size"]).replace([-np.inf, np.inf], 0)
        if price is None:
            return df
        price = price.copy()
        price.loc["cash"] = 1
        indexer = price.index.get_indexer_for(df.index)
        df["price"] = price.iloc[indexer[indexer != -1]]
        df["value"] = df["price"] * df["size"]
        df['pnl'] = ((df['price'] - df['cost']) * df['size']).where(df['size'] != 0, -df['amount'])
        return df
        
    def report(
        self, 
        price: pd.Series, 
        benchmark: pd.Series = None,
        code_level: int | str = 0,
        date_level: int | str = 1,
        image: str | bool = True,
        result: str = None,
    ) -> pd.DataFrame:
        data = self.read(["datetime", "code", "size", "amount", "commission"])
        if isinstance(price, pd.DataFrame) and price.index.nlevels == 1:
            code_level = code_level if not isinstance(code_level, int) else "code"
            date_level = date_level if not isinstance(date_level, int) else "datetime"
            price = price.stack().sort_index().to_frame("price")
            price.index.names = [date_level, code_level]
        price = price.squeeze().sort_index()
        dates = price.index.get_level_values(date_level).unique()
        dates = dates[dates >= data["datetime"].min()]

        data = data.groupby(["code", "datetime"]).sum()
        data = data.reindex(pd.MultiIndex.from_product(
            [data.index.levels[0], dates], names=["code", "datetime"]))
        cash = data.loc["cash", "amount"]
        cash = cash.fillna(0).cumsum()

        noncash = data.drop(labels="cash", axis=0)
        noncash.index.names = price.index.names
        price = price.loc[noncash.index.intersection(price.index)]
        delta = (price * noncash["size"]).groupby(level=date_level).sum()
        noncash = noncash.fillna(0).groupby(level=code_level, group_keys=False).cumsum()
        market = (price * noncash["size"]).groupby(level=date_level).sum()
        market = market.reindex(cash.index).fillna(0)
        value = market + cash
        turnover = (delta / value.shift(1)).fillna(0)

        data = pd.concat([value, cash, turnover], axis=1, keys=["value", "cash", "turnover"])
        if image or result:
            return evaluate(
                data["value"], 
                data["cash"],
                data["turnover"],
                benchmark=benchmark,
                image=image,
                result=result
            )
        return data

class Factor(PanelTable):

    def read(
        self, 
        field: str | list = None, 
        code: str | list = None, 
        start: str | list = None, 
        stop: str = None, 
        processor: list = None,
    ) -> pd.Series | pd.DataFrame:
        processor = processor or []
        if not isinstance(processor, list):
            processor = [processor]
        
        df = super().read(field, code, start, stop)
        
        if df.columns.size == 1:
            df = df.squeeze().unstack(level=self._code_level)
        
        for proc in processor:
            kwargs = {}
            if isinstance(proc, tuple):
                proc, kwargs = proc
            df = proc(df, **kwargs)
        return df.dropna(axis=0, how='all')

    def get_trading_days(
        self,
        start: str | pd.Timestamp = None,
        stop: str | pd.Timestamp = None,
    ):
        frag = self._read_fragment(self.fragments[0])
        field = frag.columns[0]
        start = start or frag.index.get_level_values(self._date_level).min()
        code = frag.index.get_level_values(self._code_level).min()
        dates = super().read(field, code=code, start=start, stop=stop
            ).droplevel(self._code_level).index
        return dates
    
    def get_trading_days_rollback(
        self, 
        date: str | pd.Timestamp = None, 
        rollback: int = 1
    ):
        date = pd.to_datetime(date or 'now')
        if rollback >= 0:
            trading_days = self.get_trading_days(start=None, stop=date)
            rollback = trading_days[trading_days <= date][-rollback - 1]
        else:
            trading_days = self.get_trading_days(start=date, stop=None)
            rollback = trading_days[min(len(trading_days) - 1, -rollback)]
        return rollback

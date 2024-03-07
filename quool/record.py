import re
import time
import random
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lxml import etree
from pathlib import Path
from .table import ItemTable


class RunRecorder(ItemTable):

    def __init__(
        self, 
        uri: str | Path, 
        model_path: str | Path,
    ):
        super().__init__(uri, True)
        self._model_path = model_path
    
    @property
    def spliter(self):
        return pd.Grouper(leve=0, freq='D')

    @property
    def namer(self):
        return lambda x: x.index[0].strftime(f"%Y-%m-%d")

    def record(self, **kwargs):
        rec = pd.DataFrame([kwargs], index=[pd.to_datetime('now')])
        existed = rec[rec.columns.isin(self.columns)]
        nonexisted = rec[~rec.columns.isin(self.columns)]
        if not existed.empty:
            self.update(rec)
        if not nonexisted.empty:
            self.add(rec)


class TradeRecorder(ItemTable):

    def __init__(
        self, 
        uri: str | Path, 
        principle: float = None, 
        start_date: str | pd.Timestamp = None,
    ):
        super().__init__(uri, True)
        if not self.fragments:
            if principle is None or start_date is None:
                raise ValueError("principle and start_date must be specified when initiating")
            self.add(pd.DataFrame([{
                "datetime": pd.to_datetime(start_date), 
                "code": "cash", "size": float(principle), 
                "price": 1.0, "amount": float(principle), "commission": 0.0
            }], index=[pd.to_datetime('now')]))
        
        if not self.columns.isin(["datetime", "code", "size", "price", "amount", "commission"]).all():
            raise ValueError("The table must have columns datetime, code, size, price, amount, commission")
    
    @property
    def spliter(self):
        return pd.Grouper(key='datetime', freq='M')

    @property
    def namer(self):
        return lambda x: x['datetime'].iloc[0].strftime('%Y%m')
    
    def prune(self):
        for frag in self.fragments:
            self._fragment_path(frag).unlink()

    def trade(
        self, 
        date: str | pd.Timestamp,
        code: str,
        size: float = None,
        price: float = None,
        amount: float = None,
        commission: float = 0,
        **kwargs,
    ):
        if size is None and price is None and amount is None:
            raise ValueError("two of size, price or amount must be specified")
        size = size if size is not None else (amount / price)
        price = price if size is not None else (amount / size)
        amount = amount if size is not None else (size * price)

        trade = pd.DataFrame([{
            "datetime": pd.to_datetime(date),
            "code": code, "size": size,
            "price": price, "amount": amount,
            "commission": commission, **kwargs
        }], index=[pd.to_datetime('now')])
        if code != "cash":
            cash = pd.DataFrame([{
                "datetime": pd.to_datetime(date),
                "code": "cash", "size": -size * price - commission,
                "price": 1, "commission": 0,
                "amount": -size * price - commission, **kwargs
            }], index=[pd.to_datetime('now')])
            trade = pd.concat([trade, cash], axis=0)
        
        self.update(trade)

    def peek(self, date: str | pd.Timestamp = None) -> pd.Series:
        df = self.read(filters=[("datetime", "<=", pd.to_datetime(date or 'now'))])
        return df.groupby("code")[["size", "amount", "commission"]].sum()
        
    def report(
        self, 
        price: pd.Series, 
        code_level: int | str = 0,
        date_level: int | str = 1,
    ) -> pd.DataFrame:
        data = self.read(["datetime", "code", "size", "amount", "commission"])
        if isinstance(price, pd.DataFrame) and price.index.nlevels == 1:
            code_level = code_level if not isinstance(code_level, int) else "code"
            date_level = date_level if not isinstance(date_level, int) else "datetime"
            price = price.stack().sort_index().to_frame("price")
            price.index.names = [date_level, code_level]
        price = price.sort_index()
        dates = price.index.get_level_values(date_level).unique()
        dates = dates[(dates <= data["datetime"].max()) & (dates >= data["datetime"].min())]

        data = data.groupby(["code", "datetime"]).sum()
        data = data.groupby("code").apply(lambda x: x.droplevel('code').reindex(
            dates.union(x.index.get_level_values('datetime').unique().sort_values())
        ))
        cash = data.loc["cash", "amount"]
        cash = cash.fillna(0).cumsum()

        noncash = data.drop(labels="cash", axis=0)
        noncash.index.names = price.index.names
        # if it raise, there are some price not available
        price = price.loc[noncash.index]
        delta = (price * noncash["size"]).groupby(level=date_level).sum()
        noncash = noncash.groupby(level=code_level, group_keys=False).apply(lambda x: x.fillna(0).cumsum())
        market = (price * noncash["size"]).groupby(level=date_level).sum()
        market = market.reindex(cash.index).fillna(0)
        value = market + cash
        turnover = delta / value.shift(1)

        data = pd.concat([value, cash, turnover], axis=1, keys=["value", "cash", "turnover"])
        return data

    @staticmethod
    def evaluate(
        value: pd.Series, 
        cash: pd.Series = None,
        turnover: pd.Series = None,
        benchmark: pd.Series = None,
        image: str = None,
        result: str = None,
    ):
        cash = cash if isinstance(cash, (pd.Series, pd.DataFrame)) else \
            pd.Series(np.zeros(value.shape[0]), index=value.index)
        returns = value.pct_change(fill_method=None).fillna(0)
        if benchmark is not None:
            benchmark = benchmark.squeeze()
            benchmark_returns = benchmark.pct_change(fill_method=None).fillna(0)
        drawdown = value / value.cummax() - 1
        
        # evaluation indicators
        evaluation = pd.Series(name='evaluation')
        evaluation['total_return(%)'] = (value.iloc[-1] / value.iloc[0] - 1) * 100
        evaluation['annual_return(%)'] = ((evaluation['total_return(%)'] / 100 + 1) ** (
            365 / (value.index.max() - value.index.min()).days) - 1) * 100
        evaluation['annual_volatility(%)'] = (returns.std() * np.sqrt(252)) * 100
        down_volatility = (returns[returns < 0].std() * np.sqrt(252)) * 100
        maxdate = drawdown.idxmax()
        startdate = drawdown.loc[:maxdate][drawdown.loc[:maxdate] == 0].index[-1]
        evaluation['max_drawdown(%)'] = (drawdown.max()) * 100
        evaluation['max_drawdown_period(days)'] = maxdate - startdate
        evaluation['max_drawdown_start'] = startdate
        evaluation['max_drawdown_stop'] = maxdate
        evaluation['daily_turnover(%)'] = turnover.mean() * 100 if turnover is not None else np.nan
        evaluation['sharpe_ratio'] = evaluation['annual_return(%)'] / evaluation['annual_volatility(%)'] \
            if evaluation['annual_volatility(%)'] != 0 else np.nan
        evaluation['sortino_ratio'] = evaluation['annual_return(%)'] / down_volatility \
            if down_volatility != 0 else np.nan
        evaluation['calmar_ratio'] = evaluation['annual_return(%)'] / evaluation['max_drawdown(%)'] \
            if evaluation['max_drawdown(%)'] != 0 else np.nan
        if benchmark is not None:
            exreturns = returns - benchmark_returns
            benchmark_volatility = (benchmark_returns.std() * np.sqrt(252)) * 100
            exvalue = (1 + exreturns).cumprod()
            evaluation['total_exreturn(%)'] = (exvalue.iloc[-1] - exvalue.iloc[0]) * 100
            evaluation['annual_exreturn(%)'] = ((evaluation['total_exreturn(%)'] / 100 + 1
                ) ** (365 / (exvalue.index.max() - exvalue.index.min()).days) - 1) * 100
            evaluation['annual_exvolatility(%)'] = (exreturns.std() * np.sqrt(252)) * 100
            evaluation['beta'] = returns.cov(benchmark_returns) / benchmark_returns.var()
            evaluation['alpha(%)'] = (returns.mean() - (evaluation['beta'] * (benchmark_returns.mean()))) * 100
            evaluation['treynor_ratio(%)'] = (evaluation['annual_exreturn(%)'] / evaluation['beta'])
            evaluation['information_ratio'] = evaluation['annual_exreturn(%)'] / benchmark_volatility \
                if benchmark_volatility != 0 else np.nan

        data = pd.concat([value, cash, returns, drawdown, turnover], 
            axis=1, keys=['value', 'cash', 'returns', 'drawdown', 'turnover'])
        if benchmark is not None:
            data = pd.concat([data, exreturns.to_frame('exreturns'),
                exvalue.to_frame('exvalue')], axis=1)
        
        if result is not None:
            data.to_excel(result, sheet_name="performances")
        
        if image is not None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
            data[["value", "cash"]].plot(ax=ax, title="Portfolio", color=['#1C1C1C', '#EE7600'])
            data[["returns", "drawdown"]].plot(ax=ax, alpha=0.5, 
                secondary_y=True, label=['returns', "drawdown"],
                color=["#9400D3", "#7CFC00"])
            if turnover is not None:
                data["turnover"].plot(ax=ax, alpha=0.5, secondary_y=True, color="#66CDAA", label="turnover")
            if isinstance(image, (str, Path)):
                fig.savefig(image)

        return evaluation


class ProxyRecorder(ItemTable):

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

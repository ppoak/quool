import re
import time
import random
import requests
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from lxml import etree
from pathlib import Path
from joblib import Parallel, delayed
from .table import ItemTable, PanelTable
from .tool import parse_commastr, evaluate


class Transaction(ItemTable):

    def __init__(
        self, uri: str | Path, 
        create: bool = False
    ):
        super().__init__(uri, create)
        self._id = itertools.count(
            self.read("reference", status="Completed").iloc[:, 0].max()
        )

    @property
    def spliter(self):
        return pd.Grouper(key='notify_time', freq='ME')

    @property
    def namer(self):
        return lambda x: x['notify_time'].iloc[0].strftime('%Y%m')

    def read(
        self, 
        column: str | list = None, 
        start: str | list = None, 
        stop: str = None, 
        code: str | list[str] = None,
        otype: str | list[str] = None,
        status: str | list[str] = None,
        filters: list[list[tuple]] = None,
    ):
        filters = filters or []
        if code is not None:
            filters += [("code", "in", parse_commastr(code))]
        if otype is not None:
            filters += [("type", "in", parse_commastr(otype))]
        if status is not None:
            filters += [("status", "in", parse_commastr(status))]
        return super().read(column, start, stop, "notify_time", filters)

    def prune(self):
        for frag in self.fragments:
            self._fragment_path(frag).unlink()
        
    def trade(
        self, 
        time: str | pd.Timestamp,
        code: str,
        price: float,
        size: float,
        commission: float,
        **kwargs,
    ):
        trade = pd.DataFrame([{
            "notify_time": time,
            "code": code,
            'reference': next(self._id),
            'type': "Buy" if size > 0 else "Sell",
            'status': "Completed",
            'created_time': time,
            'created_price': price,
            'created_size': size,
            'executed_time': time,
            'executed_price': price,
            'executed_size': size,
            'execute_type': "Market",
            'commission': commission,
            **kwargs
        }], index=[pd.to_datetime('now')])
        
        if kwargs:
            self.add(dict((k, type(v)) for k, v in kwargs.items()))
        self.update(trade)

    def summary(
        self, 
        start: str | pd.Timestamp = None, 
        stop: str | pd.Timestamp = None,
        price: pd.Series = None
    ) -> pd.Series:
        trans = self.read(
            "code, executed_size, executed_price, commission", 
            start=start, stop=stop, status="Completed"
        )
        trans["executed_amount"] = trans["executed_size"] * trans["executed_price"]
        stat = trans.groupby("code").agg(
            size=("executed_size", "sum"),
            avgcost=("executed_size", lambda x: (
                trans.loc[x.index, "executed_amount"].sum() + trans.loc[x.index, "commission"].sum()
            ) / trans.loc[x.index, "executed_size"].sum() if 
            trans.loc[x.index, "executed_size"].sum() > 0 else np.nan
            )
        )
        if price is None:
            return stat
        
        price = price.copy()
        price.loc["Cash"] = 1
        indexer = price.index.get_indexer_for(stat.index)
        stat["current"] = price.iloc[indexer[indexer != -1]]
        return stat
        
    def report(
        self, 
        price: pd.Series, 
        principle: float,
        start: str | pd.Timestamp = None,
        stop: str | pd.Timestamp = None,
        benchmark: pd.Series = None,
        code_level: int | str = 0,
        date_level: int | str = 1,
        image: str | bool = True,
    ) -> pd.DataFrame:
        code_level = code_level if not isinstance(code_level, int) else "code"
        date_level = date_level if not isinstance(date_level, int) else "datetime"

        data = self.read(
            ["code", "executed_time", "executed_size", "executed_price", "commission"],
            start=start, stop=stop, status="Completed",
        )
        data["executed_amount"] = data["executed_size"] * data["executed_price"] + data["commission"]
        data = data.groupby(["executed_time", "code"]).sum()
        # this is for `get_level_values(code_level)`
        data.index.names = [date_level, code_level]

        if isinstance(price, pd.DataFrame) and price.index.nlevels == 1:
            price = price.stack().sort_index().to_frame("price")
            price.index.names = [date_level, code_level]
        price = price.reorder_levels([date_level, code_level])
        
        codes = data.index.get_level_values(code_level).unique()
        dates = price.index.get_level_values(date_level).unique()
        cashp = pd.Series(np.ones(dates.size), index=pd.MultiIndex.from_product(
            [dates, ["Cash"]], names=[date_level, code_level]
        ))
        price = pd.concat([price, cashp], axis=0)
        price = price.squeeze().loc(axis=0)[:, codes]
        
        data = data.reindex(price.index)
        # for ensurance
        data.index.names = [date_level, code_level]
        
        _rev = pd.Series(np.ones(data.index.size), index=data.index)
        _rev = _rev.where(data.index.get_level_values(code_level) == "Cash", -1)
        cash = (data["executed_amount"] * _rev).groupby(level=date_level).sum().cumsum()
        cash += principle
        size = data["executed_size"].drop(index="Cash", level=code_level).fillna(0).groupby(level=code_level).cumsum()
        market = (size * price).groupby(level=date_level).sum()
        value = market + cash
        turnover = market.diff() / value

        return evaluate(value, cash, turnover, benchmark=benchmark, image=image)


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
    
    def get_future(
        self, 
        ptype: str,
        period: int = 1, 
        start: str | pd.Timestamp = None,
        stop: str | pd.Timestamp = None,
        skip_nonperiod_day: bool = False,
        nonrealizable: pd.DataFrame = None,
    ):
        if stop is not None:
            stop = self.get_trading_days_rollback(stop, -period - 1)
        price = self.read(ptype, start=start, stop=stop)
        price = price.where(~nonrealizable, other=np.nan)
        future = price.shift(-1 - period) / price.shift(-1) - 1
        future = future.dropna(axis=0, how='all')

        if skip_nonperiod_day:
            return future.iloc[::period].squeeze()
        return future.squeeze()
    
    def _prepare_factor(
        self, 
        factor: str | pd.DataFrame | pd.Series,
        start: str | pd.Timestamp = None,
        stop: str | pd.Timestamp = None,
        processor: list = None
    ):
        if isinstance(factor, pd.Series) and factor.index.nlevels == 2:
            factor = factor.unstack(self._code_level)

        elif isinstance(factor, pd.DataFrame):
            factor = factor.loc[start:stop]

        elif isinstance(factor, str):
            factor = self.read(factor, start=start, stop=stop)
        
        else:
            ValueError("Invalid factor type")
        
        processor = processor or []
        for proc in processor:
            kwargs = {}
            if isinstance(proc, tuple):
                proc, kwargs = proc
            factor = proc(factor, **kwargs)
        
        if isinstance(start, (list, pd.DatetimeIndex)):
            factor = factor.loc[start]
        else:
            factor = factor.loc[start:stop]

        return factor
    
    def perform_crosssection(
        self, 
        factor: str | pd.DataFrame, 
        *,
        date: str | pd.Timestamp | pd.DataFrame | pd.Series,
        processor: list = None,
        period: int = 1,
        ptype: str = "volume_weighted_price",
        image: str | bool = True, 
        result: str = None
    ):
        future = self.get_future(ptype, period, date, date)
        factor = self._prepare_factor(factor, future.name, future.name, processor)
        data = pd.concat([factor.squeeze(), future], axis=1, keys=["Factor", future.name])

        if image is not None:
            pd.plotting.scatter_matrix(data, figsize=(20, 20), hist_kwds={'bins': 100})
            
            plt.tight_layout()
            if isinstance(image, (str, Path)):
                plt.savefig(image)
            else:
                plt.show()
                
        if result is not None:
            data.to_excel(result)
        
        return data.corr()

    def perform_inforcoef(
        self,
        factor: str | pd.DataFrame,
        *,
        period: int = 1,
        start: str = None,
        stop: str = None,
        ptype: str = "volume_weighted_price",
        processor: list = None,
        rolling: int = 20, 
        method: str = 'pearson', 
        skip_nonperiod_day: bool = False,
        image: str | bool = True, 
        result: str = None
    ):
        future = self.get_future(ptype, period, start, stop)
        
        if skip_nonperiod_day:
            factor = self._prepare_factor(factor, future.index, None, processor)
        else:
            factor = self._prepare_factor(factor, start=start, stop=stop, processor=processor)

        inforcoef = factor.corrwith(future, axis=1, method=method).dropna()
        inforcoef.name = f"infocoef"

        if image is not None:
            fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            inforcoef.plot(ax=ax, label='infor-coef', alpha=0.7, title='Information Coef')
            inforcoef.rolling(rolling).mean().plot(linestyle='--', ax=ax, label='trend')
            inforcoef.cumsum().plot(linestyle='-.', secondary_y=True, ax=ax, label='cumm-infor-coef')
            pd.Series(np.zeros(inforcoef.shape[0]), index=inforcoef.index).plot(color='grey', ax=ax, alpha=0.5)
            ax.legend()
            fig.tight_layout()
            if not isinstance(image, bool):
                fig.savefig(image)
            else:
                fig.show()
        
        if result is not None:
            inforcoef.to_excel(result)
        return inforcoef
    
    def perform_grouping(
        self, 
        factor: str | pd.DataFrame,
        *,
        period: int = 1,
        start: str = None,
        stop: str = None,
        processor: list = None,
        ptype: str = "volume_weighted_price",
        ngroup: int = 5, 
        commission: float = 0.002, 
        skip_nonperiod_day: bool = True,
        n_jobs: int = 1,
        image: str | bool = True, 
        result: str = None
    ):
        future = self.get_future(ptype, period, start, stop)
        
        if skip_nonperiod_day:
            factor = self._prepare_factor(factor, start=future.index, processor=processor)
        else:
            factor = self._prepare_factor(factor, start=start, stop=stop, processor=processor)
        
        # ngroup test
        try:
            groups = factor.apply(lambda x: pd.qcut(x, q=ngroup, labels=False), axis=1) + 1
        except:
            for date in factor.index:
                try:
                    pd.qcut(factor.loc[date], q=ngroup, labels=False)
                except:
                    raise ValueError(f"on date {date}, grouping failed")
        
        def _grouping(x):
            group = groups.where(groups == x)
            weight = (group / group).fillna(0)
            weight = weight.div(weight.sum(axis=1), axis=0)
            _period = period if skip_nonperiod_day else 1
            delta = weight.diff(periods=_period).fillna(0)
            turnover = delta.abs().sum(axis=1) / 2 / _period
            ret = (future * weight).sum(axis=1).shift(1) / _period
            ret -= commission * turnover
            ret = ret.fillna(0)
            val = (ret + 1).cumprod()
            return {
                'evaluation': evaluate(val, turnover=turnover, image=False),
                'value': val, 'turnover': turnover,
            }
            
        ngroup_result = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_grouping)(i) for i in range(1, ngroup + 1))
        ngroup_evaluation = pd.concat([res['evaluation'] for res in ngroup_result], 
            axis=1, keys=range(1, ngroup + 1)).add_prefix('group')
        ngroup_value = pd.concat([res['value'] for res in ngroup_result], 
            axis=1, keys=range(1, ngroup + 1)).add_prefix('group')
        ngroup_turnover = pd.concat([res['turnover'] for res in ngroup_result], 
            axis=1, keys=range(1, ngroup + 1)).add_prefix('group')
        ngroup_returns = ngroup_value.pct_change().fillna(0)
        longshort_returns = ngroup_returns[f"group{ngroup}"] - ngroup_returns["group1"]
        longshort_value = (longshort_returns + 1).cumprod()
        longshort_evaluation = evaluate(longshort_value, image=False)
        
        # naming
        longshort_evaluation.name = "longshort"
        longshort_value.name = "longshort value"

        if image is not None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
            longshort_value.plot(ax=ax, linestyle='--')
            ngroup_value.plot(ax=ax, alpha=0.8)
            ngroup_turnover.plot(ax=ax, secondary_y=True, alpha=0.2)
            fig.tight_layout()
            if isinstance(image, (str, Path)):
                fig.savefig(image)
            else:
                fig.show()            
        
        if result is not None:
            with pd.ExcelWriter(result) as writer:
                ngroup_evaluation.to_excel(writer, sheet_name="ngroup_evaluation")
                longshort_evaluation.to_excel(writer, sheet_name="longshort_evaluation")
                ngroup_value.to_excel(writer, sheet_name="ngroup_value")
                ngroup_turnover.to_excel(writer, sheet_name="ngroup_turnover")
                longshort_value.to_excel(writer, sheet_name="longshort_value")
        
        return pd.concat([ngroup_evaluation, longshort_evaluation], axis=1)
                
    def perform_topk(
        self, 
        factor: str | pd.DataFrame,
        *,
        period: int = 1,
        start: str = None,
        stop: str = None,
        ptype: str = "volume_weighted_price",
        processor: list = None,
        topk: int = 100, 
        commission: float = 0.002, 
        skip_nonperiod_day: bool = True,
        image: str | bool = True, 
        result: str = None
    ):
        future = self.get_future(ptype, period, start, stop)

        if skip_nonperiod_day:
            factor = self._prepare_factor(factor, start=future.index, processor=processor)
        else:
            factor = self._prepare_factor(factor, start, stop, processor)
            
        topks = factor.rank(ascending=False, axis=1) < topk
        topks = factor.where(topks)
        topks = (topks / topks).div(topks.count(axis=1), axis=0).fillna(0)
        _period = period if skip_nonperiod_day else 1
        turnover = topks.diff(periods=_period).fillna(0).abs().sum(axis=1) / 2 / _period
        ret = (topks * future).sum(axis=1).shift(1).fillna(0) - turnover * commission
        ret = ret.fillna(0) / _period
        val = (1 + ret).cumprod()
        eva = evaluate(val, turnover=turnover, image=False)

        val.name = "value"
        turnover.name = "turnover"
        eva.name = "evaluation"

        if image is not None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
            val.plot(ax=ax, title="Top K")
            turnover.plot(ax=ax, secondary_y=True, alpha=0.5)
            fig.tight_layout()
            if not isinstance(image, bool):
                fig.savefig(image)
            else:
                fig.show()

        if result is not None:
            pd.concat([eva, val, turnover], axis=1).to_excel(result)

        return eva

    def get(self, name: str, trading_days: pd.DatetimeIndex, n_jobs: int = -1, start: str = None, stop: str = None):
        result = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(getattr(self, "get_" + name))(date) for date in tqdm(list(trading_days))
        )

        start = start or pd.to_datetime('now').strftime(r"%Y-%m-%d")
        stop = stop or pd.to_datetime('now').strftime(r"%Y-%m-%d")

        if isinstance(result[0], pd.Series):
            return pd.concat(result, axis=1).T.sort_index().loc[start:stop]
        elif isinstance(result[0], pd.DataFrame):
            return pd.concat(result, axis=0).sort_index().loc(axis=0)[:, start:stop]


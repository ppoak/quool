import re
import time
import random
import requests
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
            if self.fragments else 1
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
        if not self.fragments:
            return pd.DataFrame()
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
        price: pd.DataFrame,
        period: int = 1, 
        nonrealizable: pd.DataFrame = None,
    ):
        price = price.where(~nonrealizable.astype(bool), other=np.nan)
        future = price.shift(-1 - period) / price.shift(-1) - 1
        future = future.dropna(axis=0, how='all')
        future = future.replace([np.inf, -np.inf], np.nan)
        return future.squeeze()
    
    def filter_factor(
        self, 
        factor: pd.DataFrame,
        nonrealizable: pd.DataFrame = None,
    ):
        return factor.where(~nonrealizable.astype(bool), other=np.nan).dropna(how='all',axis=1)

    def industry_inforcoef(
        self, 
        ind: pd.DataFrame,
        factor: pd.DataFrame,
        future: pd.DataFrame,
        method: str = 'pearson',
    ):
        rename_map = {
            '交通运输': 'Transport',
            '传媒': 'Media',
            '农林牧渔': 'Agriculture',
            '医药': 'Medicine',
            '商贸零售': 'Retail',
            '国防军工': 'Defense',
            '基础化工': 'Chemicals',
            '家电': 'Appliances',
            '建材': 'Materials',
            '建筑': 'Construction',
            '房地产': 'RealEstate',
            '有色金属': 'NonferrousMetal',
            '机械': 'Machinery',
            '汽车': 'Auto',
            '消费者服务': 'Services',
            '煤炭': 'Coal',
            '电力及公用事业': 'Power&Utilities',
            '电力设备及新能源': 'PowerEquip&NewEnergy',
            '电子': 'Electronics',
            '石油石化': 'Petroleum',
            '纺织服装': 'Textile',
            '综合': 'Comprehensive',
            '计算机': 'IT',
            '轻工制造': 'Manufacturing',
            '通信': 'Communication',
            '钢铁': 'Steel',
            '银行': 'Bank',
            '综合金融': 'Finance',
            '非银行金融': 'NonBankFinance',
            '食品饮料': 'Food'
        }
        df = pd.concat([factor.stack(), future.stack(), ind.stack()], axis=1)
        df.columns = ['factor', 'future', 'industry']
        ind_inforcoef = df.groupby(['industry']).apply(lambda x: x['future'].unstack().corrwith(x['factor'].unstack(), method=method).mean())
        return ind_inforcoef.rename(index=rename_map)

    def style_exposure(
        self, 
        barra: pd.DataFrame,
        weight: pd.DataFrame,
        top_group: pd.DataFrame,
    ):
        style_exposure = barra.apply(lambda x: (x.unstack() * weight).sum(axis=1)).mean()
        def _exposure(df: pd.DataFrame):
            df = df.where(top_group.notna())
            _weight = (df/df).div(df.count(axis=1),axis=0)
            return (df * _weight).sum(axis=1).mean()
        factor_exposure = barra.apply(lambda x: _exposure(x.unstack()))
        return factor_exposure - style_exposure
    
    def _prepare_factor(
        self, 
        factor: str | pd.DataFrame | pd.Series,
        start: str | pd.Timestamp = None,
        stop: str | pd.Timestamp = None,
        processor: list = None,
        universe: str = None,
    ):
        if isinstance(factor, pd.Series) and factor.index.nlevels == 2:
            factor = factor.unstack(self._code_level)

        elif isinstance(factor, pd.DataFrame):
            factor = factor.loc[start:stop]

        elif isinstance(factor, str):
            factor = self.read(factor, start=start, stop=stop)
        
        else:
            ValueError("Invalid factor type")

        if universe:
            factor = self.filter_factor(factor, universe=universe)

        processor = processor or []
        for proc in processor:
            kwargs = {}
            if isinstance(proc, tuple):
                proc, kwargs = proc
            factor = proc(factor, **kwargs)
        return factor
    
    def perform_crosssection(
        self, 
        factor: str | pd.DataFrame, 
        *,
        date: str | pd.Timestamp | pd.DataFrame | pd.Series,
        processor: list = None,
        period: int = 1,
        ptype: str = "volume_weighted_price",
        universe: str = '000985.XSHG',
        image: str | bool = True, 
        result: str = None
    ):
        future = self.get_future(ptype=ptype, period=period, start=date, stop=date, universe=universe)
        factor = self._prepare_factor(factor, future.name, future.name, processor, universe)
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
        method: str = 'pearson', #spearman, pearson, weighted
        universe: str = '000985.XSHG',
        industry: bool = False, 
        image: str | bool = True, 
    ):
        future = self.get_future(ptype, period, start, stop, universe)
        factor = self._prepare_factor(factor, start=start, stop=stop, processor=processor, universe=universe)

        if method =='weighted':
            def calculate_weighted_ic(x, r):
                x_ranked = x.rank(ascending=False)
                n = len(x)
                a = -np.log(0.5) / (n / 2 - 1)
                w = np.exp(-a * (x_ranked - 1))
                w /= w.sum()

                wx = w * x
                wr = w * r
                wxr = w * x * r
                numerator = wxr.sum() - (wx.sum() * wr.sum())
                denominator = np.sqrt((w * (x ** 2)).sum() - wx.sum()**2) * np.sqrt((w * (r ** 2)).sum() - wr.sum()**2)
                return numerator / denominator if denominator != 0 else np.nan
            inforcoef = factor.apply(lambda row: calculate_weighted_ic(row.dropna(), future.loc[row.name, row.dropna().index]), axis=1)
        else:
            inforcoef = factor.corrwith(future, axis=1, method=method).dropna()
        inforcoef.name = f"infocoef"

        ind_inforcoef = None
        if industry:
            ind_inforcoef = self.industry_inforcoef(start=start, stop=stop, factor=factor, future=future, method=method)

        if image:
            # 第一张图
            fig1, ax1 = plt.subplots(1, 1, figsize=(20, 10))
            inforcoef.plot(ax=ax1, label='infor-coef', alpha=0.7, title='Information Coef')
            inforcoef.rolling(rolling).mean().plot(linestyle='--', ax=ax1, label='trend')
            inforcoef.cumsum().plot(linestyle='-.', secondary_y=True, ax=ax1, label='cumm-infor-coef')
            pd.Series(np.zeros(inforcoef.shape[0]), index=inforcoef.index).plot(color='grey', ax=ax1, alpha=0.5)
            ax1.legend()
            fig1.tight_layout()

            if not isinstance(image, bool):
                fig1.savefig(image + '_inforcoef.png')
            else:
                fig1.show()
            # 第二张图：行业信息系数
            if ind_inforcoef is not None:
                fig2, ax2 = plt.subplots(figsize=(20, 10))
                bars = ind_inforcoef.plot(kind='bar', ax=ax2, color='#A52A2A', alpha=0.7, width=0.4)
                ax2.set_facecolor('#EEDFCC')
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                ax2.grid(linewidth=1.2, color='white', alpha=0.7)
                ax2.set_title('IC-Industry Distribution - P_1', loc='left', fontsize=16)
                ax2.set_xlabel('')
                for bar in bars.patches:
                    bar.set_x(bar.get_x() + 0.5)
                ax2.yaxis.set_minor_locator(plt.MaxNLocator(50))
                plt.xticks(rotation=45, ha='right', fontsize=12) 
                fig2.tight_layout()

                if not isinstance(image, bool):
                    fig2.savefig(image + '_industry.png')
                else:
                    fig2.show()
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
        commission: float = 0.0005, 
        universe: str = '000985.XSHG',
        benchmark: pd.Series = None,
        n_jobs: int = 1,
        image: str | bool = True, 
        result: str = None
    ):
        future = self.get_future(ptype, 1, start, stop, universe)
        factor = self._prepare_factor(factor, start=start, stop=stop, processor=processor, universe=universe)
        
        # ngroup test
        try: 
            groups = factor.apply(lambda x: pd.qcut(x, q=ngroup, labels=False), axis=1) + 1
        except:
            groups = factor.apply(lambda x: pd.qcut(x.rank(method='first', ascending=True), q=ngroup, labels=False), axis=1) + 1
            # for date in factor.index:
            #     try:
            #         pd.qcut(factor.loc[date], q=ngroup, labels=False)
            #     except:
            #         raise ValueError(f"on date {date}, grouping failed")
        
        def _grouping(x):
            group = groups.where(groups == x)
            weight = (group / group).div(group.count(axis=1), axis=0).fillna(0) # 等权
            weight = weight[::period]
            turnover = weight.diff(periods=1).fillna(0).abs().sum(axis=1) / 2 
            turnover = turnover.reindex(future.index).ffill() / period
            ret = (weight.reindex(future.index).ffill() * future).sum(axis=1).shift(1).fillna(0)
            ret -= commission * turnover
            val = (1 + ret).cumprod()
            evaluation, _ = evaluate(val, turnover=turnover, benchmark=benchmark, image=False)

            return {
                'evaluation': evaluation,
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
        longshort_evaluation, _ = evaluate(longshort_value, image=False)
        
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
        commission: float = 0.0005, 
        universe: str = '000985.XSHG',
        benchmark: pd.Series = None,
        image: str | bool = True, 
        result: str = None
    ):  
        future = self.get_future(ptype, 1, start, stop, universe)
        factor = self._prepare_factor(factor, start, stop, processor, universe)
        
        topks = factor.rank(ascending=False, axis=1) < topk
        topks = factor.where(topks)
        topks = (topks/topks).div(topks.count(axis=1), axis=0).fillna(0)
        topks = topks[::period]

        turnover = topks.diff(periods=1).fillna(0).abs().sum(axis=1) / 2
        turnover = turnover.reindex(future.index).ffill() / period
        ret = (topks.reindex(future.index).ffill() * future).sum(axis=1).shift(1).fillna(0)
        ret -= commission * turnover
        val = (1 + ret).cumprod()
        eva, _ = evaluate(val, turnover=turnover, benchmark=benchmark, image=False)

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
    
    def perform_optimizer(
        self, 
        factor: pd.DataFrame | list,
        *,
        start: str = None,
        stop: str = None,
        correlation: str | bool = False, # 'ic', 'factor'
        corr_method: str = 'pearson',
        ic_method: str = 'pearson', # 'spearman', 'pearson', 'weighted'
        heatmap_image: str | bool = None,
        tscorr_image: str | bool = None, 
        orthogonalization : bool = False, 
        period: int = 1,
        ptype: str = "volume_weighted_price",
        universe: str = '000985.XSHG',
        processor: list = None,
        compound_method: str = None,  # 'equal_weighted', 'ic_weighted', 'ir_weighted', 'sharpe_weighted', 'optimize_ir', 'optimize_ic'
        commission: float = 0.0005, 
        objective: str = None,
        cons: list = None,
    )-> dict:
        if isinstance(factor, pd.DataFrame):
            start, stop = factor.index.get_level_values(self._date_level).unique()[[0, -1]]
        else:
            factor = self.read(factor, start=start, stop=stop)

        _future = self.get_future(ptype=ptype, period=1, start=start, stop=stop, universe=universe)
        factor = factor.apply(lambda x: self._prepare_factor(x.unstack(self._code_level), start, stop, processor, universe).unstack())
        
        if compound_method=='optimize_ic':
            correlation = 'factor'
        if correlation:
            future = self.get_future(ptype=ptype, period=period, start=start, stop=stop, universe=universe)

        def plot_heatmap(correlation_matrix):
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='RdBu', center=0, vmin=-1, vmax=1, cbar=True, ax=ax)
            ax.set_title('Heatmap for Correlation')
            if not isinstance(heatmap_image, bool):
                plt.savefig(heatmap_image)
                plt.close(fig)
            else:
                fig.show()

        def prepare_correlation_ts(correlations):
            correlation_ts = pd.concat(
                [matrix.unstack().rename_axis(['Feature1', 'Feature2']).reset_index().assign(Date=date) 
                for date, matrix in correlations.items() if matrix is not None], 
                axis=0
            )
            correlation_ts = correlation_ts[correlation_ts['Feature1'] != correlation_ts['Feature2']]
            correlation_ts = correlation_ts.rename(columns={0: 'Correlation'})
            correlation_ts['sorted_features'] = correlation_ts.apply(lambda x: tuple(sorted([x['Feature1'], x['Feature2']])), axis=1)
            correlation_ts = correlation_ts.drop_duplicates(subset=['Date', 'sorted_features'])
            correlation_ts.set_index(['Date', 'sorted_features'], inplace=True)
            return correlation_ts['Correlation'].unstack()
        
        def plot_ts_correlation(correlation_ts):
            num_pairs = len(correlation_ts.columns)
            num_cols = 3  
            num_rows = (num_pairs + num_cols - 1) // num_cols  

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 5))
            axes = axes.flatten()
            
            for ax, ((feature1, feature2), series) in zip(axes, correlation_ts.items()):
                ax.plot(series.index, series, marker='', linestyle='-')
                ax.set_title(f'{feature1} and {feature2}')
                ax.grid(True)
                ax.tick_params(axis='x', which='major', labelsize=10)
                for label in ax.get_xticklabels():
                    label.set_rotation(45)

            for i in range(num_pairs, len(axes)):
                axes[i].set_visible(False)
            fig.supxlabel('Date', fontsize=12) 
            fig.supylabel('Correlation', fontsize=12)
            fig.tight_layout()
            if not isinstance(tscorr_image, bool):
                plt.savefig(tscorr_image)
                plt.close(fig)
            else:
                fig.show()

        def orthogonalize(group):
            X = group.values
            C = np.cov(X.T)
            D, U = np.linalg.eigh(C)
            epsilon = 1e-11
            D_sqrt_inv = np.diag(1 / np.sqrt(D + epsilon))
            S = U @ D_sqrt_inv @ U.T
            F_hat = X @ S
            return pd.DataFrame(F_hat, index=group.index, columns=group.columns)
        
        def get_metrics(factor):
            if ic_method =='weighted':
                def calculate_weighted_ic(x, r):
                    x_ranked = x.rank(ascending=False)
                    n = len(x)
                    a = -np.log(0.5) / (n / 2 - 1)
                    w = np.exp(-a * (x_ranked - 1))
                    w /= w.sum()

                    wx = w * x
                    wr = w * r
                    wxr = w * x * r
                    numerator = wxr.sum() - (wx.sum() * wr.sum())
                    denominator = np.sqrt((w * (x ** 2)).sum() - wx.sum()**2) * np.sqrt((w * (r ** 2)).sum() - wr.sum()**2)
                    return numerator / denominator if denominator != 0 else np.nan
                inforcoef = factor.apply(lambda row: calculate_weighted_ic(row.dropna(), future.loc[row.name, row.dropna().index]), axis=1)
            else:
                inforcoef = factor.corrwith(future, axis=1, method=ic_method).dropna()

            _adj = 1
            if inforcoef.mean() < 0:
                _adj = -1
                factor = factor * _adj
                inforcoef = inforcoef * _adj
                
            ic_mean = round(inforcoef.mean(), 4)
            ic_std = round(inforcoef.std(), 4)

            topks = int(factor.shape[1] * 0.1)
            topks = factor.rank(ascending=False, axis=1) <= topks
            topks = factor.where(topks)
            topks = (topks / topks).div(topks.count(axis=1), axis=0).fillna(0)
            topks = topks[::period]

            turnover = topks.diff(periods=1).fillna(0).abs().sum(axis=1) / 2 
            turnover = turnover.reindex(_future.index).ffill() / period
            ret = (topks.reindex(_future.index).ffill() * _future).sum(axis=1).shift(1).fillna(0)
            ret -= commission * turnover
            val = (1 + ret).cumprod()
        
            net_val = val / val.iloc[0]
            returns = val.pct_change(fill_method=None).fillna(0)
            total_return = (net_val.iloc[-1] / net_val.iloc[0] - 1) * 100
            annual_return = ((total_return / 100 + 1) ** (365 / (val.index.max() - val.index.min()).days) - 1) * 100
            annual_volatility = (returns.std() * np.sqrt(252)) * 100
            annual_sharpe = annual_return / annual_volatility if annual_volatility != 0 else np.nan
    
            metrics = pd.Series({
                'IC mean': ic_mean,
                'IR': round(ic_mean /ic_std, 4),
                'Top Annual Return': round(total_return, 4),
                'Top Annual sharpe': round(annual_sharpe, 4),
                'Daily Turnover': round(turnover.mean()*100, 4),
            })
            return inforcoef, metrics, _adj
        
        def optimize_weights(method):
            if method == 'optimize_ir':
                cov_val = np.array(result['inforcoef'].cov())
            elif method == 'optimize_ic':
                _adj_matrix = np.diag(metrics.apply(lambda x: x[2]))
                _adj_correlation_matrix = _adj_matrix @ factor_correlation_matrix.values @ _adj_matrix
                cov_val = np.array(_adj_correlation_matrix)  # 因子标准化后的协方差矩阵=相关性矩阵
            inv_cov = np.linalg.inv(cov_val)
            ic_vector = np.mat(result['inforcoef'].mean())
            w = inv_cov * ic_vector.T
            w = pd.Series(np.array(w/w.sum()).flatten(), index=result['inforcoef'].columns)
            return w
        
        result = {'factor': factor}

        if orthogonalization:
            factor = factor.groupby(level='date', as_index=False).apply(lambda x: orthogonalize(x.fillna(0))).reset_index(level=0, drop=True).sort_index()
            result['symmetric_orthogonal'] = factor

        if correlation =='factor' or correlation == True:
            correlations = {date: group.corr(method=corr_method) for date, group in factor.groupby('date') 
                if not group.fillna(0).corr(method=corr_method).isnull().values.any()
            }
            factor_correlation_matrix = sum(correlations.values())/ len(correlations)
            result['factor_correlation_matrix'] = factor_correlation_matrix
            if heatmap_image:
                plot_heatmap(factor_correlation_matrix)

            if tscorr_image:
                correlation_ts = prepare_correlation_ts(correlations)
                plot_ts_correlation(correlation_ts)

        if compound_method or correlation == 'ic':
            metrics = factor.apply(lambda x: get_metrics(x.unstack(self._code_level)))
            result['inforcoef'] = metrics.apply(lambda x: x[0])
            result['metrics'] = metrics.apply(lambda x: x[1]).T
            factor = factor * metrics.apply(lambda x: x[2])

            if correlation =='ic':
                ic_correlation_matrix = result['inforcoef'].corr(method=corr_method)
                result['ic_correlation_matrix'] = ic_correlation_matrix
                if heatmap_image:
                    plot_heatmap(ic_correlation_matrix)

            if compound_method == 'ic_weighted':
                compound_weight = result['metrics']['IC mean'].abs() / result['metrics']['IC mean'].abs().sum()
            elif compound_method == 'ir_weighted':
                compound_weight = result['metrics']['IR'].abs() / result['metrics']['IR'].abs().sum()
            elif compound_method == 'sharpe_weighted':
                compound_weight = result['metrics']['Top Annual sharpe'] / result['metrics']['Top Annual sharpe'].sum()
            elif compound_method == 'equal_weighted':
                compound_weight = pd.Series(1, index=result['metrics'].index) / len(result['metrics'].index)
            elif compound_method == 'optimize_ir':
                compound_weight = optimize_weights('optimize_ir')
            elif compound_method == 'optimize_ic':
                compound_weight = optimize_weights('optimize_ic')
            else:
                compound_weight = None

            if compound_weight is not None:
                result['compound_factor'] = (factor * compound_weight).sum(axis=1).to_frame(name='compound_factor')
                res = get_metrics(result['compound_factor'].unstack(self._code_level))[1].to_frame(name='compound_factor').T
                result['metrics'] = pd.concat([result['metrics'], res])

        return result 

    def factor_analysis(
        self,
        factor: str | pd.DataFrame,
        *,
        period: int = 5,
        start: str = None,
        stop: str = None,
        ptype: str = "head_weighted_price",
        processor: list = None,
        rolling: int = 20, 
        method: str = 'spearman',
        universe: str = '000985.XSHG',
        industry_ic: bool = False, 
        style_exposure: bool = False, 
        n_jobs: int = 1,
        benchmark: pd.Series = None,
        commission: float = 0.0005, 
        ngroup: int = 10, 
        image: str | bool = True, 
    ):
        future = self.get_future(ptype, period, start, stop, universe)
        _future = self.get_future(ptype, 1, start, stop, universe)
        factor = self._prepare_factor(factor, start=start, stop=stop, processor=processor, universe=universe)
        crosssection = pd.concat([factor.loc[factor.index[0]], future.loc[future.index[0]]], axis=1, keys=["Factor", future.index[0]])

        if benchmark is None:
            weight = (factor / factor).div(factor.count(axis=1), axis=0).fillna(0)
            weight = weight[::period]
            turnover = weight.diff(periods=1).fillna(0).abs().sum(axis=1) / 2 
            turnover = turnover.reindex(_future.index).ffill() / period
            ret = (weight.reindex(_future.index).ffill() * _future).sum(axis=1).shift(1).fillna(0)
            ret -= commission * turnover
            benchmark = (1 + ret).cumprod()

        if method =='weighted':
            def calculate_weighted_ic(x, r):
                x_ranked = x.rank(ascending=False)
                n = len(x)
                a = -np.log(0.5) / (n / 2 - 1)
                w = np.exp(-a * (x_ranked - 1))
                w /= w.sum()

                wx = w * x
                wr = w * r
                wxr = w * x * r
                numerator = wxr.sum() - (wx.sum() * wr.sum())
                denominator = np.sqrt((w * (x ** 2)).sum() - wx.sum()**2) * np.sqrt((w * (r ** 2)).sum() - wr.sum()**2)
                return numerator / denominator if denominator != 0 else np.nan
            inforcoef = factor.apply(lambda row: calculate_weighted_ic(row.dropna(), future.loc[row.name, row.dropna().index]), axis=1)
        else:
            inforcoef = factor.corrwith(future, axis=1, method=method).dropna()
        inforcoef.name = f"infocoef"

        if industry_ic:
            ind_inforcoef = self.industry_inforcoef(start=start, stop=stop, factor=factor, future=future, method=method)

        try: 
            groups = factor.apply(lambda x: pd.qcut(x, q=ngroup, labels=False), axis=1) + 1
        except:
            groups = factor.apply(lambda x: pd.qcut(x.rank(method='first', ascending=True), q=ngroup, labels=False), axis=1) + 1

        def _grouping(x):
            group = groups.where(groups == x)
            weight = (group / group).div(group.count(axis=1), axis=0).fillna(0) # 等权
            weight = weight[::period]
            turnover = weight.diff(periods=1).fillna(0).abs().sum(axis=1) / 2 
            turnover = turnover.reindex(_future.index).ffill() / period
            ret = (weight.reindex(_future.index).ffill() * _future).sum(axis=1).shift(1).fillna(0)
            ret -= commission * turnover
            val = (1 + ret).cumprod()
            evaluation, exvalue = evaluate(val, turnover=turnover, benchmark=benchmark, image=False)
            return {
                'evaluation': evaluation, 'exvalue': exvalue,
                'turnover': turnover, 'value': val,
            }
        
        ngroup_result = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(_grouping)(i) for i in range(1, ngroup + 1))
        ngroup_evaluation = pd.concat([res['evaluation'] for res in ngroup_result], 
            axis=1, keys=range(1, ngroup + 1)).add_prefix('group')
        ngroup_value = pd.concat([res['value'] for res in ngroup_result], 
            axis=1, keys=range(1, ngroup + 1)).add_prefix('group')
        ngroup_exvalue = pd.concat([res['exvalue'] for res in ngroup_result], 
            axis=1, keys=range(1, ngroup + 1)).add_prefix('group')
        ngroup_turnover = pd.concat([res['turnover'] for res in ngroup_result], 
            axis=1, keys=range(1, ngroup + 1)).add_prefix('group')
        ngroup_returns = ngroup_value.pct_change().fillna(0)
        longshort_returns = ngroup_returns[f"group{ngroup}"] - ngroup_returns["group1"]
        longshort_value = (longshort_returns + 1).cumprod()
        longshort_evaluation, _ = evaluate(longshort_value, image=False)
        
        # naming
        longshort_evaluation.name = "longshort"
        longshort_value.name = "longshort value"
        group = pd.concat([ngroup_evaluation, longshort_evaluation], axis=1)

        year = ngroup_exvalue.apply(lambda x: x.resample('YE').last()/x.resample('YE').first() -1).T

        if style_exposure:
            x = 1 if inforcoef.mean() < 0 else 10
            top_group = groups.where(groups == x)
            exposure = self.style_exposure(start=start, stop=stop, top_group=top_group, universe=universe)

        if image:
            fig1, ax1 = plt.subplots(1, 1, figsize=(20, 10))
            inforcoef.plot(ax=ax1, label='infor-coef', alpha=0.7, title='Information Coef')
            inforcoef.rolling(rolling).mean().plot(linestyle='--', ax=ax1, label='trend')
            inforcoef.cumsum().plot(linestyle='-.', secondary_y=True, ax=ax1, label='cumm-infor-coef')
            pd.Series(np.zeros(inforcoef.shape[0]), index=inforcoef.index).plot(color='grey', ax=ax1, alpha=0.5)
            ax1.legend()
            fig1.tight_layout()

            if not isinstance(image, bool):
                fig1.savefig(image + '_inforcoef.png')
            else:
                fig1.show()

            if industry_ic:
                fig2, ax2 = plt.subplots(figsize=(20, 10))
                bars = ind_inforcoef.plot(kind='bar', ax=ax2, color='#A52A2A', alpha=0.7, width=0.4)
                ax2.set_facecolor('#EEDFCC')
                ax2.spines['top'].set_visible(False)
                ax2.spines['right'].set_visible(False)
                ax2.grid(linewidth=1.2, color='white', alpha=0.7)
                ax2.set_title('IC-Industry Distribution - P_1', loc='left', fontsize=16)
                ax2.set_xlabel('')
                for bar in bars.patches:
                    bar.set_x(bar.get_x() + 0.5)
                ax2.yaxis.set_minor_locator(plt.MaxNLocator(50))
                plt.xticks(rotation=45, ha='right', fontsize=12) 
                fig2.tight_layout()

                if not isinstance(image, bool):
                    fig2.savefig(image + '_industry.png')
                else:
                    fig2.show()

            fig3, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
            longshort_value.plot(ax=ax3, linestyle='--')
            ngroup_value.plot(ax=ax3, alpha=0.8)
            ngroup_turnover.plot(ax=ax3, secondary_y=True, alpha=0.2)
            fig3.tight_layout()
            if isinstance(image, (str, Path)):
                fig3.savefig(image + f'_group{ngroup}.png')
            else:
                fig3.show()  

            index = np.arange(len(year.index))
            bar_width = 0.08
            group_gap = 0.05 
            bar_position_shift = (bar_width + group_gap / len(year.columns))
            fig4, ax4 = plt.subplots(figsize=(24, 10))
            for i, year_label in enumerate(year.columns):
                ax4.bar(index + i * bar_position_shift, year[year_label], bar_width, alpha=0.8, label=year_label.strftime('%Y'))
            ax4.set_title('Annualized EX-Returns by Group', fontsize=20) 
            ax4.set_xticks(index + bar_position_shift * (len(year.columns) / 2))
            ax4.set_xticklabels(year.index, rotation=0, fontsize=14)
            ax4.yaxis.set_tick_params(labelsize=14)
            ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
            ax4.grid(True, axis='y', linestyle='--', linewidth=0.5)
            ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=len(year.columns), fontsize=12, title_fontsize='13')
            ax4.spines[['top', 'right', 'left', 'bottom']].set_visible(False)
            ax4.tick_params(axis='both', which='both', length=0)
            fig4.tight_layout()
            if isinstance(image, (str, Path)):
                fig4.savefig(f'{image}_Returns.png')
            else:
                fig4.show()

            if style_exposure:
                fig5, ax5 = plt.subplots(figsize=(20, 10))
                exposure.plot(kind='bar', ax=ax5, color='steelblue', alpha=0.8)
                ax5.grid(True, which='both', linestyle='--', linewidth=0.5)
                ax5.axhline(0, color='black', linewidth=0.8)  # 添加 y=0 的线
                plt.xticks(rotation=45, ha='right', fontsize=12) 
                fig5.tight_layout()

                if isinstance(image, (str, Path)):
                    fig5.savefig(f'{image}_exposure.png')
                else:
                    fig5.show()

            pd.plotting.scatter_matrix(crosssection, figsize=(20, 10), hist_kwds={'bins': 100}, alpha=0.8)
            plt.tight_layout(pad=2.0)
            
            if isinstance(image, (str, Path)):
                plt.savefig(f'{image}_crosssection.png')
            else:
                plt.show()

        print({
                'IC mean': round(inforcoef.mean(), 4),
                'IR': round(inforcoef.mean() /inforcoef.std(), 4),
                'ABS_IC>2%': round(len(inforcoef[abs(inforcoef) > 0.02].dropna())/len(inforcoef), 4),
            })
        return group

    def get(self, name: str, trading_days: pd.DatetimeIndex, n_jobs: int = -1, start: str = None, stop: str = None):
        result = Parallel(n_jobs=n_jobs, backend='loky')(
            delayed(getattr(self, "get_" + name))(date) for date in tqdm(list(trading_days))
        )
        if isinstance(result[0], pd.Series):
            return pd.concat(result, axis=1).T.sort_index().loc[start:stop]
        elif isinstance(result[0], pd.DataFrame):
            return pd.concat(result, axis=0).sort_index().loc(axis=0)[:, start:stop]




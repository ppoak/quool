import re
import abc
import time
import random
import requests
import itertools
import numpy as np
import pandas as pd
from lxml import etree
from pathlib import Path
from .util import parse_commastr, evaluate


class Table(abc.ABC):

    def __init__(
        self,
        uri: str | Path,
        create: bool = False,
    ):
        self.path = Path(uri).expanduser().resolve()
        self.name = self.path.stem
        if create:
            self.path.mkdir(parents=True, exist_ok=True)
        elif not self.path.exists():
            raise NotADirectoryError(f"{self.path} does not exist.")
    
    @property
    def minfrag(self):
        if self.fragments:
            return sorted(list(self.path.glob('**/*.parquet')), key=lambda f: f.stat().st_size)[0]
    
    @property
    def spliter(self):
        return lambda _: 1
    
    @property
    def namer(self):
        return lambda _: self.name
    
    @property
    def fragments(self):
        return sorted([f.stem for f in list(self.path.glob('**/*.parquet'))])
    
    @property
    def columns(self):
        if self.fragments:
            return self._read_fragment(self.minfrag).columns
        else:
            return pd.Index([])
    
    @property
    def dtypes(self):
        if self.fragments:
            return self._read_fragment(self.minfrag).dtypes
        else:
            return pd.Series([])
    
    @property
    def dimshape(self):
        if self.fragments:
            return (self._read_fragment(self.minfrag)).dimshape
        return None
    
    def get_levelname(self, level: int | str) -> int | str:
        minfrag = self.minfrag
        if isinstance(level, int) and minfrag:
            return self._read_fragment(minfrag).index.names[level] or level
        return level

    def _fragment_path(self, fragment: str):
        return (self.path / fragment).with_suffix('.parquet')
    
    def _related_frag(self, df: pd.DataFrame | pd.Series):
        # in case of empty dataframe
        frags = df.groupby(self.spliter).apply(
            lambda x: np.nan if x.empty else self.namer(x)
        ).dropna()
        if frags.empty:
            return []
        return frags.unique().tolist()
    
    def _update_frag(self, frag: pd.DataFrame):
        # in case of empty dataframe
        if frag.empty:
            return
        
        name = self.namer(frag)
        if self.fragments and name in self.fragments:
            frag_dat = self._read_fragment(name)
            common_idx = frag.index.intersection(frag_dat.index)
            new_idx = frag.index.difference(frag_dat.index)
            upd_dat = frag.loc[common_idx].astype(self.dtypes.loc[frag.columns])
            new_dat = frag.loc[new_idx].astype(self.dtypes.loc[frag.columns])
            if not upd_dat.empty:
                frag_dat.loc[common_idx, frag.columns] = upd_dat
            if not (new_dat.empty or new_dat.isna().all().all()):
                frag_dat = pd.concat([frag_dat, 
                    new_dat.reindex(columns=frag_dat.columns).astype(self.dtypes)
                ], axis=0)
            frag_dat.sort_index().to_parquet(self._fragment_path(name))
        elif not self.fragments:
            frag.sort_index().to_parquet(self._fragment_path(name))
        else:
            frag.reindex(columns=self.columns).astype(
                self.dtypes).sort_index().to_parquet(self._fragment_path(name))
    
    def _read_fragment(
        self,
        fragment: list | str = None,
    ):
        fragment = fragment or self.fragments
        fragment = [fragment] if not isinstance(fragment, list) else fragment
        fragment = [self._fragment_path(frag) for frag in fragment]
        return pd.read_parquet(fragment, engine='pyarrow')
    
    def read(
        self,
        columns: str | list[str] | None = None,
        filters: list[list[tuple]] = None,
    ) -> pd.DataFrame | pd.Series:
        df = pd.read_parquet(
            self.path, 
            engine = 'pyarrow', 
            columns = parse_commastr(columns),
            filters = filters,
        )
        return df
    
    def update(
        self,
        df: pd.DataFrame | pd.Series,
    ):
        if isinstance(df, pd.Series):
            df = df.to_frame()
        
        if not df.columns.difference(self.columns).empty and self.fragments:
            raise ValueError("new field found, please add first")

        df.groupby(self.spliter).apply(self._update_frag)

    def add(self, column: dict):
        if not self.columns.intersection(column.keys()).empty:
            raise ValueError("existing field found, please update it")
        
        for frag in self.fragments:
            d = self._read_fragment(frag)
            d[list(column.keys())] = np.nan
            d = d.astype(column)
            d.sort_index().to_parquet(self._fragment_path(frag))
    
    def delete(self, index: pd.Index):
        related_fragment = self._related_frag(pd.Series(np.ones(len(index)), index=index))
        for frag in related_fragment:
            df = self._read_fragment(frag)
            df = df.drop(index=index.intersection(df.index))
            df.sort_index().to_parquet(self._fragment_path(frag))

    def remove(self, fragment: str | list = None):
        fragment = fragment or self.fragments
        fragment = [fragment] if not isinstance(fragment, list) else fragment
        for frag in fragment:
            ((self.path / frag).with_suffix('.parquet')).unlink()
    
    def sub(self, column: str | list):
        column = parse_commastr(column)
        for frag in self.fragments:
            df = self._read_fragment(frag)
            df = df.drop(column, axis=1)
            df.sort_index().to_parquet(self._fragment_path(frag))

    def rename(self, column: dict):
        column = parse_commastr(column)
        for frag in self.fragments:
            df = self._read_fragment(frag)
            df = df.rename(columns=column)
            df.sort_index().to_parquet(self._fragment_path(frag))
    
    def __str__(self) -> str:
        return (f'Table at <{self.path.absolute()}>\n' + 
                (f'\tfragments: <{self.fragments[0]}> - <{self.fragments[-1]}>\n' 
                 if self.fragments else '\tfragments: EMPTY\n') + 
                (f'\tcolumns: <{self.columns.to_list()}>\n' 
                 if self.fragments else '\tcolumns: EMPTY\n'))
    
    def __repr__(self) -> str:
        return self.__str__()


class ItemTable(Table):

    def read(
        self,
        column: str | list = None,
        start: str | list = None,
        stop: str = None,
        datecol: str = None,
        filters: list[list[tuple]] = None,
    ):
        filters = filters or []
        if datecol is None:
            datecol = self.get_levelname(0)
            datecol = '__index_level_0__' if isinstance(datecol, int) else datecol

        if isinstance(start, (list, pd.DatetimeIndex)):
            filters.append((datecol, "in", pd.to_datetime(start)))
        elif isinstance(start, (str, pd.Timestamp)):
            filters.append((datecol, ">=", pd.to_datetime(start)))
        if isinstance(stop, str) and not isinstance(start, list):
            filters.append((datecol, "<=", pd.to_datetime(stop)))
        
        if not len(filters):
            filters = None
        
        return super().read(parse_commastr(column), filters)


class DatetimeTable(Table):

    def __init__(
        self, 
        uri: str | Path,
        freq: str = 'M',
        format: str = '%Y%m',
        create: bool = False,
    ):
        self._freq = freq
        self._format = format
        super().__init__(uri, create)
    
    @property
    def spliter(self):
        return pd.Grouper(level=0, freq=self._freq, sort=True)
    
    @property
    def namer(self):
        return lambda x: x.index[0].strftime(self._format)
    
    def get_levelname(self) -> int | str:
        return super().get_levelname(0)
    
    def read(
        self, 
        field: str | list = None,
        start: str | list = None,
        stop: str = None,
        filters: list[list[tuple]] = None,
    ) -> pd.Series | pd.DataFrame:
        index_name = self.get_levelname()
        field = parse_commastr(field)
        filters = filters or []
        start = pd.to_datetime(start or "20000104")
        stop = pd.to_datetime(stop or 'now')

        if not isinstance(start, pd.DatetimeIndex):
            filters += [
                (index_name, ">=", start),
                (index_name, "<=", stop), 
            ]
            return super().read(field, filters)
        
        else:
            filters += [(index_name, "in", start)]
            return super().read(field, filters)


class PanelTable(Table):
    
    def __init__(
        self,
        uri: str | Path,
        code_level: str | int = 0,
        date_level: str | int = 1,
        freq: str = "ME",
        format: str = r"%Y%m",
        create: bool = False,
    ):
        self._code_level = code_level
        self._date_level = date_level
        self._freq = freq
        self._format = format
        super().__init__(uri, create)
    
    @property
    def spliter(self):
        return pd.Grouper(level=self.get_levelname(self._date_level), freq=self._freq, sort=True)
    
    @property
    def namer(self):
        return lambda x: x.index.get_level_values(self.get_levelname(self._date_level))[0].strftime(self._format)
        
    def read(
        self, 
        field: str | list = None,
        code: str | list = None,
        start: str | list = None,
        stop: str = None,
        filters: list[list[tuple]] = None,
    ) -> pd.Series | pd.DataFrame:
        date_level = self.get_levelname(self._date_level)
        code_level = self.get_levelname(self._code_level)
        date_level = f'__index_level_{date_level}__' if isinstance(date_level, int) else date_level
        code_level = f'__index_level_{code_level}__' if isinstance(code_level, int) else code_level
        
        code = parse_commastr(code)
        field = parse_commastr(field)
        filters = filters or []
        start = pd.to_datetime(start if start is not None else "20000104")
        stop = pd.to_datetime(stop or 'now')

        if not isinstance(start, pd.DatetimeIndex):
            filters += [
                (date_level, ">=", start),
                (date_level, "<=", stop), 
            ]
            if code is not None:
                filters.append((code_level, "in", code))
            return super().read(field, filters)
        
        else:
            filters += [(date_level, "in", start)]
            if code is not None:
                filters.append((code_level, "in", code))
            return super().read(field, filters)
        
    def __str__(self) -> str:
        return super().__str__() + (f'\tindex: '
            f'<code {self.get_levelname(self._code_level)}> <date {self.get_levelname(self._date_level)}>')


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

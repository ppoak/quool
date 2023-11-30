import re
import time
import random
import datetime
import requests
import numpy as np
import pandas as pd
import akshare as ak
from tqdm import tqdm
from math import ceil
from lxml import etree
from bs4 import BeautifulSoup
from urllib.parse import quote
from joblib import Parallel, delayed
from .tools import strip_stock_code, format_code


class Request:

    ua = [
        'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/30.0.1599.101',
        'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.122',
        'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.71',
        'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95',
        'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.71',
        'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; QQDownload 732; .NET4.0C; .NET4.0E)',
        'Mozilla/5.0 (Windows NT 5.1; U; en; rv:1.8.1) Gecko/20061208 Firefox/2.0.0 Opera 9.50',
        'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:34.0) Gecko/20100101 Firefox/34.0',
    ]
    basic_headers = {
        'Accept': '*/*',
        'Connection': 'keep-alive',
        'Accept-Language': 'zh-CN,zh;q=0.8'
    }

    def __init__(
        self,
        url: str | list,
        method: str = 'get',
        headers: dict = None,
        proxies: list[dict] = None,
        timeout: float = None,
        retry: int = None,
        delay: float = 0,
        verbose: bool = False,
        **kwargs
    ) -> None:
        self.url = url if isinstance(url, list) else [url]
        self.method = method

        self.headers = headers or {}
        if not (self.headers.get('user-agent') and self.headers.get('User-Agent')):
            self.headers['User-Agent'] = random.choice(self.ua)
        if headers:
            self.headers.update(headers)

        self.proxies = proxies or [{}]
        self.timeout = timeout
        self.retry = retry
        self.delay = delay
        self.verbose = verbose
        self.kwargs = kwargs
    
    def _req(self, url: str) -> requests.Response:
        method = getattr(requests, self.method)
        retry = self.retry or 1

        for t in range(1, retry + 1):
            try:
                resp = method(
                    url, headers=self.headers, proxies=random.choice(self.proxies),
                    timeout=self.timeout, **self.kwargs
                )
                resp.raise_for_status()
                if self.verbose:
                    print(f'[+] {url} try {t}')
                return resp
            except Exception as e:
                if self.verbose:
                    print(f'[-] {e} {url} try {t}')
                time.sleep(self.delay)

        return None
    
    def request(self) -> list[requests.Response]:
        responses = []
        for url in self.url:
            resp = self._req(url)
            responses.append(resp)
        self.responses = responses
        return self
    
    def para_request(self) -> list[requests.Response]:
        self.responses = Parallel(n_jobs=-1, backend='loky')(
            delayed(self._req)(url) for url in self.url
        )
        return self

    def callback(self, *args, **kwargs):
        return self.responses

    @property
    def json(self):
        return [res.json() if res is not None else None for res in self.responses]

    @property
    def etree(self):
        return [etree.HTML(res.text) if res is not None else None for res in self.responses]
    
    @property
    def html(self):
        return [res.text  if res is not None else None for res in self.responses]
    
    @property
    def soup(self):
        return [BeautifulSoup(res.text, 'html.parser')  if res is not None else None for res in self.responses]

    def __call__(self, para: bool = True, *args, **kwargs):
        if para:
            self.para_request()
        else:
            self.request()
        return self.callback(*args, **kwargs)


class AkShare:
    TODAY = pd.to_datetime(datetime.datetime.today()).normalize()
    START = '20050101'
    
    @classmethod
    def market_daily(cls, code: str, start: str = None, end: str = None):
        """Get market daily prices for one specific stock
        
        code: str, the code of the stock
        start: str, start date in string format
        end: str, end date in string format
        """
        code = strip_stock_code(code)
        start = start or cls.START
        end = end or cls.TODAY.strftime('%Y%m%d')

        price = ak.stock_zh_a_hist(symbol=code, start_date=start, end_date=end, adjust='')
        if not price.empty:
            price = price.set_index('日期')
        else:
            return price
        adjprice = ak.stock_zh_a_hist(symbol=code, start_date=start, end_date=end, adjust='hfq')
        if not adjprice.empty:
            adjprice = adjprice.set_index('日期')
        else:
            return adjprice
        adjfactor = adjprice['收盘'] / price['收盘']
        adjfactor.name = 'adjfactor'
        price = pd.concat([price, adjfactor], axis=1)
        price = price.rename(columns = {
            "开盘": "open",
            "收盘": "close",
            "最高": "high",
            "最低": "low",
            "成交量": "volume",
            "成交额": "amount",
            "涨跌幅": "pctchange",
            "振幅": "vibration",
            "涨跌额": "change",
            "换手率": "turnover",
        }).astype('f')
        price.index = pd.to_datetime(price.index)
        price.index.name = 'datetime'

        return price

    @classmethod
    def stock_quote(cls, code_only: bool = False):
        """Get the realtime quote amoung the a stock share market

        code_only: bool, decide only return codes on the market
        """
        price = ak.stock_zh_a_spot_em()
        price = price.set_index('代码').drop('序号', axis=1)
        if code_only:
            return price.index.to_list()
        return price

    @classmethod
    def plate_quote(cls, name_only: bool = False):
        data = ak.stock_board_industry_name_em()
        data = data.set_index('板块名称')
        if name_only:
            return data.index.to_list()
        return data

    @classmethod
    def etf_market_daily(cls, code: str, start: str = None, end: str = None):
        code = strip_stock_code(code)
        start = start or cls.START
        end = end or cls.TODAY.strftime('%Y%m%d')
        price = ak.fund_etf_fund_info_em(code, start, end).set_index('净值日期')
        price.index = pd.to_datetime(price.index)
        return price
    
    @classmethod
    def stock_fund_flow(cls, code: str):
        code, market = code.split('.')
        if market.isdigit():
            code, market = market, code
        market = market.lower()
        funds = ak.stock_individual_fund_flow(stock=code, market=market)
        funds = funds.set_index('日期')
        funds.index = pd.MultiIndex.from_product([[code], 
            pd.to_datetime(funds.index)], names=['日期', '代码'])
        return funds
    
    @classmethod
    def stock_fund_rank(cls):
        datas = []
        for indi in ['今日', '3日', '5日', '10日']:
            datas.append(ak.stock_individual_fund_flow_rank(indicator=indi
                ).drop('序号', axis=1).set_index('代码').rename(columns={'最新价': f'{indi}最新价'}))
        datas = pd.concat(datas, axis=1)
        datas['简称'] = datas.iloc[:, 0]
        datas = datas.drop('名称', axis=1)
        datas = datas.replace('-', None).apply(pd.to_numeric, errors='ignore')
        datas.index = pd.MultiIndex.from_product([[cls.today], datas.index], names=['日期', '代码'])
        return datas
    
    @classmethod
    def plate_info(cls, plate: str):
        data = ak.stock_board_industry_cons_em(symbol=plate).set_index('代码')
        return data

    @classmethod
    def balance_sheet(cls, code):
        try:
            data = ak.stock_balance_sheet_by_report_em(symbol=code)
            if data.empty:
                return None
            data = data.drop([
                'SECURITY_CODE', 'SECURITY_NAME_ABBR', 'ORG_CODE', 'ORG_TYPE', 'REPORT_TYPE',
                'REPORT_DATE_NAME', 'SECURITY_TYPE_CODE', 'UPDATE_DATE', 'CURRENCY', 'LISTING_STATE'
            ], axis=1)
            data = data.replace({None: np.nan})
            data = data.astype('float32', errors='ignore')
            data[['REPORT_DATE', 'NOTICE_DATE']] = data[['REPORT_DATE', 'NOTICE_DATE']].astype('datetime64[ns]')
            data = data.set_index('REPORT_DATE')
            data = data.reindex(pd.date_range(data.index.min(), data.index.max(), freq='q'))
            data['SECUCODE'] = data['SECUCODE'][~data['SECUCODE'].isna()].iloc[0]
            data = data.set_index(['SECUCODE', 'NOTICE_DATE'], append=True)
            data.index.names = ['report_date', 'secucode', 'notice_date']
            data = data.rename(columns=lambda x: x.lower())
            return data
        except:
            print(f'{code} get balance sheet failed!, please try again mannually')
            return None

    @classmethod
    def profit_sheet(cls, code):
        try:
            data = ak.stock_profit_sheet_by_report_em(symbol=code)
            if data.empty:
                return None
            data = data.drop([
                'SECURITY_CODE', 'SECURITY_NAME_ABBR', 'ORG_CODE', 'ORG_TYPE', 'REPORT_TYPE',
                'REPORT_DATE_NAME', 'SECURITY_TYPE_CODE', 'UPDATE_DATE', 'CURRENCY'
            ], axis=1)
            data = data.replace({None: np.nan})
            data = data.astype('float32', errors='ignore')
            data[['REPORT_DATE', 'NOTICE_DATE']] = data[['REPORT_DATE', 'NOTICE_DATE']].astype('datetime64[ns]')
            data = data.set_index('REPORT_DATE')
            data = data.reindex(pd.date_range(data.index.min(), data.index.max(), freq='q'))
            data['SECUCODE'] = data['SECUCODE'][~data['SECUCODE'].isna()].iloc[0]
            data = data.set_index(['SECUCODE', 'NOTICE_DATE'], append=True)
            data.index.names = ['report_date', 'secucode', 'notice_date']
            data = data.rename(columns=lambda x: x.lower())
            return data
        except:
            print(f'{code} get balance sheet failed!, please try again mannually')
            return None

    @classmethod
    def cashflow_sheet(cls, code):
        try:
            data = ak.stock_cash_flow_sheet_by_report_em(symbol=code)
            if data.empty:
                return None
            data = data.drop([
                'SECURITY_CODE', 'SECURITY_NAME_ABBR', 'ORG_CODE', 'ORG_TYPE', 'REPORT_TYPE',
                'REPORT_DATE_NAME', 'SECURITY_TYPE_CODE', 'UPDATE_DATE', 'CURRENCY'
            ], axis=1)
            data = data.replace({None: np.nan})
            data = data.astype('float32', errors='ignore')
            data[['REPORT_DATE', 'NOTICE_DATE']] = data[['REPORT_DATE', 'NOTICE_DATE']].astype('datetime64[ns]')
            data = data.set_index('REPORT_DATE')
            data = data.reindex(pd.date_range(data.index.min(), data.index.max(), freq='q'))
            data['SECUCODE'] = data['SECUCODE'][~data['SECUCODE'].isna()].iloc[0]
            data = data.set_index(['SECUCODE', 'NOTICE_DATE'], append=True)
            data.index.names = ['report_date', 'secucode', 'notice_date']
            data = data.rename(columns=lambda x: x.lower())
            return data
        except:
            print(f'{code} get balance sheet failed!, please try again mannually')
            return None

        
    @classmethod
    def index_weight(cls, code: str):
        data = ak.index_stock_cons_weight_csindex(code)
        return data


class Em:

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.6 Safari/605.1.15",
        "Referer": "http://guba.eastmoney.com/",
        "Host": "gubacdn.dfcfw.com"
    }

    @classmethod
    def look_updown(cls, code: str):
        today = datetime.datetime.today().date()
        code = format_code(code, '{market}{code}')
        url = f"http://gubacdn.dfcfw.com/LookUpAndDown/{code}.js"
        res = requests.get(url, headers=cls.headers)
        res.raise_for_status()
        res = eval(res.text.strip('var LookUpAndDown=').replace('null', f'"{today}"'))
        data = pd.Series(res['Data'])
        data['code'] = code
        return data


class StockUS:
    
    __root = "https://api.stock.us/api/v1/"
    headers = {
        "Host": "api.stock.us",
        "Origin": "https://stock.us",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.6 Safari/605.1.15",
        "Accept-Language": "zh-CN",
    }
    category = {
        1: "宏观经济",
        2: "投资策略",
        3: "行业研究",
        4: "晨会早报",
        8: "金工量化",
        9: "债券研究",
        10: "期货研究",
    }
    todaystr = datetime.datetime.today().strftime(r'%Y%m%d')
            
    @classmethod
    def index_price(
        cls, 
        index: str, 
        start: str = None, 
        end: str = None,
    ):
        start = start or '19900101'
        end = end or cls.todaystr
        url = cls.__root + f"index-price?security_code={index}&start={start}&stop={end}"
        res = requests.get(url, headers=cls.headers).json()
        price = pd.DataFrame(res['price'])
        price['date'] = price['date'].astype('datetime64[ns]')
        price = price.set_index('date')
        return price
    
    @classmethod
    def cn_price(
        cls, 
        code: str, 
        start: str = None,
        end: str = None,
    ):
        start = start or '19900101'
        end = end or cls.todaystr
        url = cls.__root + f"cn-price?security_code={code}&start={start}&stop={end}"
        res = requests.get(url, headers=cls.headers).json()
        price = pd.DataFrame(res['price'])
        price['date'] = price['date'].astype('datetime64[ns]')
        price = price.set_index('date')
        return price
    
    @classmethod
    def report_list(
        cls, 
        category: str = 8,
        sub_category: str = 0,
        keyword: str = '', 
        period: str = 'all', 
        org_name: str = '', 
        author: str = '',
        xcf_years: str = '', 
        search_fields: str = 'title',
        page: int = 1, 
        page_size: int = 100
    ):
        '''Get report data in quant block
        ---------------------------------------
        category: str, category to the field, use StockUS.category to see possible choices
        keyword: str, key word to search, default empty string to list recent 100 entries
        period: str, report during this time period
        q: str, search keyword
        org_name: str, search by org_name
        author: str, search by author
        xcf_years: str, search by xcf_years
        search_fields: str, search in fields, support "title", "content", "content_fp"
        page: int, page number
        page_size: int, page size
        '''
        url = cls.__root + 'research/report-list'
        params = (f'?category={category}&dates={period}&q={keyword}&org_name={org_name}'
                  f'&author={author}&xcf_years={xcf_years}&search_fields={search_fields}'
                  f'&page={page}&page_size={page_size}')
        if category != 8:
            params += f'&sub_category={sub_category}'
        headers = {
            "Referer": "https://stock.us/cn/report/quant",
        }
        headers.update(cls.headers)
        url += params
        res = requests.get(url, headers=headers).json()
        data = pd.DataFrame(res['data'])
        data[['pub_date', 'pub_week']] = data[['pub_date', 'pub_week']].astype('datetime64[ns]')
        data.authors = data.authors.map(
            lambda x: ' '.join(list(map(lambda y: y['name'] + ('*' if y['prize'] else ''), x))))
        data = data.set_index('id')
        return data
    
    @classmethod
    def report_search(
        cls, 
        keyword: str = '', 
        period: str = '3m', 
        org_name: str = '', 
        author_name: str = '',
        xcf_years: str = '', 
        search_fields: str = 'title',
        page: int = 1, 
        page_size: int = 100
    ):
        '''Search report in stockus database
        ---------------------------------------
        keyword: str, key word to search, default empty string to list recent 100 entries
        period: str, report during this time period
        org_name: str, search by org_name
        author: str, search by author
        xcf_years: str, search by xcf_years
        search_fields: str, search in fields, support "title", "content", "content_fp"
        page: int, page number
        page_size: int, page size
        '''
        url = cls.__root + 'research/report-search'
        params = (f'?dates={period}&q={keyword}&org_name={org_name}&author_name={author_name}'
                  f'&xcf_years={xcf_years}&search_fields={search_fields}&page={page}'
                  f'&page_size={page_size}')
        url += params
        res = requests.get(url, headers=cls.headers).json()
        data = pd.DataFrame(res['data'])
        data['pub_date'] = data['pub_date'].astype('datetime64[ns]')
        data.authors = data.authors.map(
            lambda x: ' '.join(list(map(lambda y: y['name'] + ('*' if y['prize'] else ''), x)))
            if isinstance(x, list) else '')
        data = data.set_index('id')
        return data


class Cnki:

    __search_url = "https://kns.cnki.net/KNS8/Brief/GetGridTableHtml"

    @classmethod
    def generic_search(cls, keyword: str, page: int = 3):
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'Host': 'kns.cnki.net',
            'Origin': 'https://kns.cnki.net',
            'Referer': 'https://kns.cnki.net/kns8/defaultresult/index',
        }
        data = {
            "IsSearch": 'true',
            "QueryJson": '{"Platform":"","DBCode":"SCDB","KuaKuCode":"CJFQ,CDMD,CIPD,CCND,BDZK,CISD,SNAD,CCJD,GXDB_SECTION,CJFN,CCVD,CLKLK","QNode":{"QGroup":[{"Key":"Subject","Title":"","Logic":1,"Items":[{"Title":"主题","Name":"SU","Value":"' + f'{keyword}' + '","Operate":"%=","BlurType":""}],"ChildItems":[]}]}}',
            "PageName": 'DefaultResult',
            "DBCode": 'SCDB',
            "KuaKuCodes": 'CJFQ,CDMD,CIPD,CCND,BDZK,CISD,SNAD,CCJD,GXDB_SECTION,CJFN,CCVD,CLKLK',
            "SearchSql": "0645419CC2F0B23BC604FFC82ADF67C6E920108EDAD48468E8156BA693E89F481391D6F5096D7FFF3585B29E8209A884EFDF8EF1B43B4C7232E120D4832CCC896D30C069E762ACAB990E5EBAAD03C09721B4573440249365A4157D3C93DC874963F6078A465F9A4E6BEED14E5FD119B250F0488206491CF1C7F670020480B48EE2FF3341B3B9C8A0A38F9913EF596174EDD44BBA8277DA2BE793C92DF83782297DE55F70BBF92D5397159D64D1D3DAC96FAD28213BD3E1912A5B4A4AD58E5965CBDBA01069691140F14FD0298FBD1F452C7779EFF17124633292E356C88367122976245AA928FA07D061C0E091BB1136031750CD76D7D64E9D75B7FBAB11CAA5B80183AC60BB0885D2C0A0938C7D1F849656014326473DCB797D5D273C845DAF7FCE49D21478E9B06B77ADE6253ACD4FE1D87EE31B4B2C94E071EE733B3A64EA6EE9CD5F222FCD3DA1D83D9133EF8C9BED9ED3E55DA15F3B4A37C85463B60D2F0BEA46FC7135898D7D93F63AF8B2246716E32B699238901588EE5D1DEF30A01DCE9957CF6934E8B11E273747F9A9BB8ADF535E5E76F6A9386CFBE605748C132DA05E2D31832199B0A4ECF170ACA47154423CF6BBD9607FC505765E95637F93DC865AA738F5EE92B26DB9AF56509A5FC96FF9C3A1720633EBDDC62EC2162E7D5349CAC851ED0AD4E36DCF6FE25EBEAB42BF931DBE3CF4ED1A7BB8FD887C3C33D86B768B0BA7267C4E0E7DEE53D0931F71F07AE13BAFC46034A444EC24C7EA8F0086FAD197A8D2F18C6CBC5DF48050AF8D4C84DE03B9A6F1DF928D63286B1C924B7EC3BA8C2591D60491F95D271F0E7F02AA2AA93C3888B8CCEBB0414BD7145AD15A3166DB4860F85BC476B1B193C219EAE52E33E6BBC9B3AAAD97196977B7DABA36C04093ED723AD874EC6480477C6412B0F589DE6CC7D959855E41265213DCBB4D91238716DF38BF78C951259572F8E5968FAC5C5CDC006DBE919EEB5E5518F51162FCE7CDE520F60093D333FBE121D3164C6D2451F6431FB7973C659E6A9D287B545EC044DE2CBE170F3627719F8418D44E17987CEC7A89B52CB5525AF795DA892475ABF871C3A5A5FCBC5B03EB9BEC8598C8ADD7A68984BBBEF1244DD90386C05756687AB9D87A0B521319C093C3EC0D5EBEFDAB5459E29F1DA03D4C25DE740BF9FA2BC07DD510386E3BBE89F10D45513E29C8CF904763E723CE4BF2928D4DC2A731DD53595E9AACED90679FCDDACED022ECD59D72600A736D555A8B76BFE4CCD861E6A7F5A219EBE9A228BD008928299DB999D18F9CDD2E57E8C03EDF236E62EDB17A1FE5B023CF6E5A11892A5FA17EE5CFE348CA290DC691987A535223133D8CA101E8ABF13EFCAD929635E090B3C6BB6838E33B7C78C1DBA274101A6584300EF8D38C983AD544264217F6793562D19715CD711295C5410C72E88A64BD23D9049E5DF15EA6B3EB4473C1DDEBB416459322FEF0CC61D894476DCD62569527BE23FB7F66DF3F5182ABF2472FB60039CA77218F356D7F82E4EBAAA4C6875B5BD4729C81A29BDF55ED223AA0DAB04E1B248524FC504711360C330186327A780D6487BA831ABE55AAE38E69A0FBEF89D560E7AA26B991966E4B644338863E80AD9D1ACAD459EA933644C5A0D2EA44AD17205AED3BE66AEC01F48BA032EEBD620E2713082FE8D31E4A05A34F18BD389587FA4D3A9DFBB8C16AEE9C5FA9E667BA12A07B757D82F7BB41AC8867D9947CCBA3BB26381EC6D0D3966338DB6FA3D1A61F99A978C3B5ED2B31B7C14D54A4F688C4925C8AF99CB3EE3C2C06C7D35AD891BF0CFC820529FD990F2FF319BE195B1AD23C1667031C072EB1964F8512BB779125E46773C01714FCF0E339AEB0C44FB91B896A7A95AF4F81EB49006B570BC03ECA7D8DA45679F3B46A7AE3B46ED8D319CED49A3A5881A37CD3770703BDF026ACEF7D8662F85AFDBDD36C540FD419E18F30EA0483D24350B7C34C43F3D0065F339EAC15749DF8849F3880378FEA4AD7CCBAA827C828A5CAF7D56E97A87A3FAEEAE136B35FB37E8CE0233D9AF8DEABD47BD5B36A1B42B995D4F96FE744A2E25E9B6107801CACCA0DDC2B7ED5BFD39F68AB2E2BB66AB8286061049F3B5FFE871FFA520A7C0EEE3DEDF417D078DF9013B5F5251A07AE3D4D00B9AF1560200CC981D0E8BE17C9CE204C21E5E543C9E55421D4FCE2C309C68D376E3787AB4640FA99B82988A288FD22A2E0C9225E39A5DAA7EBEB0376912C9CA255A7AE49F3C5AB262B4FFFBA98A9548623C16D0C97C7315DF5FFD1507102EAA730E5247F1C492D49A45121347CFF39A5181729F1D33F28FA48035CBC02CF87DAF72067D70B524421AB21FF137A2C7AB2F90DAD1BA1786C16728E7B78DB0461B5B1E8CF7B88E765E67AF4E458EF3A5125D90DA88CE97D9AB9C4363E4A7D6B7F3B0420B93FEDF72248E076EC0871EDFC5744AC6F9F591CEC4CE3E0E681E1C1B21AFCC5BF5B22116F7E7A3ABA561F68F8AE685DA926756CD70C0E6057C7737537F972F8942CCFD073400F0D5C23F107F55FC07745ED334FB97130860A0B7B0B5B4B2B23417EA63C65BAF1624254BBA167373F1D6C0E0BB5A67F92008CFCA4F24276E725FD05802F94A5CC7E52CC005017C58A8757BDEDED54538DA513E975DFCDC7D3FA95552E960ABA05EB7C33CA37CCA1C93DFF13A493174A9BB3228118E0F2AEBBAEE074D557B6FA6000F0E5C73D563BB8E3598B4D8E94DDCAFEB5BBCDF74D39CCC8AD27A5D3C0CAB59DA24BEB86C10F8584878FA94BE9F1F9D2FA01023A5B838BDCD18C58E4F08C0BF1C31ED25B32438C95D613B5227B0C63CE5B090A49B23416A06BCB9365406EE953CB1245CA00A7791C1F10267F95FD6A5B93F78DBDA6C96F036928F943A8CED955AEF96C63CF849B30EFD0B94BC88E124F1CE2B186D0120F40",
            "CurPage": '1',
            "RecordsCntPerPage": '50',
            "CurDisplayMode": 'listmode',
            "CurrSortField": r'%e5%8f%91%e8%a1%a8%e6%97%b6%e9%97%b4%2f(%e5%8f%91%e8%a1%a8%e6%97%b6%e9%97%b4%2c%27TIME%27)',
            "CurrSortFieldType": 'desc',
            "IsSentenceSearch": 'false',
            "Subject": '',
        }
        results = []
        # first attempt to get result and total pages
        req = requests.post(cls.__search_url, headers=headers, data=data)
        text = req.text
        total = int(re.findall(r'共找到.{0,}?([\d,]+).{0,}?条结果', text)[0].replace(',', ''))
        if page == -1:
            page = ceil(total / 50)
        if page == 1:
            return pd.concat(results, axis=1)
        page = min(page, ceil(total / 50))
        print(f'[+] Current page 1 / {page}')
        result = pd.read_html(text)[0]
        results.append(result)

        # now if the crawl didn't end, it will enter the cycle for crawling
        for p in range(2, page + 1):
            data.update(CurPage=f'{p}', IsSearch='false')
            req = requests.post(cls.__search_url, headers=headers, data=data)
            text = req.text
            print(f'[+] Current page {p} / {page}')
            result = pd.read_html(text)[0]
            results.append(result)
        results = pd.concat(results, axis=0)
        results = results.drop(results.columns[0], axis=1)
        return results


class WeiboSearch:
    '''A search crawler engine for weibo
    ====================================
    sample usage:
    >>> result = WeiboSearch.search("keyword")
    '''

    __base = "https://m.weibo.cn/api/container/getIndex?containerid=100103type%3D1%26q%3D{}&page_type=searchall&page={}"

    @classmethod
    def _get_content(cls, url, headers):

        def _parse(mblog):
            blog = {
                "created_at": mblog["created_at"],
                "text": re.sub(r'<(.*?)>', '', mblog['text']),
                "id": mblog["id"],
                "link": f"https://m.weibo.cn/detail/{mblog['id']}",                    
                "source": mblog["source"],
                "username": mblog["user"]["screen_name"],
                "reposts_count": mblog["reposts_count"],
                "comments_count": mblog["comments_count"],
                "attitudes_count": mblog["attitudes_count"],
                "isLongText": mblog["isLongText"],
            }
            if blog["isLongText"]:
                headers = {
                    "Referer": f"https://m.weibo.cn/detail/{blog['id']}",
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.2 Safari/605.1.15"
                }
                resp = requests.get(f"https://m.weibo.cn/statuses/extend?id={blog['id']}", headers=headers).json()
                blog["full_text"] = resp["data"]["longTextContent"]
            return blog

        # First try to get resources
        res = requests.get(url, headers=headers).json()
        # if it is end
        if res.get("msg"):
            return False

        # if it contains cards
        cards = res["data"]["cards"]
        blogs = []
        for card in cards:
            # find 'mblog' tag and append to result blogs
            mblog = card.get("mblog")
            card_group = card.get("card_group")
            if card.get("mblog"):
                blog = _parse(mblog)
                blogs.append(blog)
            elif card_group:
                for cg in card_group:
                    mblog = cg.get("mblog")
                    if mblog:
                        blog = _parse(mblog)
                        blogs.append(blog)
        return blogs
    
    @classmethod
    def _get_full(cls, keyword: str):
        page = 1
        result = []
        headers = {
            "Referer": f"https://m.weibo.cn/search?containerid=100103type%3D1%26q%3D{quote(keyword, 'utf-8')}",
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
            }
        print(f"Start in keyword: {keyword}")
        while True:
            print(f"Getting {keyword}, currently at page: {page} ... ")
            url = cls.__base.format(keyword, page)
            blogs = cls._get_content(url, headers)
            if not blogs:
                break
            result.extend(blogs)
            page += 1
            time.sleep(random.randint(5, 8))
        print(f"Finished in keyword: {keyword}!")
        return result
    
    @classmethod
    def _get_assigned(cls, keyword: str, pages: int):
        result = []
        print(f"Start in keyword: {keyword}")
        headers = {
            "Referer": f"https://m.weibo.cn/search?containerid=100103type%3D1%26q%3D{quote(keyword, 'utf-8')}",
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
            }
        for page in tqdm(range(1, pages+1)):
            print(f"Getting {keyword}, currently at page: {page} ... ")
            url = cls.__base.format(keyword, page)
            blogs = cls._get_content(url, headers)
            result.extend(blogs)
            time.sleep(random.randint(5, 8))
        print(f"Finished in keyword: {keyword}!")
        return result          
    
    @classmethod
    def search(cls, keyword: str, pages: int = -1):
        """Search for the keyword
        --------------------------
        
        keyword: str, keyword
        pages: int, how many pages you want to get, default -1 to all pages
        """

        keyword = keyword.replace('#', '%23')
        if pages == -1:
            result = cls._get_full(keyword)
        else:
            result = cls._get_assigned(keyword, pages)
        result = pd.DataFrame(result)
        return result


class HotTopic:
    """A Second Level Crawler for Hot Topic
    ========================================
    sample usage:
    >>> result = HotTopic.search('keyword')
    """

    __list = "https://google-api.zhaoyizhe.com/google-api/index/mon/list"
    __search = "https://google-api.zhaoyizhe.com/google-api/index/mon/sec?isValid=ads&keyword={}"
    __trend = "https://google-api.zhaoyizhe.com/google-api/index/superInfo?keyword={}"
    
    @classmethod
    def search(cls, keyword: str = None, date: str = None):
        if keyword is None and date is None:
            url = cls.__list
        elif keyword is None and date is not None:
            url = cls.__search.format(date)
        elif keyword is not None and date is None:
            url = cls.__search.format(keyword)
        result = requests.get(url).json()
        data = result["data"]
        data = pd.DataFrame(data)
        data = data.drop("_id", axis=1)
        return data

    @classmethod
    def trend(cls, keyword: str):
        url = cls.__trend.format(keyword)
        result = requests.get(url).json()
        data = pd.DataFrame(map(lambda x: x['value'], result), 
            columns=['datetime', 'hot', 'tag']).set_index('datetime')
        return data

    @classmethod
    def trend_history(cls, keyword: str, freq: str = '3m'):
        if freq not in ['1h', '24h', '1m', '3m']:
            raise ValueError('Freq parameter must be in ["1h", "24h', "1m", "3m]")
        if freq.endswith('h'):
            freq += 'our'
        elif freq.endswith('m'):
            freq += 'onth'
        url = "https://data.weibo.com/index/ajax/newindex/searchword"
        data = {
            "word": f"{keyword}"
        }
        headers = {
            "Host": "data.weibo.com",
            "Origin": "https://data.weibo.com",
            "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Mobile/15E148 MicroMessenger/8.0.16(0x18001041) NetType/WIFI Language/zh_CN",
            "Content-Length": "23",
            "Referer": "https://data.weibo.com/index/newindex?visit_type=search"
        }
        html = requests.post(url, data=data, headers=headers)
        html = BeautifulSoup(html.text, 'html.parser')
        res = html.find_all('li')
        wids = [(r.attrs["wid"].strip(r'\"'), eval('"' + r.attrs["word"].replace(r'\"', '') + '"')) for r in res]

        url = "https://data.weibo.com/index/ajax/newindex/getchartdata"
        results = []
        for wid in wids:
            post_params = {
                "wid": wid[0],
                "dateGroup": freq
            }
            res = requests.post(url, data=post_params, headers=headers).json()
            data = res["data"]
            index = data[0]["trend"]['x']
            index = list(map(lambda x: x.replace("月", '-').replace("日", ''), index))
            volume = data[0]["trend"]['s']
            result = pd.Series(volume, index=index, name=wid[1])
            results.append(result)
        results = pd.concat(results, axis=1)
        return results


class KaiXin(Request):

    def __init__(self, page_count: int = 10):
        url = [f"http://www.kxdaili.com/dailiip/2/{i}.html" for i in range(1, page_count + 1)]
        super().__init__(url = url)
    
    def callback(self):
        results = []
        etrees = self.etree
        for tree in etrees:
            if tree is None:
                continue
            for tr in tree.xpath("//table[@class='active']//tr")[1:]:
                ip = "".join(tr.xpath('./td[1]/text()')).strip()
                port = "".join(tr.xpath('./td[2]/text()')).strip()
                results.append({
                    "http": "http://" + "%s:%s" % (ip, port),
                    "https": "https://" + "%s:%s" % (ip, port)
                })
        return pd.DataFrame(results)


class KuaiDaili(Request):

    def __init__(self, page_count: int = 20):
        url_pattern = [
            'https://www.kuaidaili.com/free/inha/{}/',
            'https://www.kuaidaili.com/free/intr/{}/'
        ]
        url = []
        for page_index in range(1, page_count + 1):
            for pattern in url_pattern:
                url.append(pattern.format(page_index))
        super().__init__(url=url, delay=4)

    def callback(self):
        results = []
        for tree in self.etree:
            if tree is None:
                continue
            proxy_list = tree.xpath('.//table//tr')
            for tr in proxy_list[1:]:
                results.append({
                    "http": "http://" + ':'.join(tr.xpath('./td/text()')[0:2]),
                    "https": "http://" + ':'.join(tr.xpath('./td/text()')[0:2])
                })
        return pd.DataFrame(results)


class Ip3366(Request):

    def __init__(self, page_count: int = 3):
        url = []
        url_pattern = ['http://www.ip3366.net/free/?stype=1&page={}', "http://www.ip3366.net/free/?stype=2&page={}"]
        for page in range(1, page_count + 1):
            for pat in url_pattern:
                url.append(pat.format(page))
        super().__init__(url=url)

    def callback(self, *args, **kwargs):
        results = []
        for text in self.html:
            if text is None:
                continue
            proxies = re.findall(r'<td>(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})</td>[\s\S]*?<td>(\d+)</td>', text)
            for proxy in proxies:
                results.append({"http": "http://" + ":".join(proxy), "https": "http://" + ":".join(proxy)})
        return pd.DataFrame(results)


class Ip98(Request):

    def __init__(self, page_count: int = 20):
        super().__init__(url=[f"https://www.89ip.cn/index_{i}.html" for i in range(1, page_count + 1)])
    
    def callback(self, *args, **kwargs):
        results = []
        for text in self.html:
            if text is None:
                continue
            proxies = re.findall(
                r'<td.*?>[\s\S]*?(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})[\s\S]*?</td>[\s\S]*?<td.*?>[\s\S]*?(\d+)[\s\S]*?</td>',
                text
            )
            for proxy in proxies:
                results.append({"http": "http://" + ":".join(proxy), "https": "http://" + ":".join(proxy)})
        return pd.DataFrame(results)


class Checker(Request):

    def __init__(self, proxies: list[dict], url: str = "http://httpbin.org/ip"):
        super().__init__(
            url = [url] * len(proxies),
            proxies = proxies,
            timeout = 2.0,
            retry = 1,
        )

    def _req(self, url: str, proxy: dict):
        method = getattr(requests, self.method)
        retry = self.retry or 1
        for t in range(1, retry + 1):
            try:
                resp = method(
                    url, headers=self.headers, proxies=proxy,
                    timeout=self.timeout, **self.kwargs
                )
                resp.raise_for_status()
                if self.verbose:
                    print(f'[+] {url} try {t}')
                return resp
            except Exception as e:
                if self.verbose:
                    print(f'[-] {e} {url} try {t}')
                time.sleep(self.delay)
        return None

    def request(self) -> list[requests.Response]:
        responses = []
        for proxy, url in zip(self.proxies, self.url):
            resp = self._req(url=url, proxy=proxy)
            responses.append(resp)
        self.responses = responses
        return self
    
    def para_request(self) -> list[requests.Response]:
        self.responses = Parallel(n_jobs=-1, backend='loky')(
            delayed(self._req)(url, proxy) for url, proxy in zip(self.url, self.proxies)
        )
        return self
    
    def callback(self):
        results = []
        for i, res in enumerate(self.responses):
            if res is None:
                continue
            results.append(self.proxies[i])
        return pd.DataFrame(results)

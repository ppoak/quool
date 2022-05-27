import datetime
import pandas as pd
from ..tools import *


class StockUS():
    
    root = "https://api.stock.us/api/v1/"
    category = {
        "金工量化": 8,
    }

    def __init__(self, sessionid: str = None):
        StockUS.sessionid = sessionid
        StockUS.headers = {
            "Host": "api.stock.us",
            "Origin": "https://stock.us"
        }
        if sessionid is not None:
            StockUS.headers["Cookie"] = f"sessionid={sessionid}",
            
    @classmethod
    @cache()
    def index_price(cls, index: str, start: str, end: str):
        url = cls.root + f"index-price?security_code={index}&start={start}&stop={end}"
        res = Request(url, headers=cls.headers).get().json
        price = pd.DataFrame(res['price'])
        price['date'] = price['date'].astype('datetime64[ns]')
        price = price.set_index('date')
        return price
    
    @classmethod
    @cache()
    def cn_price(cls, code: str, start: str, end: str):
        url = cls.root + f"cn-price?security_code={code}&start={start}&stop={end}"
        res = Request(url, headers=cls.headers).get().json
        price = pd.DataFrame(res['price'])
        price['date'] = price['date'].astype('datetime64[ns]')
        price = price.set_index('date')
        return price
    
    @classmethod
    @cache()
    def research_report(cls, category: str = '金工量化', period: str = '1m', 
                        q: str = '', org_name: str = '', author: str = '',
                        xcf_years: str = '', search_fields: str = 'title',
                        page: int = 1, page_size: int = 100):
        url = cls.root + 'research/report-list'
        params = (f'?category={cls.category[category]}&dates={period}&q={q}&org_name={org_name}'
                  f'&author={author}&xcf_years={xcf_years}&search_fields={search_fields}'
                  f'&page={page}&page_size={page_size}')
        url += params
        res = Request(url, headers=cls.headers).get().json
        data = pd.DataFrame(res['data'])
        data[['pub_date', 'pub_week']] = data[['pub_date', 'pub_week']].astype('datetime64[ns]')
        data.authors = data.authors.map(
            lambda x: ' '.join(list(map(lambda y: y['name'] + ('*' if y['prize'] else ''), x))))
        data = data.set_index('id')
        return data

class Em:
    __data_center = "https://datacenter-web.eastmoney.com/api/data/v1/get"

    @classmethod
    @cache()
    def active_opdep(cls, date: 'str | datetime.datetime') -> pd.DataFrame:
        '''Update data for active oprate department
        --------------------------------------------

        date: str or datetime, the given date
        return: pd.DataFrame, a dataframe containing information on eastmoney
        '''
        date = time2str(date)
        params = {
            "sortColumns": "TOTAL_NETAMT,ONLIST_DATE,OPERATEDEPT_CODE",
            "sortTypes": "-1,-1,1",
            "pageSize": 100000,
            "pageNumber": 1,
            "reportName": "RPT_OPERATEDEPT_ACTIVE",
            "columns": "ALL",
            "source": "WEB",
            "client": "WEB",
            "filter": f"(ONLIST_DATE>='{date}')(ONLIST_DATE<='{date}')"
        }
        headers = {
            "Referer": "https://data.eastmoney.com/"
        }
        res = Request(cls.__data_center, headers=headers, params=params).get().json
        data = res['result']['data']
        data = pd.DataFrame(data)
        data = data.rename(columns=dict(zip(data.columns, data.columns.map(lambda x: x.lower()))))
        data.onlist_date = pd.to_datetime(data.onlist_date)
        return data
    
    @classmethod
    @cache()
    def active_opdep_details(cls, date: 'str | datetime.datetime') -> pd.DataFrame:
        date = time2str(date)
        params = {
            "sortColumns": "TOTAL_NETAMT,ONLIST_DATE,OPERATEDEPT_CODE",
            "sortTypes": "-1,-1,1",
            "pageSize": 100000,
            "pageNumber": 1,
            "reportName": "RPT_OPERATEDEPT_ACTIVE",
            "columns": "ALL",
            "source": "WEB",
            "client": "WEB",
            "filter": f"(ONLIST_DATE>='{date}')(ONLIST_DATE<='{date}')"
        }
        headers = {
            "Referer": "https://data.eastmoney.com/"
        }
        res = Request(cls.__data_center, headers=headers, params=params).get().json
        data = res['result']['data']
        data = pd.DataFrame(data)
        data = data.rename(columns=dict(zip(data.columns, data.columns.map(lambda x: x.lower()))))
        data.onlist_date = pd.to_datetime(data.onlist_date)
        data_parent = data[['onlist_date', 'operatedept_code']]
        datas = []
        for i in range(len(data)):
            opdep_code = data_parent.iloc[i, :]['operatedept_code']
            params = {
                "sortColumns": "TRADE_DATE,SECURITY_CODE",
                "sortTypes": "-1,1",
                "pageSize": 100000,
                "pageNumber": 1,
                "reportName": "RPT_OPERATEDEPT_TRADE_DETAILS",
                "columns": "ALL",
                "filter": f"(OPERATEDEPT_CODE={opdep_code})",
                "source": "WEB",
                "client": "WEB"
            }
            headers = {
                "Referer": "https://data.eastmoney.com/"
            }
            res = Request(cls.__data_center, headers=headers, params=params).get().json
            data = res['result']['data']
            data = pd.DataFrame(data)
            datas.append(data)
            
        datas = pd.concat(datas, axis=0).reset_index(drop=True)
        datas = datas.rename(columns=dict(zip(datas.columns, datas.columns.map(lambda x: x.lower()))))
        datas.trade_date = pd.to_datetime(datas.trade_date)
        datas = datas.loc[datas['trade_date'] == date]
        return datas
        
    @classmethod
    @cache()
    def institution_trade(cls, date: 'str | datetime.datetime') -> pd.DataFrame:
        date = time2str(date)
        params = {
            'sortColumns' : 'NET_BUY_AMT,TRADE_DATE,SECURITY_CODE',
            'sortTypes' : '-1,-1,1',
            'pageSize' : '100000',
            'pageNumber' : '1',
            'reportName' : 'RPT_ORGANIZATION_TRADE_DETAILS',
            'columns' : 'ALL', 
            'source' : 'WEB',
            'client' :  'WEB', 
            'filter' : f"(TRADE_DATE='{date}')"
        }
    
        headers = {
            "Referer": "https://data.eastmoney.com/"
        }
        res = Request(cls.__data_center, headers=headers, params=params).get().json
        data = res['result']['data']
        data = pd.DataFrame(data)
        data = data.rename(columns=dict(zip(data.columns, data.columns.map(lambda x: x.lower()))))
        data.trade_date = pd.to_datetime(data.trade_date)
        return data
        
    @classmethod
    @cache()
    def oversea_institution_holding(cls, date: 'str | datetime.datetime') -> pd.DataFrame:
        import requests
        import numpy as np
        import re
        date = time2str(date)
        main_page = 'https://data.eastmoney.com/hsgtcg/InstitutionQueryMore.html'
        res = requests.get(main_page)
        res.raise_for_status()
        institution_list = re.findall(r'var jgList= \[.*\];', res.text)[0].split('=')[1].strip(';')
        institution_list = eval(institution_list)
        datas = []
        for institution in institution_list:
            name = institution['PARTICIPANT_CODE']
            callbackfunc = 'jQuery1123032132491687413733_1646408202496'
            for i in range(1, 10):
                params = {
                    'callback': callbackfunc,
                    'sortColumns': 'HOLD_DATE',
                    'sortTypes': '-1',
                    'pageSize': 500,
                    'pageNumber': i,
                    'reportName': 'RPT_MUTUAL_HOLD_DET',
                    'columns': 'ALL',
                    'source': 'WEB',
                    'client': 'WEB',
                    'filter': f'(PARTICIPANT_CODE="{name}")' + \
                        f'(MARKET_CODE in ("001","003"))(HOLD_DATE=\'{date}\')',
                }
                headers = {
                    "Referer": "https://data.eastmoney.com/"
                }
                res = Request(url=cls.__data_center, headers=headers, params=params).get().response
                data = eval(res.text.replace('true', 'True').replace('false', 'False').\
                    replace('null', 'np.nan').replace(callbackfunc, '')[1:-2])
                if data['result'] is not np.nan:
                    data = pd.DataFrame(data['result']['data'])
                    datas.append(data)
                else:
                    break
            
        datas = pd.concat(datas, axis=0).reset_index(drop=True)
        datas = datas.rename(columns=dict(zip(datas.columns, datas.columns.map(lambda x: x.lower()))))
        datas.hold_date = pd.to_datetime(datas.hold_date)
        return datas
    
    @classmethod
    @cache()
    def stock_buyback(cls, date: 'str | datetime.datetime') -> pd.DataFrame:
        date = time2str(date)
        datas = []
        for i in range(50):
            params = {
                'sortColumns': 'dim_date',
                'sortTypes': -1,
                'pageSize': 500,
                'pageNumber': i,
                'reportName': 'RPTA_WEB_GETHGLIST',
                'columns': 'ALL',
                'source': 'WEB',
            }
            headers = {
                "Referer": "https://data.eastmoney.com/"
            }
            res = Request(cls.__data_center, headers=headers, params=params).get().json
            if res['result'] is None:
                break
            data = res['result']['data']
            data = pd.DataFrame(data)
            datas.append(data)
        datas = pd.concat(datas, axis=0).reset_index(drop=True)
        datas = datas.rename(columns=dict(zip(datas.columns, datas.columns.map(lambda x: x.lower()))))
        datas.repurenddate = pd.to_datetime(datas.repurenddate)
        datas.repurstarrtdate = pd.to_datetime(datas.repurstartdate)
        datas.updatedate = pd.to_datetime(datas.updatedate)
        datas.dim_date = pd.to_datetime(datas.dim_date)
        datas.dim_tradedate = pd.to_datetime(datas.dim_tradedate)
        datas = datas.loc[datas['dim_date'] == date]
        return datas

if __name__ == "__main__":
    data = Em.oversea_institution_holding('20220523')


import datetime
import pandas as pd
from functools import lru_cache
from ..tools import *


class StockUS():
    
    root = "https://api.stock.us/api/v1/"

    def __init__(self, sessionid: str):
        StockUS.sessionid = sessionid
        StockUS.headers = {
            "Cookie": f"sessionid={sessionid}",
            "Host": "api.stock.us",
            "Origin": "https://stock.us"
        }
    
    @classmethod
    @lru_cache(maxsize=None, typed=False)
    def index_price(cls, index: str, start: str, end: str):
        url = cls.root + f"index-price?security_code={index}&start={start}&stop={end}"
        res = Request(url, headers=cls.headers).get().json
        price = pd.DataFrame(res['price'])
        price['date'] = pd.to_datetime(price['date'])
        price = price.set_index('date')
        return price
    
    @classmethod
    @lru_cache(maxsize=None, typed=False)
    def cn_price(cls, code: str, start: str, end: str):
        url = cls.root + f"cn-price?security_code={code}&start={start}&stop={end}"
        res = Request(url, headers=cls.headers).get().json
        price = pd.DataFrame(res['price'])
        price['date'] = pd.to_datetime(price['date'])
        price = price.set_index('date')
        return price

class Em:
    __data_center = "https://datacenter-web.eastmoney.com/api/data/v1/get"

    @classmethod
    @lru_cache(maxsize=None, typed=False)
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
    @lru_cache(maxsize=None, typed=False)
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
        data = data[['onlist_date', 'operatedept_code']]
        datas = []
        for i in range(len(data)):
            opdep_code = data.iloc[i, :]['operatedept_code']
            print(opdep_code)
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
            data_sub = res['result']['data']
            data_sub = pd.DataFrame(data_sub)
            data_sub = data_sub.rename(columns=dict(zip(data_sub.columns, data_sub.columns.map(lambda x: x.lower()))))
            data_sub.trade_date = pd.to_datetime(data_sub.trade_date)
            data_sub = data_sub.loc[data_sub['trade_date'] == date]
            datas.append(data_sub)
            
        datas = pd.concat(datas, axis=0)
        return datas

if __name__ == "__main__":
    data = Em.active_opdep_details('20210506')
    data.to_csv('data.csv')
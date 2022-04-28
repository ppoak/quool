import os
import datetime
import pandas as pd
from ..tools import *
from functools import lru_cache
from sqlalchemy.pool import NullPool
from sqlalchemy import create_engine


@pd.api.extensions.register_dataframe_accessor("filer")
@pd.api.extensions.register_series_accessor("filer")
class Filer(Worker):
    
    def to_multisheet_excel(self, path, **kwargs):
        if self.type_ == Worker.PN and self.is_frame:
            with pd.ExcelWriter(path) as writer:
                for column in self.dataframe.columns:
                    self.dataframe[column].unstack(level=1).to_excel(writer, sheet_name=str(column), **kwargs)
        
        elif self.type_ == Worker.PN and not self.is_frame:
            self.data.unstack().to_excel(path, **kwargs)
        
        else:
            self.dataframe.to_excel(path, **kwargs)

    @staticmethod
    def read_excel(path, perspective: str = None, **kwargs):
        '''A dummy function of pd.read_csv, which provide multi sheet reading function'''
        if perspective is None:
            return pd.read_excel(path, **kwargs)
        
        sheets_dict = pd.read_excel(path, sheet_name=None, **kwargs)
        datas = []
        if perspective == "indicator":
            for indicator, data in sheets_dict.items():
                data = data.stack()
                data.name = indicator
                datas.append(data)
            datas = pd.concat(datas, axis=1)

        elif perspective == "asset":
            for asset, data in sheets_dict.items():
                data.index = pd.MultiIndex.from_product([data.index, [asset]])
                datas.append(data)
            datas = pd.concat(datas)
            datas = data.sort_index()

        elif perspective == "datetime":
            for datetime, data in sheets_dict.items():
                data.index = pd.MultiIndex.from_product([[datetime], data.index])
                datas.append(data)
            datas = pd.concat(datas)

        else:
            raise ValueError('perspective must be in one of datetime, indicator or asset')
        
        return datas

    @staticmethod
    def read_csv_directory(path, perspective: str, **kwargs):
        '''A enhanced function for reading files in a directory to a panel DataFrame
        ----------------------------------------------------------------------------

        path: path to the directory
        perspective: 'datetime', 'asset', 'indicator'
        kwargs: other arguments for pd.read_csv

        **note: the name of the file in the directory will be interpreted as the 
        sign(column or index) to the data, so set it to the brief one
        '''
        files = os.listdir(path)
        datas = []
        
        if perspective == "indicator":
            for file in files:
                name = os.path.splitext(file)[0]
                data = pd.read_csv(os.path.join(path, file), **kwargs)
                data = data.stack()
                data.name = name
                datas.append(data)
            datas = pd.concat(datas, axis=1).sort_index()

        elif perspective == "asset":
            for file in files:
                name = os.path.splitext(file)[0]
                data = pd.read_csv(os.path.join(path, file), **kwargs)
                data.index = pd.MultiIndex.from_product([data.index, [name]])
                datas.append(data)
            datas = pd.concat(datas).sort_index()
            
        elif perspective == "datetime":
            for file in files:
                name = os.path.splitext(file)[0]
                data = pd.read_csv(os.path.join(path, file), **kwargs)
                data.index = pd.MultiIndex.from_product([pd.to_datetime([name]), data.index])
                datas.append(data)
            datas = pd.concat(datas).sort_index()
        
        return datas

    @staticmethod
    def read_excel_directory(path, perspective: str, **kwargs):
        '''A enhanced function for reading files in a directory to a panel DataFrame
        ----------------------------------------------------------------------------

        path: path to the directory
        perspective: 'datetime', 'asset', 'indicator'
        kwargs: other arguments for pd.read_excel

        **note: the name of the file in the directory will be interpreted as the 
        sign(column or index) to the data, so set it to the brief one
        '''
        files = os.listdir(path)
        datas = []
        
        if perspective == "indicator":
            for file in files:
                name = os.path.splitext(file)[0]
                data = pd.read_excel(os.path.join(path, file), **kwargs)
                data = data.stack()
                data.name = name
                datas.append(data)
            datas = pd.concat(datas, axis=1).sort_index()

        elif perspective == "asset":
            for file in files:
                name = os.path.splitext(file)[0]
                data = pd.read_excel(os.path.join(path, file), **kwargs)
                data.index = pd.MultiIndex.from_product([data.index, [name]])
                datas.append(data)
            datas = pd.concat(datas).sort_index()
            
        elif perspective == "datetime":
            for file in files:
                name = os.path.splitext(file)[0]
                data = pd.read_excel(os.path.join(path, file), **kwargs)
                data.index = pd.MultiIndex.from_product([pd.to_datetime([name]), data.index])
                datas.append(data)
            datas = pd.concat(datas).sort_index()
        
        return datas

class Database(object):

    def __init__(self, user: str, password: str):
        Database.user = user
        Database.password = password

        base = "mysql+pymysql://{user}:{password}@127.0.0.1/{database}?charset=utf8"
        Database.stock = create_engine(base.format(user=user, password=password, database="stock"),
            poolclass=NullPool, connect_args={"charset": "utf8", "connect_timeout": 10})
        Database.fund = create_engine(base.format(user=user, password=password, database="fund"),
            poolclass=NullPool, connect_args={"charset": "utf8", "connect_timeout": 10})
        Database.factor = create_engine(base.format(user=user, password=password, database="factor"),
            poolclass=NullPool, connect_args={"charset": "utf8", "connect_timeout": 10})
        
        today = datetime.datetime.today()
        if today.hour > 20:
            Database.today = (datetime.datetime.today() - datetime.timedelta(days=1)).strftime(r'%Y%m%d')
        else:
            Database.today = today.strftime(r'%Y%m%d')
    
    @classmethod
    def _get_panel_data(cls, start: str, end: str, date_col: str,
        code: 'str | list', code_col: str, fields: list, 
        conditions: 'str | list', table: str,
        database, index: list) -> pd.DataFrame:
        start = start or '20100101'
        end = end or cls.today
        if isinstance(code, str):
            code = [code]
        if isinstance(conditions, str):
            conditions = [conditions]
        
        if fields:
            fields = ','.join(fields)
        else:
            fields = '*'
        
        query = f'select {fields} from {table}' \
            f' where ( {date_col} between "{start}" and "{end}" )' \
            
        if code:
            query += ' and ' + '(' + ' or '.join([rf'{code_col} like "%%{c}%%"' for c in code]) + ')'
        
        if conditions:
            query += ' and ' + '(' + ' or '.join([rf'{c}' for c in conditions]) + ')'
        
        data = pd.read_sql(query, database)
        
        index = list(filter(lambda x: x in fields, index))
        if index:
            data = data.set_index(index)
        
        return data

    @classmethod
    def trade_date(cls, start: str = None,
        end: str = None, freq: str = 'daily',
        weekday: int = None) -> list[str]:
        '''get trade date during a period
        ---------------------------------

        start: datetime or str
        end: datetime or date or str, end date in 3 forms
        freq: str, frequency in either 'daily', 'weekly' or 'monthly'
        '''
        start = start or "20100101"
        end = end or cls.today
        
        query = f"select trade_date from trade_date_{freq} " \
                f"where trade_date >= '{start}' " \
                f"and trade_date <= '{end}'" 

        if weekday and freq == 'daily':
            query += f" and weekday = {weekday}"

        data = pd.read_sql(query, cls.stock)
        data = data.trade_date.sort_values().tolist()
        return data

    @classmethod
    def market_daily(cls, start: str = None, end: str = None,
        code: 'str | list' = None, fields: list = None,
        conditions: 'str | list' = None) -> pd.DataFrame:
        '''get market data in daily frequency
        -------------------------------------

        start: datetime or date or str, start date in 3 forms
        end: datetime or date or str, end date in 3 forms
        fields: list, the field names you want to get
        '''
        data = cls._get_panel_data(
            start = start,
            end = end,
            date_col = "trade_date",
            code = code,
            code_col = "code",
            fields = fields,
            conditions = conditions,
            table = 'market_daily',
            database = cls.stock,
            index = ['trade_date', 'code']
        )
        return data

    @classmethod
    def index_market_daily(cls, start: str = None, end: str = None,
        code: 'str | list' = None, fields: list = None,
        conditions: 'str | list' = None) -> pd.DataFrame:
        '''get index market data in daily frequency
        -------------------------------------

        start: datetime or date or str, start date in 3 forms
        end: datetime or date or str, end date in 3 forms
        code: str, the index code
        fields: list, the field names you want to get
        conditions: list, a series of conditions like "code = '000001.SZ'" listed in a list
        '''
        data = cls._get_panel_data(
            start = start,
            end = end,
            date_col = "trade_date",
            code = code,
            code_col = "index_code",
            fields = fields,
            conditions = conditions,
            table = 'index_market_daily',
            database = cls.stock,
            index = ['trade_date', 'code']
        )
        return data

    @classmethod
    def plate_info(cls, start: str = None, end: str = None, 
        code: 'list | str' = None, fields: list = None,
        conditions: 'str | list' = None) -> pd.DataFrame:
        '''get plate info in daily frequency
        -------------------------------------

        start: datetime or date or str, start date in 3 forms
        end: datetime or date or str, end date in 3 forms
        fields: list, the field names you want to get
        conditions: list, a series of conditions like "code = '000001.SZ'" listed in a list
        '''
        data = cls._get_panel_data(
            start = start,
            end = end,
            date_col = 'trade_date',
            code = code,
            code_col = "code",
            fields = fields,
            conditions = conditions,
            table = 'plate_info',
            database = cls.stock,
            index = ['trade_date', 'code']
        )
        return data

    @classmethod
    def derivative_indicator(cls, start: str = None, end: str = None,
        code: 'str | list' = None, fields: list = None,
        conditions: 'str | list' = None) -> pd.DataFrame:
        '''get derivative indicator in daily frequecy
        ---------------------------------------------

        start: datetime or date or str, start date in 3 forms
        end: datetime or date or str, end date in 3 forms
        fields: list, the field names you want to get
        conditions: list, a series of conditions like "code = '000001.SZ'" listed in a list
        '''
        data = cls._get_panel_data(
            start = start,
            end = end,
            date_col = 'trade_date',
            code = code,
            code_col = "wind_code",
            fields = fields,
            conditions = conditions,
            table = 'derivative_indicator',
            database = cls.stock,
            index = ['trade_date', 'code']
        )
        return data
    
    @classmethod
    def active_opdep(cls, start: 'str' = None, end: 'str' = None,
        code: 'list | str' = None, fields: list = None,
        conditions: 'str | list' = None) -> pd.DataFrame:
        """
        Get Longhubang data
        -------------------

        date: str, the date
        """
        data = cls._get_panel_data(
            start = start,
            end = end,
            date_col = "onlist_date",
            code = code,
            code_col = "buy_stock_code",
            fields = fields,
            conditions = conditions,
            table = 'plate_info',
            database = cls.stock,
            index = ['trade_date', 'code']
        )
        return data

    @classmethod
    def active_opdep_react_stock(cls, date: str, code: str, gap: int) -> datetime.datetime:
        """Get the reactivating date for a stock from active_opdep for a given date
        ------------------------------------------------------------------------------

        date: datetime, date
        code: str, stock code
        gap: int, the days gap at least between the two activating date
        return: datetime, return the date
        """
        date = time2str(date)
        sql = f'select opdep_abbrname, onlist_date, buy_stock_code, buy_stock_name ' \
            f'from active_opdep ' \
            f'where buy_stock_code like "%%{code}%%" ' \
            f'and onlist_date <= "{date}" ' \
            f'order by onlist_date desc'
        data = pd.read_sql(sql, cls.stock)
        date = data['onlist_date'].to_frame()
        date['previous'] = data['onlist_date'].shift(periods=-1)
        date['diff'] = date['onlist_date'] - date['previous']
        early_date = ''
        for i in range(len(date)):
            if date.iloc[i, :]['diff'] > datetime.timedelta(days=gap):
                early_date = date.iloc[i, :]['onlist_date'];break
        early_date = str2time(early_date)
        return early_date
        
    @classmethod
    def active_opdep_react_plate(cls, date: str, plate: str, gap: int) -> datetime:
        """Get the reactivating date for a plate from active_opdep for a given date
        ---------------------------------------------------------------------------

        date: datetime, the given date
        plate: str, industry or plate
        gap: int, the days gap at least between the two activating date
        return: datetime, return the date
        """
        date = time2str(date)
        sql = f'select * from active_opdep_plates ' \
            f'where plates like "%%{plate}%%" ' \
            f'and trade_date <= "{date}"' \
            f'order by trade_date desc ' \
            f'limit 1000'
        data = pd.read_sql(sql, cls.stock)
        date = data['trade_date'].to_frame()
        date['previous'] = data['trade_date'].shift(periods=-1)
        date['diff'] = date['trade_date'] - date['previous']
        early_date = ''
        for i in range(len(date)):
            if date.iloc[i, :]['diff'] > datetime.timedelta(days=gap):
                early_date = date.iloc[i, :]['trade_date'];break
        early_date = str2time(early_date)
        return early_date
    
    @classmethod
    def shhk_transaction(cls, start: str = None, end: str = None,
        fields: list = None, conditions: 'str | list' = None) -> pd.DataFrame:
        '''Get north money buy data
        ---------------------------

        start: str, start date
        end: str, end date
        '''
        data = cls._get_panel_data(
            start = start,
            end = end,
            date_col = 'trade_date',
            code = None,
            code_col = None,
            fields = fields,
            conditions = conditions,
            table = 'shhk_transaction',
            database = cls.stock,
            index = ['trade_date']
        )
        return data

    @classmethod
    def szhk_transaction(cls, start: str = None, end: str = None,
        fields: list = None, conditions: 'str | list' = None) -> pd.DataFrame:
        '''Get north money buy data
        ---------------------------

        start: str, start date
        end: str, end date
        '''
        data = cls._get_panel_data(
            start = start,
            end = end,
            date_col = 'trade_date',
            code = None,
            code_col = None,
            fields = fields,
            conditions = conditions,
            table = 'szhk_transaction',
            database = cls.stock,
            index = ['trade_date']
        )
        return data

    @classmethod
    def lgt_holdings_api(cls, start: str = None, end: str = None,
        code: 'str | list' = None, fields: list = None,
        conditions: 'str | list' = None) -> pd.DataFrame:
        '''Get lugutong holdings data from api
        -----------------------------

        start: str, start time,
        end: str, end time,
        code: str, stock code,
        fields: list, fields to be selected
        '''
        data = cls._get_panel_data(
            start = start,
            end = end,
            date_col = 'trade_date',
            code = code,
            code_col = 'wind_code',
            fields = fields,
            conditions = conditions,
            table = 'lgt_holdings_api',
            database = cls.stock,
            index = ['trade_date', 'wind_code']
        )
        return data

    @classmethod
    def lgt_holdings_em(cls, start: str = None, end: str = None, 
        code: 'list | str' = None, fields: list = None,
        conditions: 'str | list' = None) -> pd.DataFrame:
        '''Get lugutong holdings data from em
        -----------------------------

        start: str, start time,
        end: str, end time,
        code: str, stock code,
        fields: list, fields to be selected
        '''
        data = cls._get_panel_data(
            start = start,
            end = end,
            date_col = 'trade_date',
            code = code,
            code_col = 'wind_code',
            fields = fields,
            conditions = conditions,
            table = 'lgt_holdings_em',
            database = cls.stock,
            index = ['trade_date', 'security_code']
        )
        return data

    @classmethod
    def oversea_institution_holding(cls, start: str = None, end: str = None,
        code: 'str | list' = None, fields: list = None,
        conditions: 'str | list' = None) -> pd.DataFrame:
        data = cls._get_panel_data(
            start = start,
            end = end,
            date_col = 'hold_date',
            code = code,
            code_col = 'wind_code',
            fields = fields,
            conditions = conditions,
            table = 'oversea_institution_holding',
            database = cls.stock,
            index = ['hold_date', 'secucode']
        )
        return data

    @classmethod
    def dividend(cls, start: str = None, end = None, 
        code: 'str | list' = None, fields: list = None,
        conditions: 'str | list' = None) -> pd.DataFrame:
        data = cls._get_panel_data(
            start = start,
            end = end,
            date_col = 'trade_date',
            code = code,
            code_col = 'code',
            fields = fields,
            conditions = conditions,
            table = 'dividend',
            database = cls.stock,
            index = ['trade_date', 'code']
        )
        return data 

    @classmethod
    def balance_sheet(cls, start: str = None, end = None,
        code: 'str | list' = None, fields: list = None,
        conditions: 'str | list' = None) -> pd.DataFrame:
        data = cls._get_panel_data(
            start = start,
            end = end,
            date_col = 'report_period',
            code = code,
            code_col = 'code',
            fields = fields,
            conditions = conditions,
            table = 'balance_sheet',
            database = cls.stock,
            index = ['report_period', 'code']
        )
        return data

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


if __name__ == '__main__':
    # fetcher = StockUS("guflrppo3jct4mon7kw13fmv3dsz9kf2")
    # price = fetcher.cn_price('000001.SZ', '20100101', '20200101')
    # print(price)
    
    database = Database('kali', 'kali123')
    data = database.balance_sheet(code=('000001.SZ', '000002.SZ'), start='20210101', 
        end='20211231', fields=['code', 'report_period', 'acct_rcv'],
        conditions=['statement_type = 408001000'])
    print(data.describe())

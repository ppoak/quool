import os
import re
import sys
import datetime
import akshare as ak
import pandas as pd
import sqlalchemy as sql
from ..tools import *
from functools import lru_cache


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
    def read_csv_directory(path, perspective: str, name_pattern: str = None, **kwargs):
        '''A enhanced function for reading files in a directory to a panel DataFrame
        ----------------------------------------------------------------------------

        path: path to the directory
        perspective: 'datetime', 'asset', 'indicator'
        name_pattern: pattern to match the file name, which will be extracted as index
        kwargs: other arguments for pd.read_csv

        **note: the name of the file in the directory will be interpreted as the 
        sign(column or index) to the data, so set it to the brief one
        '''
        files = sorted(os.listdir(path))
        datas = []
        
        if perspective == "indicator":
            if name_pattern is None:
                name_pattern = r'.*'
            for file in files:
                basename = os.path.splitext(file)[0]
                name = re.findall(name_pattern, basename)[0]
                data = pd.read_csv(os.path.join(path, file), **kwargs)
                data = data.stack()
                data.name = name
                datas.append(data)
            datas = pd.concat(datas, axis=1).sort_index()

        elif perspective == "asset":
            if name_pattern is None:
                name_pattern = r'[a-zA-Z\d]{6}\.[a-zA-Z]{2}|[a-zA-Z]{0,2}\..{6}'
            for file in files:
                basename = os.path.splitext(file)[0]
                name = re.findall(name_pattern, basename)[0]
                data = pd.read_csv(os.path.join(path, file), **kwargs)
                data.index = pd.MultiIndex.from_product([data.index, [name]])
                datas.append(data)
            datas = pd.concat(datas).sort_index()
            
        elif perspective == "datetime":
            if name_pattern is None:
                name_pattern = r'\d{4}[./-]\d{2}[./-]\d{2}|\d{4}[./-]\d{2}[./-]\d{2}\s?\d{2}[:.]\d{2}[:.]\d{2}'
            for file in files:
                basename = os.path.splitext(file)[0]
                name = re.findall(name_pattern, basename)[0]
                data = pd.read_csv(os.path.join(path, file), **kwargs)
                data.index = pd.MultiIndex.from_product([pd.to_datetime([name]), data.index])
                datas.append(data)
            datas = pd.concat(datas).sort_index()
        
        return datas

    @staticmethod
    def read_excel_directory(path, perspective: str, name_pattern: str = None, **kwargs):
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
            if name_pattern is None:
                name_pattern = r'.*'
            for file in files:
                basename = os.path.splitext(file)[0]
                name = re.findall(name_pattern, basename)[0]
                data = pd.read_excel(os.path.join(path, file), **kwargs)
                data = data.stack()
                data.name = name
                datas.append(data)
            datas = pd.concat(datas, axis=1).sort_index()

        elif perspective == "asset":
            if name_pattern is None:
                name_pattern = r'[a-zA-Z\d]{6}\.[a-zA-Z]{2}|[a-zA-Z]{0,2}\..{6}'
            for file in files:
                basename = os.path.splitext(file)[0]
                name = re.search(name_pattern, basename).group()
                data = pd.read_excel(os.path.join(path, file), **kwargs)
                data.index = pd.MultiIndex.from_product([data.index, [name]])
                datas.append(data)
            datas = pd.concat(datas).sort_index()
            
        elif perspective == "datetime":
            if name_pattern is None:
                name_pattern = r'\d{4}[./-]\d{2}[./-]\d{2}|\d{4}[./-]\d{2}[./-]\d{2}\s?\d{2}[:.]\d{2}[:.]\d{2}'
            for file in files:
                basename = os.path.splitext(file)[0]
                name = re.search(name_pattern, basename).group()
                data = pd.read_excel(os.path.join(path, file), **kwargs)
                data.index = pd.MultiIndex.from_product([pd.to_datetime([name]), data.index])
                datas.append(data)
            datas = pd.concat(datas).sort_index()
        
        return datas

    def to_parquet(self, path, append: bool = True, 
        compression: str = 'snappy', index: bool = None, **kwargs):
        '''A enhanced function enables parquet file appending to existing file
        -----------------------------------------------------------------------
        
        path: path to the parquet file
        append: whether to append to the existing file
        compression: compression method
        index: whether to write index to the parquet file
        '''
        import pyarrow

        if append:
            original_data = pd.read_parquet(path)
            data = pd.concat([original_data, self.data])
            data = pyarrow.Table.from_pandas(data)
            with pyarrow.parquet.ParquetWriter(path, data.schema, compression=compression) as writer:
                writer.write_table(table=data)
        else:
            self.data.to_parquet(path, compression=compression, index=index, **kwargs)
        
class Database(object):

    def __init__(self, user: str, password: str):
        Database.user = user
        Database.password = password

        base = "mysql+pymysql://{user}:{password}@127.0.0.1/{database}?charset=utf8"
        Database.stock = sql.create_engine(base.format(user=user, password=password, database="stock"),
            poolclass=sql.pool.NullPool, connect_args={"charset": "utf8", "connect_timeout": 10})
        Database.fund = sql.create_engine(base.format(user=user, password=password, database="fund"),
            poolclass=sql.pool.NullPool, connect_args={"charset": "utf8", "connect_timeout": 10})
        Database.factor = sql.create_engine(base.format(user=user, password=password, database="factor"),
            poolclass=sql.pool.NullPool, connect_args={"charset": "utf8", "connect_timeout": 10})
        
        today = datetime.datetime.today()
        if today.hour > 20:
            Database.today = (datetime.datetime.today() - datetime.timedelta(days=1)).strftime(r'%Y%m%d')
        else:
            Database.today = today.strftime(r'%Y%m%d')
    
    @classmethod
    def _get_panel_data(cls, 
        start: str, 
        end: str, 
        date_col: str,
        code: 'str | list', 
        code_col: str, 
        fields: list, 
        conditions: 'str | list', 
        conditions_: 'str | list',
        table: str,
        database, index: list) -> pd.DataFrame:
        
        start = start or '20100101'
        end = end or cls.today
        code = item2list(code)
        conditions = item2list(conditions)
        conditions_ = item2list(conditions_)
        
        if fields:
            fields = ','.join(fields)
        else:
            fields = '*'
        
        query = f'select {fields} from {table}'
        query += f' where ' if any([date_col, code, conditions, conditions_]) else ''
        
        if date_col:
            query += f' ( {date_col} between "{start}" and "{end}" )'
            
        if code:
            query += ' and ' if any([date_col]) else ''
            query += '(' + ' or '.join([rf'{code_col} like "%%{c}%%"' for c in code]) + ')'
        
        if conditions:
            query += ' and ' if any([date_col, code]) else ''
            query += '(' + ' and '.join(conditions) + ')'
        
        if conditions_:
            query += ' and ' if any([date_col, code, conditions]) else ''
            query += '(' + ' or '.join(conditions_) + ')'
        
        data = pd.read_sql(query, database)
        
        if index:
            index = list(filter(lambda x: x in fields, index))
            data = data.set_index(index)
        
        return data

    @classmethod
    def trade_date(cls, start: str = None,
        end: str = None, freq: str = 'daily',
        weekday: bool = False, month: bool = False) -> pd.DataFrame:
        '''get trade date during a period
        ---------------------------------

        start: datetime or str
        end: datetime or date or str, end date in 3 forms
        freq: str, frequency in either 'daily', 'weekly' or 'monthly'
        '''
        table = f'trade_date_{freq}'
        fields = ['trade_date']
        fields += ['weekday'] if weekday else []
        fields += ['month'] if month else []
        data = cls._get_panel_data(
            start = start,
            end = end,
            date_col = 'trade_date',
            code = None,
            code_col = None,
            fields = fields,
            conditions = None,
            conditions_ = None,
            table = table,
            database = cls.stock,
            index = None
        )
        return data

    @classmethod
    def nearby_n_trade_date(cls, date: str, n: int) -> datetime.datetime:
        '''Get last n trading dates
        -------------------------------------
        
        date: str, datetime or date, the given date
        n: int, the number of dates around the given date,
            negative value means before
        return: list, a list of dates
        '''
        cover_date = str2time(date) + datetime.timedelta(days=n * 9 + 2)
        if n < 0:
            dates = cls.trade_date(cover_date, date)
        else:
            dates = cls.trade_date(date, cover_date)
        if len(dates) < abs(n + 1):
            return None
        elif n > 0:
            return dates[n]
        else:
            return dates[n - 1]
    
    @classmethod
    def index_hs300_close_weight(cls, start: str = None,
        end: str = None, code: 'str | list' = None,
        fields: 'list | str' = None,
        conditions: 'str | list' = None,
        conditions_: 'str | list' = None) -> pd.DataFrame:
        '''Get HS300 weight when market is closed'''
        data = cls._get_panel_data(
            start = start,
            end = end,
            date_col = 'trade_dt',
            code = code,
            code_col = 's_info_windcode',
            fields = fields,
            conditions = conditions,
            conditions_ = conditions_,
            index = ['trade_dt', 's_info_windcode'],
            table = 'index_hs300_close_weight',
            database = cls.stock,
        )
        return data
    
    @classmethod
    def market_daily(cls, start: str = None, end: str = None,
        code: 'str | list' = None, fields: list = None,
        conditions: 'str | list' = None, 
        conditions_: 'str | list' = None) -> pd.DataFrame:
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
            conditions_ = conditions_,
            table = 'market_daily',
            database = cls.stock,
            index = ['trade_date', 'code']
        )
        return data
    
    @classmethod
    def index_description(cls, code: 'str | list' = None, 
        fields: 'str | list' = None, conditions: 'str | list' = None, 
        conditions_: 'str | list' = None) -> pd.DataFrame:
        '''get index description data
        -------------------------------------

        start: datetime or date or str, start date in 3 forms
        end: datetime or date or str, end date in 3 forms
        code: str, the index code
        fields: list, the field names you want to get
        conditions: list, a series of conditions like "code = '000001.SZ'" listed in a list
        '''
        data = cls._get_panel_data(
            start = None,
            end = None, 
            date_col = None,
            code = code,
            code_col = 's_info_windcode',
            fields = fields,
            conditions = conditions,
            conditions_ = conditions_,
            table = 'index_description',
            database = cls.stock,
            index = ['s_info_windcode']
        )
        return data

    @classmethod
    def index_market_daily(cls, start: str = None, end: str = None,
        code: 'str | list' = None, fields: list = None,
        conditions: 'str | list' = None,
        conditions_: 'str | list' = None) -> pd.DataFrame:
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
            conditions_ = conditions_,
            table = 'index_market_daily',
            database = cls.stock,
            index = ['trade_date', 'index_code']
        )
        return data

    @classmethod
    def plate_info(cls, start: str = None, end: str = None, 
        code: 'list | str' = None, fields: list = None,
        conditions: 'str | list' = None,
        conditions_: 'str | list' = None) -> pd.DataFrame:
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
            conditions_ = conditions_,
            table = 'plate_info',
            database = cls.stock,
            index = ['trade_date', 'code']
        )
        return data

    @classmethod
    def derivative_indicator(cls, start: str = None, end: str = None,
        code: 'str | list' = None, fields: list = None,
        conditions: 'str | list' = None,
        conditions_: 'str | list' = None) -> pd.DataFrame:
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
            conditions_ = conditions_,
            table = 'derivative_indicator',
            database = cls.stock,
            index = ['trade_date', 'code']
        )
        return data
    
    @classmethod
    def active_opdep(cls, start: 'str' = None, end: 'str' = None,
        code: 'list | str' = None, fields: list = None,
        conditions: 'str | list' = None,
        conditions_: 'str | list' = None) -> pd.DataFrame:
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
            conditions_ = conditions_,
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
        fields: list = None, conditions: 'str | list' = None,
        conditions_: 'str | list' = None) -> pd.DataFrame:
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
            conditions_ = conditions_,
            table = 'shhk_transaction',
            database = cls.stock,
            index = ['trade_date']
        )
        return data

    @classmethod
    def szhk_transaction(cls, start: str = None, end: str = None,
        fields: list = None, conditions: 'str | list' = None,
        conditions_: 'str | list' = None) -> pd.DataFrame:
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
            conditions_ = conditions_,
            table = 'szhk_transaction',
            database = cls.stock,
            index = ['trade_date']
        )
        return data

    @classmethod
    def lgt_holdings_api(cls, start: str = None, end: str = None,
        code: 'str | list' = None, fields: list = None,
        conditions: 'str | list' = None,
        conditions_: 'str | list' = None) -> pd.DataFrame:
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
            conditions_ = conditions_,
            table = 'lgt_holdings_api',
            database = cls.stock,
            index = ['trade_date', 'wind_code']
        )
        return data

    @classmethod
    def lgt_holdings_em(cls, start: str = None, end: str = None, 
        code: 'list | str' = None, fields: list = None,
        conditions: 'str | list' = None,
        conditions_: 'str | list' = None) -> pd.DataFrame:
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
            conditions_ = conditions_,
            table = 'lgt_holdings_em',
            database = cls.stock,
            index = ['trade_date', 'security_code']
        )
        return data

    @classmethod
    def oversea_institution_holding(cls, start: str = None, end: str = None,
        code: 'str | list' = None, fields: list = None,
        conditions: 'str | list' = None,
        conditions_: 'str | list' = None) -> pd.DataFrame:
        data = cls._get_panel_data(
            start = start,
            end = end,
            date_col = 'hold_date',
            code = code,
            code_col = 'wind_code',
            fields = fields,
            conditions = conditions,
            conditions_ = conditions_,
            table = 'oversea_institution_holding',
            database = cls.stock,
            index = ['hold_date', 'secucode']
        )
        return data

    @classmethod
    def dividend(cls, start: str = None, end = None, 
        code: 'str | list' = None, fields: list = None,
        conditions: 'str | list' = None,
        conditions_: 'str | list' = None,) -> pd.DataFrame:
        data = cls._get_panel_data(
            start = start,
            end = end,
            date_col = 'trade_date',
            code = code,
            code_col = 'code',
            fields = fields,
            conditions = conditions,
            conditions_ = conditions_,
            table = 'dividend',
            database = cls.stock,
            index = ['trade_date', 'code']
        )
        return data 

    @classmethod
    def balance_sheet(cls, start: str = None, end = None,
        code: 'str | list' = None, fields: list = None,
        conditions: 'str | list' = None,
        conditions_: 'str | list' = None) -> pd.DataFrame:
        data = cls._get_panel_data(
            start = start,
            end = end,
            date_col = 'report_period',
            code = code,
            code_col = 'code',
            fields = fields,
            conditions = conditions,
            conditions_ = conditions_,
            table = 'balance_sheet',
            database = cls.stock,
            index = ['report_period', 'code']
        )
        return data

    @classmethod
    def income_sheet(cls, start: str = None, end = None,
        code: 'str | list' = None, fields: list = None,
        conditions: 'str | list' = None,
        conditions_: 'str | list' = None) -> pd.DataFrame:
        data = cls._get_panel_data(
            start = start,
            end = end,
            date_col = 'report_period',
            code = code,
            code_col = 'code',
            fields = fields,
            conditions = conditions,
            conditions_ = conditions_,
            table = 'income_sheet',
            database = cls.stock,
            index = ['report_period', 'code']
        )
        return data

    @classmethod
    def cashflow_sheet(cls, start: str = None, end = None,
        code: 'str | list' = None, fields: list = None,
        conditions: 'str | list' = None,
        conditions_: 'str | list' = None) -> pd.DataFrame:
        data = cls._get_panel_data(
            start = start,
            end = end,
            date_col = 'report_period',
            code = code,
            code_col = 'code',
            fields = fields,
            conditions = conditions,
            conditions_ = conditions_,
            table = 'cashflow_sheet',
            database = cls.stock,
            index = ['report_period', 'code']
        )
        return data

@pd.api.extensions.register_dataframe_accessor("databaser")
@pd.api.extensions.register_series_accessor("databaser")
class Databaser(Worker):

    def __sql_cols(self, data, usage="sql"):
        '''internal usage: get sql columns from dataframe df'''
        cols = tuple(data.columns)
        if usage == "sql":
            cols_str = str(cols).replace("'", "`")
            if len(data.columns) == 1:
                cols_str = cols_str[:-2] + ")"  # to process dataframe with only one column
            return cols_str
        elif usage == "format":
            base = "'%%(%s)s'" % cols[0]
            for col in cols[1:]:
                base += ", '%%(%s)s'" % col
            return base
        elif usage == "values":
            base = "%s=VALUES(%s)" % (cols[0], cols[0])
            for col in cols[1:]:
                base += ", `%s`=VALUES(`%s`)" % (col, col)
            return base
        elif usage == "excluded":
            base = "%s=excluded.%s" % (cols[0], cols[0])
            for col in cols[1:]:
                base += ', `%s`=excluded.`%s`' % (col, col)
            return base

    def to_sql(self, table: str, database: sql.engine.base.Engine, index: bool = True,
        on_duplicate: str = "update", chunksize: int = 2000):
        """Save current dataframe to database, only support for mysql and sqlite
        ------------------------------------------------------------------------

        table: str, table to insert data;
        database: Sqlalchemy Engine object
        kind: str, optional {"update", "replace", "ignore"}, default "update" specified the way to update
            "update": "INSERT ... ON DUPLICATE UPDATE ...", 
            "replace": "REPLACE ...",
            "ignore": "INSERT IGNORE ..."
        chunksize: int, size of records to be inserted each time;
        """
        # we should ensure data is in a frame form and no index can be assigned
        data = self.data.copy()
        if not self.is_frame:
            data = data.to_frame()
        if index:
            if isinstance(self.data.index, pd.MultiIndex):
                index_col = '(`' + '`, `'.join(self.data.index.names) + '`)'
            else:
                index_col = f'(`{self.data.index.name}`)'
            if data.index.has_duplicates:
                print('[!] Warning: index has duplicates, will be ignored except the first one')
                data = data[~data.index.duplicated(keep='first')]
            data = data.reset_index()

        engine_type = database.name
        # check whether table exists
        if engine_type == "sqlite":
            with database.connect() as conn:
                check = database.execute("SELECT name FROM sqlite_master"
                    " WHERE type='table' AND name='%s'" % table).fetchall()
        elif engine_type == "mysql": 
            with database.connect() as conn:
                check = database.execute("SHOW TABLES LIKE '%s'" % table).fetchall()       
        
        # if table does not exist, create table
        if not check:
            data.to_sql(table, database, index=False)
            if engine_type == "mysql":
                with database.connect() as conn:
                    conn.execute("ALTER TABLE %s ADD  (%s)" % (table, index_col))
            elif engine_type == "sqlite":
                # unluckily sqlite3 donesn't support alter table, 
                # so we have to drop and recreate
                # https://www.yiibai.com/sqlite/primary-key.html
                with database.connect() as conn:
                    sql_create = conn.execute("SELECT sql FROM sqlite_master WHERE tbl_name = '%s'"
                         % table).fetchall()[0][0]
                    conn.execute("ALTER TABLE %s RENAME TO %s_temp" % (table, table))
                    sql_create_index = sql_create.replace(")", ", PRIMARY KEY %s)" % index_col)
                    conn.execute(sql_create_index)
                    conn.execute("INSERT INTO %s SELECT * FROM %s_temp" % (table, table))
                    conn.execute("DROP TABLE %s_temp" % table)
            return
        
        table = ".".join(["`" + x + "`" for x in table.split(".")])

        data = data.fillna("None")
        data = data.applymap(lambda x: re.sub(r'([\'\"\\])', '\\\g<1>', str(x)))
        cols_str = self.__sql_cols(data)
        for i in range(0, len(data), chunksize):
            # print("chunk-{no}, size-{size}".format(no=str(i/chunksize), size=chunksize))
            tmp = data[i: i + chunksize]

            if on_duplicate == "replace":
                if engine_type == 'mysql':
                    sql_base = f"REPLACE INTO {table} {cols_str}"
                elif engine_type == 'sqlite':
                    sql_base = f"INSERT OR REPLACE INTO {table} {cols_str}"

            elif on_duplicate == "update":
                sql_base = f"INSERT INTO {table} {cols_str}"
                if engine_type == "mysql":
                    sql_update = f" ON DUPLICATE KEY UPDATE {self.__sql_cols(tmp, 'values')}"
                elif engine_type == "sqlite":
                    sql_update = f" ON CONFLICT DO UPDATE SET {self.__sql_cols(tmp, 'excluded')}"

            elif on_duplicate == "ignore":
                if engine_type == "mysql":
                    sql_base = f"INSERT IGNORE INTO {table} {cols_str}"
                elif engine_type == "sqlite":
                    sql_base = f"INSERT OR IGNORE INTO {table} {cols_str}"

            sql_val = self.__sql_cols(tmp, "format")
            vals = tuple([sql_val % x for x in tmp.to_dict("records")])
            sql_vals = " VALUES ({x})".format(x=vals[0])
            for i in range(1, len(vals)):
                sql_vals += ", ({x})".format(x=vals[i])
            sql_vals = sql_vals.replace("'None'", "NULL")

            sql_main = sql_base + sql_vals
            if on_duplicate == "update":
                sql_main += sql_update

            if sys.version_info.major == 2:
                sql_main = sql_main.replace("u`", "`")
            if sys.version_info.major == 3:
                sql_main = sql_main.replace("%", "%%")
            with database.connect() as conn:
                conn.execute(sql_main)
        
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
    import numpy as np
    data = pd.DataFrame(np.array([['cc', 'dd', 'czxvvx', 'd', 'e'], ['cc', 'b', 'ee', 'd', 'e']]).T,
        columns=['name', 'addr'], index=[100, 101, 102, 103, 106])
    data.index.name = 'id'
    conn = sql.create_engine('sqlite:///./test.db')
    # conn = sql.create_engine('mysql+pymysql://windreader:password1@221.226.96.114:10087/wind')
    data.databaser.to_sql('c', conn, on_duplicate='replace')
    
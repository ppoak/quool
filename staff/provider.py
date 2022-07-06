import datetime
import pandas as pd
import sqlalchemy as sql
from ..tools import item2list, str2time


class Stock:

    def __init__(self, database: sql.engine.Engine) -> None:
        self.today = datetime.datetime.today().isoformat()
        self.engine = database
        
    def __query(self, table: str, start: str, end: str, date_col: str, code: 'str | list',
        code_col: str, fields: 'list | str', index_col: 'str | list', and_: 'list | str', or_: 'str | list'):
        start = start or '20070104'
        end = end or self.today
        start = str2time(start)
        end = str2time(end)
        code = item2list(code)
        fields = item2list(fields)
        index_col = item2list(index_col)
        and_ = item2list(and_)
        or_ = item2list(or_)

        if fields:
            fields = set(fields).union(set(index_col))
            fields = ','.join(fields)
        else:
            fields = '*'

        query = f'select {fields} from {table}'
        query += f' where ' if any([date_col, code, and_, or_]) else ''
        
        if date_col:
            query += f' ( {date_col} between "{start}.000000" and "{end}.000000" )'
            
        if code:
            query += ' and ' if any([date_col]) else ''
            query += '(' + ' or '.join([f'{code_col} like "%%{c}%%"' for c in code]) + ')'
        
        if and_:
            query += ' and ' if any([date_col, code]) else ''
            query += '(' + ' and '.join(and_) + ')'
        
        if or_:
            query += ' and ' if any([date_col, code, and_]) else ''
            query += '(' + ' or '.join(or_) + ')'
        
        return query

    def get(self, 
        table: str,
        start: str = None, 
        end: str = None,
        date_col: str = 'trading_date',
        code: 'str | list' = None,
        code_col: str = 'wind_code',
        fields: list = None, 
        index_col: 'str | list' = None,
        and_: 'str | list' = None, 
        or_: 'str | list' = None,
        ) -> pd.DataFrame:
        query = self.__query(
            table,
            start,
            end,
            date_col,
            code,
            code_col,
            fields,
            index_col,
            and_,
            or_
        )
        data = pd.read_sql(query, self.engine, 
            parse_dates=['trading_date', 'report_period', 'date', 'datetime',
                'trading_index', 'trading_week', 'trading_month', 'month'])
        if index_col:
            data = data.set_index(index_col)
        return data
    
    def market_daily(self, start: str = None, end: str = None,
        code: 'str | list' = None, fields: list = None,
        and_: 'str | list' = None, 
        or_: 'str | list' = None) -> pd.DataFrame:
        '''get market data in daily frequency
        -------------------------------------

        start: datetime or date or str, start date in 3 forms
        end: datetime or date or str, end date in 3 forms
        fields: list, the field names you want to get
        '''
        return self.get(
            table = 'market_daily',
            start = start,
            end = end,
            date_col = "trading_date",
            code = code,
            code_col = "wind_code",
            fields = fields,
            index_col = ['trading_date', 'wind_code'],
            and_ = and_,
            or_ = or_,
        )
    
    def index_market_daily(self, start: str = None, end: str = None,
        code: 'str | list' = None, fields: list = None,
        and_: 'str | list' = None,
        or_: 'str | list' = None) -> pd.DataFrame:
        '''get index market data in daily frequency
        -------------------------------------

        start: datetime or date or str, start date in 3 forms
        end: datetime or date or str, end date in 3 forms
        code: str, the index code
        fields: list, the field names you want to get
        conditions: list, a series of conditions like "code = '000001.SZ'" listed in a list
        '''
        return self.get(
            table = 'index_market_daily',
            start = start,
            end = end,
            date_col = "trading_date",
            code = code,
            code_col = "index_code",
            fields = fields,
            index_col = ['trading_date', 'index_code'],
            and_ = and_,
            or_ = or_,
        )
  
    def plate_info(self, start: str = None, end: str = None, 
        code: 'list | str' = None, fields: list = None,
        and_: 'str | list' = None,
        or_: 'str | list' = None) -> pd.DataFrame:
        '''get plate info in daily frequency
        -------------------------------------

        start: datetime or date or str, start date in 3 forms
        end: datetime or date or str, end date in 3 forms
        fields: list, the field names you want to get
        conditions: list, a series of conditions like "code = '000001.SZ'" listed in a list
        '''
        return self.get(
            table = 'plate_info',
            start = start,
            end = end,
            date_col = 'date',
            code = code,
            code_col = "code",
            fields = fields,
            index_col = ['date', 'wind_code'],
            and_ = and_,
            or_ = or_,
        )

    def derivative_indicator(self, start: str = None, end: str = None,
        code: 'str | list' = None, fields: list = None,
        and_: 'str | list' = None,
        or_: 'str | list' = None) -> pd.DataFrame:
        '''get derivative indicator in daily frequecy
        ---------------------------------------------

        start: datetime or date or str, start date in 3 forms
        end: datetime or date or str, end date in 3 forms
        fields: list, the field names you want to get
        conditions: list, a series of conditions like "code = '000001.SZ'" listed in a list
        '''
        return self.get(
            table = 'derivative_indicator',
            start = start,
            end = end,
            date_col = 'trading_date',
            code = code,
            code_col = "wind_code",
            fields = fields,
            index_col = ['trading_date', 'wind_code'],
            and_ = and_,
            or_ = or_,
        )

    def balance_sheet(self, start: str = None, end = None,
        code: 'str | list' = None, fields: list = None,
        and_: 'str | list' = None,
        or_: 'str | list' = None) -> pd.DataFrame:
        data = self.get(
            table = 'balance_sheet',
            start = start,
            end = end,
            date_col = 'report_period',
            code = code,
            code_col = 'wind_code',
            fields = fields,
            index_col = ['report_period', 'wind_code'],
            and_ = and_,
            or_ = or_,
        )
        return data

    def income_sheet(self, start: str = None, end = None,
        code: 'str | list' = None, fields: list = None,
        and_: 'str | list' = None,
        or_: 'str | list' = None) -> pd.DataFrame:
        data = self.get(
            table = 'income_sheet',
            start = start,
            end = end,
            date_col = 'report_period',
            code = code,
            code_col = 'wind_code',
            fields = fields,
            index_col = ['report_period', 'wind_code'],
            and_ = and_,
            or_ = or_,
        )
        return data

    def cashflow_sheet(self, start: str = None, end = None,
        code: 'str | list' = None, fields: list = None,
        and_: 'str | list' = None,
        or_: 'str | list' = None) -> pd.DataFrame:
        data = self.get(
            table = 'cashflow_sheet',
            start = start,
            end = end,
            date_col = 'report_period',
            code = code,
            code_col = 'wind_code',
            fields = fields,
            index_col = ['report_period', 'wind_code'],
            and_ = and_,
            or_ = or_,
        )
        return data

    def index_weights(self, start: str = None, end: str = None,
        code: 'str | list' = None, fields: list = None,
        and_: 'str | list' = None,
        or_: 'str | list' = None) -> pd.DataFrame:
        data = self.get(
            table = 'index_weights',
            start = start,
            end = end,
            date_col = 'date',
            code = code,
            code_col = 'index_id',
            fields = fields,
            index_col = ['date', 'index_id', 'order_book_id'],
            and_ = and_,
            or_ = or_,
        )
        return data

    def pit_financial(self, start: str = None, end: str = None,
        code: 'str | list' = None, fields: list = None,
        and_: 'str | list' = None,
        or_: 'str | list' = None) -> pd.DataFrame:
        return self.get(
            table = 'pit_financial',
            start = start,
            end = end,
            date_col = 'trading_date',
            code = code,
            code_col = 'code',
            fields = fields,
            index_col = ['trading_date', 'code'],
            and_ = and_,
            or_ = or_,
        )

    def financial_indicator(self, start: str = None, end: str = None,
        code: 'str | list' = None, fields: list = None,
        and_: 'str | list' = None,
        or_: 'str | list' = None) -> pd.DataFrame:
        return self.get(
            table = 'financial_indicator',
            start = start,
            end = end,
            date_col = 'report_period',
            code = code,
            code_col = 'code',
            fields = fields,
            index_col = ['report_period', 'code'],
            and_ = and_,
            or_ = or_,
        )

    def intensity_trend(self, start: str = None, end: str = None,
        code: 'str | list' = None, fields: list = None,
        and_: 'str | list' = None,
        or_: 'str | list' = None) -> pd.DataFrame:
        return self.get(
            table = 'intensity_trend',
            start = start,
            end = end,
            date_col = 'trading_date',
            code = code,
            code_col = 'code',
            fields = fields,
            index_col = ['trading_date', 'code'],
            and_ = and_,
            or_ = or_,
        )


if __name__ == '__main__':
    pass

import datetime
import pandas as pd
import sqlalchemy as sql
from ..tools import item2list, str2time, CHD


class Data:
    def __init__(self, database: sql.engine.Engine) -> None:
        self.today = datetime.datetime.today()
        self.engine = database
        
    def __query(self, table: str, start: str, end: str, date_col: str, 
        code: 'str | list', code_col: str, fields: 'list | str', 
        index_col: 'str | list', and_: 'list | str', or_: 'str | list'):
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
            query += f' ( {date_col} between "{start}" and "{end}" )'
            
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
        start: str, 
        end: str,
        date_col: str,
        code: 'str | list',
        code_col: str,
        fields: list, 
        index_col: 'str | list',
        and_: 'str | list', 
        or_: 'str | list',
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
        data = pd.read_sql(query, self.engine, parse_dates=date_col)
        if index_col is not None:
            data = data.set_index(index_col)
        return data

class Stock(Data):
    
    def market_daily(self, 
        start: str = None, 
        end: str = None,
        code: 'str | list' = None, 
        fields: list = None,
        and_: 'str | list' = None, 
        or_: 'str | list' = None
    ) -> pd.DataFrame:
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
            date_col = "date",
            code = code,
            code_col = "order_book_id",
            fields = fields,
            index_col = ["date", 'order_book_id'],
            and_ = and_,
            or_ = or_,
        )
  
    def plate_info(self, 
        start: str = None, 
        end: str = None, 
        code: 'list | str' = None, 
        fields: list = None,
        and_: 'str | list' = None,
        or_: 'str | list' = None
    ) -> pd.DataFrame:
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
            code_col = "order_book_id",
            fields = fields,
            index_col = ['date', 'order_book_id'],
            and_ = and_,
            or_ = or_,
        )

    def index_weights(self, 
        start: str = None, 
        end: str = None,
        code: 'str | list' = None, 
        fields: list = None,
        and_: 'str | list' = None,
        or_: 'str | list' = None
    ) -> pd.DataFrame:
        return self.get(
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

    def instruments(self, 
        code: 'str | list' = None, 
        fields: list = None,
        and_: 'str | list' = None, 
        or_: 'str | list' = None
    ) -> pd.DataFrame:
        return self.get(
            table = 'instrument',
            start = None,
            end = None,
            date_col = None,
            code = code,
            code_col = 'order_book_id',
            fields = fields,
            index_col = 'order_book_id',
            and_ = and_,
            or_ = or_,
        )

    def index_market_daily(self, 
        start: str = None, 
        end: str = None,
        code: 'str | list' = None, 
        fields: list = None,
        and_: 'str | list' = None,
        or_: 'str | list' = None
    ) -> pd.DataFrame:
        return self.get(
            table = 'index_market_daily',
            start = start,
            end = end,
            date_col = 'date',
            code = code,
            code_col = 'order_book_id',
            fields = fields,
            index_col = ['date', 'order_book_id'],
            and_ = and_,
            or_ = or_,
        )


if __name__ == '__main__':
    stock = Stock('sqlite:////Users/oak/DataBase/Data/stock.db')
    stock.index_weights(start='20200101', end='20200110', code='000300.XSHG', 
        fields=None).round(4).printer.display(title='test')

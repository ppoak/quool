import datetime
import time
import configparser
import pandas as pd
import pandasquant as pq
import sqlalchemy as sql

Config = configparser.ConfigParser()
Config.read('./pandasquant/config.ini')
engin_type = Config['database']['ENGINE_TYPE']

path, dbname_1 = Config['database']['PATH'], Config['database']['dbname_1']
today = datetime.datetime.today() if datetime.datetime.today().hour >= 22 \
    else datetime.datetime.today() - datetime.timedelta(days=1)
    
if engin_type == 'sqlite':
    stockdb = sql.create_engine("sqlite:///%s/%s" % (path, dbname_1))
    # check whether the trade_date table exists
    is_trade_date_table_exists = stockdb.execute("SELECT name FROM sqlite_master"
        " WHERE type='table' AND name='trade_date'").fetchall()
    if not is_trade_date_table_exists:
        # table not exists
        pq.Api.trade_date(start='20070101', end='20231231').databaser.\
            to_sql('trade_date', stockdb, index=True, on_duplicate=True)
            
    current_trade_dates = pd.read_sql(f"select trading_date from trade_date where trading_date <= '{today}' and trading_date >= '2022-05-10'", 
        stockdb, index_col="trading_date", parse_dates='trading_date').index
    current_report_dates = pd.date_range(start='2007-01-01', end=today, freq='Q')
    
elif engin_type == 'mysql+pymysql':
    stockdb = sql.create_engine("mysql+pymysql://%s/%s?charset=utf8" % (path, dbname_1),
              connect_args={"charset": "utf8", "connect_timeout": 50})
    test_sql = f'SELECT table_name FROM information_schema.tables ' + \
               f'WHERE table_name = "trade_date"'
    try:
        is_trade_date_table_exists = stockdb.execute(test_sql).fetchall()
    except sql.exc.OperationalError:
        is_trade_date_table_exists = []
    if not is_trade_date_table_exists:
        # table not exists
        pq.Api.trade_date(start='20070101', end='20231231').databaser.\
            to_sql('trade_date', stockdb, index=True, on_duplicate=True)

    current_trade_dates = pd.read_sql(f"select trading_date from trade_date where trading_date <= '{today}' and trading_date >= '2022-05-12'", 
        stockdb, index_col="trading_date", parse_dates='trading_date').index
    current_report_dates = pd.date_range(start='2020-01-01', end=today, freq='Q')


tables = {     
    "trade_date": {
        "func": pq.Api.trade_date,
        "date_col": "trading_date",
        "check_date": current_trade_dates,
        "database": stockdb
    },
    "plate_info": {
        "func": pq.Api.plate_info,
        "date_col": "date",
        "check_date": current_trade_dates,
        "database": stockdb
    }, 
    "market_daily": {
        "func": pq.Api.market_daily,
        "date_col": "trading_date",
        "check_date": current_trade_dates,
        "database": stockdb
    }, 
    "index_market_daily": {
        "func": pq.Api.index_market_daily,
        "date_col": "trading_date",
        "check_date": current_trade_dates,
        "database": stockdb
    }, 
    "derivative_indicator": {
        "func": pq.Api.derivative_indicator,
        "date_col": "trading_date",
        "check_date": current_trade_dates,
        "database": stockdb
    }, 
    "pit_financial": {
        "func": pq.Api.pit_financial,
        "date_col": "trading_date",
        "check_date": current_trade_dates,
        "database": stockdb
    }, 
    "balance_sheet": {
        "func": pq.Api.balance_sheet,
        "date_col": "report_period",
        "check_date": current_report_dates,
        "database": stockdb
    }, 
    "cashflow_sheet": {
        "func": pq.Api.cashflow_sheet,
        "date_col": "report_period",
        "check_date": current_report_dates,
        "database": stockdb
    }, 
    "income_sheet": {
        "func": pq.Api.income_sheet,
        "date_col": "report_period",
        "check_date": current_report_dates,
        "database": stockdb
    },
    "index_weight": {
        "func": pq.Api.index_weight,
        "date_col": "date",
        "check_date": current_trade_dates[current_trade_dates >= '20100104'],
        "database": stockdb
    },
    "financial_indicator": {
        "func": pq.Api.financial_indicator,
        "date_col": "report_period",
        "check_date": current_report_dates,
        "database": stockdb
    },
    "intensity_trend": {
        "func": pq.Api.intensity_trend,
        "date_col": "trading_date",
        "check_date": current_trade_dates,
        "database": stockdb,
    }
}

def sqlite_check(conn: sql.engine.base.Engine, table: str, date_col: str):
    # check whether the table exists
    table_status = conn.execute("SELECT name FROM sqlite_master"
        " WHERE type='table' AND name='%s'" % table).fetchall()
    if table_status:
        # table exists, check the date diffrence
        sql = f"select distinct({date_col}) from {table}"
        dates = pd.read_sql(sql, conn, index_col=date_col, parse_dates=True).index
        dates = pd.to_datetime(dates)
        diff = tables[table]['check_date'].difference(dates)
        return diff
    else:
        # table not exists, return the total date
        return tables[table]['check_date']
        
def mysql_check(conn: sql.engine.base.Engine, table: str, date_col: str):
    test_sql = f'SELECT table_name FROM information_schema.tables ' + \
               f'WHERE table_schema = "{conn.url.database}" and table_name = "{table}"'
    table_status = conn.execute(test_sql).fetchone()
    if table_status:
        # table exists, check the date diffrence
        sql = f"select distinct({date_col}) from {table}"
        dates = pd.read_sql(sql, conn, index_col=date_col, parse_dates=True).index
        dates = pd.to_datetime(dates)
        diff = tables[table]['check_date'].difference(dates)
        return diff
    else:
        # table not exists, return the total date
        return tables[table]['check_date']

def save_or_update():
    for table, conf in tables.items():
        print(f'[*] Getting latest data for {table} ...')
        if engin_type == 'sqlite':
            diff = sqlite_check(conf['database'], table, conf['date_col'])
        elif engin_type == 'mysql+pymysql':
            diff = mysql_check(conf['database'], table, conf['date_col'])
        print(f'[*] {len(diff)} rows need to be updated')
        for day in diff:
            print(f'[*] Updating {day} in {table} ...')
            conf['func'](start=day, end=day).databaser.to_sql(
                table=table, database=conf['database'], index=True, on_duplicate="update")
        print(f'[+] Update {table} success')
    
    print(f'[+] All Tables are up to date now')

if __name__ == "__main__":
    save_or_update()

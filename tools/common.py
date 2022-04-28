import re
import sys
import time
import pymysql
import datetime
import pandas as pd


def time2str(date: 'str | datetime.datetime') -> str:
    if isinstance(date, (datetime.datetime, datetime.date)):
        date = date.strftime(r'%Y-%m-%d')
    return date

def str2time(date: 'str | datetime.datetime') -> datetime.datetime:
    if isinstance(date, (str, datetime.date)):
        date = pd.to_datetime(date)
    elif isinstance(date, (float, int)):
        date = pd.to_datetime(str(int(date)))
    return date

def item2list(item) -> list:
    if not isinstance(item, (list, tuple, set, dict)):
        return [item]
    else:
        return item

def to_sql(tb_name: str,
        conn, dataframe, 
        type: str = "update", 
        chunksize: int = 2000, 
        debug: bool = False) -> None:
    '''Dummy of pandas.to_sql, support "REPLACE INTO ..." and "INSERT ... ON DUPLICATE KEY UPDATE (keys) VALUES (values)"
    SQL statement.

    tb_name: str, Table to insert get_data;
    conn: DBAPI Instance
    dataframe: pandas.DataFrame, Dataframe instance
    type: str, optional {"update", "replace", "ignore"}, default "update"
            Specified the way to update get_data. If "update", then `conn` will execute "INSERT ... ON DUPLICATE UPDATE ..."
            SQL statement, else if "replace" chosen, then "REPLACE ..." SQL statement will be executed; else if "ignore" chosen,
            then "INSERT IGNORE ..." will be excuted;
    chunksize: int, Size of records to be inserted each time;
    return: None
    '''
    def _sql_cols(df, usage="sql"):
        cols = tuple(df.columns)
        if usage == "sql":
            cols_str = str(cols).replace("'", "`")
            if len(df.columns) == 1:
                # to process dataframe with only one column
                cols_str = cols_str[:-2] + ")"
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

    tb_name = ".".join(["`" + x + "`" for x in tb_name.split(".")])

    df = dataframe.copy(deep=False)
    df = df.fillna("None")
    df = df.applymap(lambda x: re.sub('([\'\"\\\])', '\\\\\g<1>', str(x)))
    cols_str = _sql_cols(df)
    sqls = []
    for i in range(0, len(df), chunksize):
        # print("chunk-{no}, size-{size}".format(no=str(i/chunksize), size=chunksize))
        df_tmp = df[i: i + chunksize]

        if type == "replace":
            sql_base = "REPLACE INTO {tb_name} {cols}".format(
                tb_name=tb_name,
                cols=cols_str
            )

        elif type == "update":
            sql_base = "INSERT INTO {tb_name} {cols}".format(
                tb_name=tb_name,
                cols=cols_str
            )
            sql_update = "ON DUPLICATE KEY UPDATE {0}".format(
                _sql_cols(df_tmp, "values")
            )

        elif type == "ignore":
            sql_base = "INSERT IGNORE INTO {tb_name} {cols}".format(
                tb_name=tb_name,
                cols=cols_str
            )

        sql_val = _sql_cols(df_tmp, "format")
        vals = tuple([sql_val % x for x in df_tmp.to_dict("records")])
        sql_vals = "VALUES ({x})".format(x=vals[0])
        for i in range(1, len(vals)):
            sql_vals += ", ({x})".format(x=vals[i])
        sql_vals = sql_vals.replace("'None'", "NULL")

        sql_main = sql_base + sql_vals
        if type == "update":
            sql_main += sql_update

        if sys.version_info.major == 2:
            sql_main = sql_main.replace("u`", "`")
        if sys.version_info.major == 3:
            sql_main = sql_main.replace("%", "%%")

        if debug is False:
            try:
                conn.execute(sql_main)
            except pymysql.err.InternalError as e:
                print("ENCOUNTERING ERROR: {e}, RETRYING".format(e=e))
                time.sleep(10)
                conn.execute(sql_main)
        else:
            sqls.append(sql_main)
    if debug:
        return sqls


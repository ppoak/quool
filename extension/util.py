import requests
import pandas as pd
import factor as f
import database as d
from retrying import retry


@retry
def get_spot_price() -> pd.DataFrame:
    url = "http://82.push2.eastmoney.com/api/qt/clist/get"
    params = {
        "pn": "1", "pz": "50000", "po": "1", "np": "1", 
        "ut": "bd1d9ddb04089700cf9c27f6f7426281", "fltt": "2", "invt": "2",
        "fid": "f3", "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
        "fields": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152",
        "_": "1623833739532",
    }
    r = requests.get(url, proxies=d.prx.pick('http'), params=params, timeout=2)
    data_json = r.json()
    if not data_json["data"]["diff"]:
        return pd.DataFrame()
    temp_df = pd.DataFrame(data_json["data"]["diff"])
    temp_df.columns = [
        "_", "latest_price", "change_rate", "change_amount", "volume",
        "turnover", "amplitude", "turnover_rate", "pe_ratio_dynamic", 
        "volume_ratio", "five_minute_change", "code", "_", "name", "highest",
        "lowest", "open", "previous_close", "market_cap", "circulating_market_cap", 
        "speed_of_increase", "pb_ratio", "sixty_day_change_rate", 
        "year_to_date_change_rate", "-", "-", "-", "-", "-", "-", "-",
    ]
    
    temp_df = temp_df.dropna(subset=["code"]).set_index("code")
    temp_df = temp_df.drop(["-", "_"], axis=1)
    for col in temp_df.columns:
        if col != 'name':
            temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce')
    return temp_df

def get_spot_return(day: int = 1):
    spot = d.get_spot_price()

    if day <= 1:
        return spot
    
    last_date = f.fqtd.get_trading_days_rollback(rollback=day)
    price = f.fqtd.read("close", start=last_date, stop=last_date)
    price.index = price.index.str.slice(0, 6)
    spot["change_rate"] = (spot["latest_price"] / price - 1).dropna() * 100
    return spot

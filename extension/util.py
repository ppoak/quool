import requests
import squarify
import pandas as pd
import quool as q
import factor as f
import database as d
import matplotlib.pyplot as plt
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

def get_spot_return(day: int = 0):
    spot = get_spot_price()

    if day <= 1:
        return spot
    
    last_date = f.fqtd.get_trading_days_rollback(rollback=day)
    price = f.fqtd.read("close", start=last_date, stop=last_date)
    price.index = price.index.str.slice(0, 6)
    spot["change_rate"] = (spot["latest_price"] / price - 1).dropna() * 100
    return spot

def get_factor_value(fct: f.Factor, name: str, day: int = 0, processor: list = None):
    last_date = f.fqtd.get_trading_days_rollback(rollback=day)
    factor = fct.read(name, start=last_date, stop=last_date, processor=processor).squeeze()
    return factor

def get_holding_spot_return(traderec: q.TradeRecorder):
    def _mapcode(x):
        if x.startswith("6"):
            return x + '.XSHG'
        elif x.startswith("3") or x.startswith("0"):
            return x + '.XSHE'
        else:
            return x

    logger = q.Logger()
    spot = get_spot_price()
    price = spot["latest_price"]
    price.index = price.index.map(_mapcode)
    holdings = traderec.peek(price=price)
    holdings["pnl(%)"] = holdings["pnl"] * 100
    holdings = holdings.drop("pnl", axis=1)
    logger.info(f"Portfolio Value: {holdings['value'].sum():.2f}; "
                f"PNL Value: {holdings['value'].sum() - holdings['amount'].sum():.2f}; "
                f"PNL Rate: {100 *(holdings['value'].sum() / holdings['amount'].sum() - 1):.2f}%")
    return holdings

def treemap(size: pd.Series, nday: int = 0):
    size.index = size.index.str.slice(0, 6)
    size.name = "plotting_size"
    data = pd.concat([size, get_spot_return(day=nday)], axis=1)
    
    bins = [-float('inf'), -20, -15, -10, -8, -7, -6, -5, -4,
        -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, float('inf')]
    labels = [
        "#00FF00", "#00EE00", "#00D000", "#00C000", "#00B000", "#00A000", 
        "#009C00", "#007C00", "#005C00", "#003C00", "#002C00", 
        "#001C00", "#1C0000",
        "#2C0000", "#3C0000", "#5C0000", "#7C0000", "#9C0000", 
        "#A00000", "#B00000", "#C00000", "#D00000", "#EE0000", "#FF0000"
    ]
    data["colors"] = pd.cut(data["change_rate"], bins=bins, labels=labels)
    data = data.dropna(subset=["plotting_size", "change_rate"], axis=0, how='any')
    data = data.sort_values(by="plotting_size", ascending=False)
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    squarify.plot(sizes=data["plotting_size"], color=data["colors"], ax=ax, alpha=0.8)
    ax.xaxis.set_visible(False); ax.yaxis.set_visible(False)
    fig.tight_layout()
    return fig

def scatter(factor: pd.Series, nday: int = 1):
    ret = get_spot_return(day=nday)
    factor.index = factor.index.str.slice(0, 6)
    data = pd.concat([factor, ret["change_rate"]], axis=1)
    pd.plotting.scatter_matrix(data, alpha=0.5, figsize=(20, 20), hist_kwds={"bins": 100})
    fig = plt.gcf()
    fig.tight_layout()
    return fig

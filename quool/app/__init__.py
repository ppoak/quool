try:
    import argparse
    import importlib
    import pandas as pd
    import akshare as ak
    import streamlit as st
    from pathlib import Path
    from quool import ParquetManager
    from quool import Broker as QBroker
    import plotly.subplots as sp
    import plotly.graph_objects as go

except ImportError as e:
    print(e)


def parsearg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', "--refresh", type=int, default=5, help="refresh interval")
    parser.add_argument('-p', "--root", type=str, default="app", help="root path")
    args = parser.parse_args()
    return args

args = parsearg()
REFRESH_INTERVAL = args.refresh
ASSET_PATH = Path(args.root)
TEMPLATE_PATH = Path(ASSET_PATH) / "template"
BROKER_PATH = Path(ASSET_PATH) / "broker"
STRATEGIES_PATH = Path(ASSET_PATH) / "strategy"
LOG_PATH = Path(ASSET_PATH) / "log"


def is_trading_time(time = None):
    time = pd.to_datetime(time or 'now')
    if (
        time.date() in ak.tool_trade_date_hist_sina().squeeze().to_list()
        and (
            (time.time() >= pd.to_datetime("09:30:00").time() and time.time() <= pd.to_datetime("11:30:00").time()) 
            or (time.time() >= pd.to_datetime("13:00:00").time() and time.time() <= pd.to_datetime("15:00:00").time())
        )
    ):
        return True
    return False

class Broker(QBroker):

    def realtime_update(self):
        if is_trading_time():
            self.update(None, fetch_realtime())
            while self._pendings.qsize() > 0:
                self.update(None, fetch_realtime())

def fetch_realtime(prerealtime: pd.DataFrame = None):
    data = ak.stock_zh_a_spot_em().set_index("代码", drop=True).drop(columns="序号")
    if prerealtime is not None:
        data["oepn"] = prerealtime["close"]
        data["high"] = prerealtime["close"].max(prerealtime["high"])
        data["low"] = prerealtime["close"].min(prerealtime["low"])
        data["close"] = data["最新价"]
        data["volume"] = data["成交量"] - prerealtime["成交量"]
    else:
        data["open"] = data["最新价"]
        data["high"] = data["最新价"]
        data["low"] = data["最新价"]
        data["close"] = data["最新价"]
        data["volume"] = data["成交量"]
    return data

def fetch_kline(symbol, format=True):
    data = ak.stock_zh_a_hist(symbol=symbol)
    if format:
        data = data.drop(columns="股票代码")
        data = data.rename(columns={
            "日期": "datetime", "开盘": "open", "最高": "high", 
            "最低": "low", "收盘": "close", "成交量": "volume"
        })
        data["datetime"] = pd.to_datetime(data["datetime"])
        data.set_index("datetime", inplace=True)
    return data

def read_market(begin, end, extra: str = None):
    if not (begin is None and end is None):
        begin = begin or pd.Timestamp("2015-01-01")
        end = end or pd.Timestamp.now()
        return ParquetManager("D:/Documents/DataBase/quotes_day").read(
            date__ge=begin, date__le=end, index=["date", "code"],
            columns=["open", "high", "low", "close", "volume"] + (extra.split(',') or [])
        )
    else:
        return None

@st.fragment(run_every=REFRESH_INTERVAL)
def update_broker():
    broker = st.session_state.get('broker')
    strategy = st.session_state.get('strategy')
    if broker is not None and is_trading_time():
        broker.update(None, fetch_realtime())
        broker.store(BROKER_PATH / f"{broker.brokid}.json")
    elif broker is not None:
        broker.update(None, pd.DataFrame())
        broker.store(BROKER_PATH / f"{broker.brokid}.json")

@st.fragment(run_every=REFRESH_INTERVAL)
def display_realtime():
    st.header("Realtime")
    realtime = fetch_realtime()
    selection = st.dataframe(realtime, selection_mode="single-row", on_select='rerun')
    if selection['selection']["rows"]:
        code = realtime.index[selection['selection']["rows"][0]]
        name = realtime.loc[code, "名称"]
        kline =fetch_kline(symbol=code)
        fig = sp.make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.8, 0.2])
        fig.add_trace(go.Candlestick(
            x=kline.index,
            open=kline.open,
            high=kline.high,
            low=kline.low,
            close=kline.close,
            name=name,
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            x=kline.index, y=kline.volume, name="volume"
        ), row=2, col=1)
        st.plotly_chart(fig)

@st.dialog("Edit your strategy", width="large")
def display_editor(code):
    height = max(len(code.split("\n")) * 20, 68)
    code = st.text_area(label="*edit your strategy*", value=code, height=height)
    if st.button("save", use_container_width=True):
        st.session_state.spath.write_text(code)
        st.rerun()

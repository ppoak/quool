import importlib
import numpy as np
import pandas as pd
import akshare as ak
import streamlit as st
from pathlib import Path
from quool import ParquetManager


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

def raw2ricequant(code: str):
    if code.startswith("6"):
        return code + ".XSHG"
    else:
        return code + ".XSHE"

def fetch_realtime(code_styler: callable = None):
    data = ak.stock_zh_a_spot_em().set_index("代码", drop=True).drop(columns="序号")
    data["open"] = data["最新价"]
    data["high"] = data["最新价"]
    data["low"] = data["最新价"]
    data["close"] = data["最新价"]
    data["volume"] = data["成交量"] * 100
    if code_styler is not None:
        data.index = pd.MultiIndex.from_product([
            [pd.to_datetime("now")], data.index.map(code_styler)
        ], names=["datetime", "code"])
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

def read_market(begin, end, backadj: bool = True, extra: str = None):
    if not (begin is None and end is None):
        begin = begin or pd.Timestamp("2015-01-01")
        end = end or pd.Timestamp.now()
        quotes_day = ParquetManager("D:/Documents/DataBase/quotes_day")
        data = quotes_day.read(
            date__ge=begin, date__le=end, index=["date", "code"],
            columns=["open", "high", "low", "close", "volume"] + (extra.split(',') if extra else [])
        )
        if backadj:
            adj = quotes_day.read(
                date__ge=begin, date__le=end, index=["date", "code"], columns=["adjfactor"]
            )
            data.loc[:, ["open", "high", "low", "close"]] = data[["open", "high", "low", "close"]].mul(adj["adjfactor"], axis=0)
        return data
    else:
        return None

def update_strategy(app_path: str , name: str):
    filepath = str((Path(app_path) / "strategy") / f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    st.session_state.strategy = module
    params = getattr(st.session_state.strategy, "params", None)
    update = getattr(st.session_state.strategy, "update", None)
    if params is None or update is None:
        raise ValueError("Strategy must have params, update functions")

def update_market(keep_kline: int):
    timepoints = st.session_state.timepoints
    market = fetch_realtime(raw2ricequant)
    if timepoints.size >= keep_kline:
        st.session_state.market = pd.concat([
            st.session_state.market.loc[timepoints[-239]:, :], market
        ])
    else:
        st.session_state.market = pd.concat([st.session_state.market, market])
    st.session_state.timepoints = st.session_state.market.index.get_level_values(0).unique()

def update_broker(app_path: str | Path, refresh_interval: int | str, keep_kline: int):
    @st.fragment(run_every=refresh_interval)
    def _update_broker():
        broker = st.session_state.get('broker')
        strategy = st.session_state.get('strategy')
        strategy_stop = st.session_state.get('strategy_stop', True)
        strategy_args = st.session_state.get('strategy_args', {})
        if broker is not None and is_trading_time():
            update_market(keep_kline=keep_kline)
            market = st.session_state.market
            timepoints = market.index.get_level_values(0).unique()
            market_now = market.loc[timepoints[-1]].copy()
            market_pre = market.loc[timepoints[-2]]
            market_now["open"] = market_pre["close"]
            market_now["high"] = np.maximum(market_now["high"], market_pre["high"])
            market_now["low"] = np.minimum(market_now["low"], market_pre["low"])
            market_now["volume"] = market_now["volume"] - market_pre["volume"]
            if not strategy_stop:
                module = importlib.import_module(
                    str(strategy).replace("/", ".").replace("\\", ".")[:-3]
                )
                getattr(module, "update")(broker, pd.to_datetime('now'), **strategy_args)
            broker.update(time=pd.to_datetime('now'), market=market_now)
            broker.store((Path(app_path) / "broker") / f"{broker.brokid}.json")
        elif broker is not None:
            broker.update(time=pd.to_datetime('now'), market=pd.DataFrame())
            broker.store((Path(app_path) / "broker") / f"{broker.brokid}.json")
    return _update_broker

def display_realtime(refresh_interval: int | str = "3s"):
    @st.fragment(run_every=refresh_interval)
    def _display_realtime():
        st.header("Realtime")
        market = st.session_state.market.loc[st.session_state.timepoints[-1]]
        st.dataframe(market)
    return _display_realtime

@st.dialog("Edit your strategy", width="large")
def display_editor(app_path, code, name):
    height = max(len(code.split("\n")) * 20, 68)
    code = st.text_area(label="*edit your strategy*", value=code, height=height)
    if st.button("save", use_container_width=True):
        ((Path(app_path) / "strategy") / f"{name}.py").write_text(code)
        st.rerun()

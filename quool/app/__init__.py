try:
    import pandas as pd
    import akshare as ak
    import streamlit as st
    from pathlib import Path
    import plotly.subplots as sp
    import plotly.graph_objects as go

    REFRESH_INTERVAL = 5
    ASSET_PATH = Path("asset")
    BROKER_PATH = Path(ASSET_PATH) / "broker"

    def fetch_realtime(format=True):
        data = ak.stock_zh_a_spot_em().set_index("代码", drop=True).drop(columns="序号")
        if format:
            data["open"] = data["今开"]
            data["high"] = data["最高"]
            data["low"] = data["最低"]
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
    
    @st.fragment(run_every=REFRESH_INTERVAL)
    def display_realtime():
        st.header("Realtime")
        realtime = fetch_realtime(format=False)
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


except Exception as e:
    print(e)
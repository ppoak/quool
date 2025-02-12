import pandas as pd
import streamlit as st
import plotly.subplots as sp
import plotly.graph_objects as go
from io import BytesIO
from pathlib import Path
from .tool import setup_broker, read_quotes, setup_market, update_broker


def display_monitor(refresh_interval: str | int, placeholder, broker_path: Path):
    @st.fragment(run_every=refresh_interval)
    def _display_monitor(placeholder):
        broker = st.session_state.broker
        market = st.session_state.market
        timepoints = st.session_state.timepoints
        value = broker.get_value(market.loc[timepoints[-1]])
        pre_value = broker.get_value(market.loc[timepoints[-2]]) if len(timepoints) > 1 else value
        placeholder.empty()
        with placeholder.container():
            st.header("Metrics")
            st.metric("Total Value", value=round(value, 3), delta=value - pre_value)
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Balance", value=round(broker.balance, 3))
            with col2:
                st.metric("Market", value=round(value - broker.balance, 3))
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Number of Pendings", value=len(broker.pendings))
            with col2:
                st.metric("Number of Orders", value=len(broker.orders))
        broker.store(Path(broker_path) / f"{broker.brokid}.json")
    return _display_monitor(placeholder)

def display_transact():
    st.header("Transact")
    broker = st.session_state.broker
    code = st.text_input("*input symbol*")
    quantity = st.number_input("*input quantity*", step=100)
    limit = st.number_input("*limit price*", step=0.01, value=None)
    trigger = st.number_input("*trigger price*", step=0.01, value=None)
    exectype = st.selectbox("*execution type*", ["MARKET", "LIMIT", "STOP", "STOPLIMIT"], index=0)
    valid = pd.to_datetime(st.text_input("*valid time*", value=None))
    subcol1, subcol2 = st.columns(2)
    with subcol1:
        if st.button("buy", use_container_width=True):
            order = broker.buy(
                code=code, quantity=quantity, limit=limit, 
                trigger=trigger, exectype=exectype, valid=valid
            )
            st.toast(f"**{order}**", icon="✅")
                
    with subcol2:
        if st.button("sell", use_container_width=True):
            order = broker.sell(
                code=code, quantity=quantity, limit=limit,
                trigger=trigger, exectype=exectype, valid=valid
            )
            st.toast(f"**{order}**", icon="✅")

def display_cancel(refresh_interval="3s"):
    @st.fragment(run_every=refresh_interval)
    def display_cancel():
        broker = st.session_state.broker
        st.header("Cancel")
        options = broker.pendings["ordid"].tolist() if not broker.pendings.empty else []
        ordids = st.multiselect(
            "*cancel some order(s)*", options=sorted(options), 
            format_func=lambda x: str(broker.get_order(x))
        )
        for ordid in ordids:
            broker.cancel(ordid)
            st.toast(f"**order {ordid[:5]} canceled**", icon="✅")
    return display_cancel

def display_transfer(broker_path: str | Path):
    amount = st.number_input("*transfer principle*", value=1000000, step=10000)
    broker = st.session_state.broker
    if st.button("transfer", use_container_width=True):
        broker.transfer(time=None, amount=amount)
        broker.store(Path(broker_path) / f"{broker.brokid}.json")
        st.toast("**transfered**", icon="✅")

def display_bracket_transact():
    st.header("Bracket Transact")
    broker = st.session_state.broker
    file = st.file_uploader("*upload bracket orders*", type="xlsx")
    subcol1, subcol2 = st.columns(2)
    with subcol1:
        bio = BytesIO()
        writer = pd.ExcelWriter(bio, engine="xlsxwriter")
        pd.DataFrame(
            columns=["code", "quantity", "limit", "trigger", "exectype", "valid", "side"]
        ).to_excel(writer, sheet_name="bracket transaction", index=False)
        writer._save()
        bio.seek(0)
        st.download_button(
            "*download template*", data=bio.read(), 
            file_name="bracket_template.xlsx", use_container_width=True
        )
    with subcol2:
        if file is None:
            return
        data = pd.read_excel(BytesIO(file.getvalue()))
        for i, item in data.iterrows():
            if item["side"] == "buy":
                item.pop("side")
                broker.buy(**item)
            elif item["side"] == "sell":
                item.pop("side")
                broker.sell(**item)

def display_curve(values):
    fig = sp.make_subplots(3, 1, shared_xaxes=True, row_heights=[0.7, 0.15, 0.15])
    fig.add_traces([
        go.Scatter(
            x=values.index, 
            y=values["total"], 
            name="value"
        ),
        go.Scatter(
            x=values.index,
            y=values["market"],
            name="market"
        ),
    ], rows=1, cols=1)
    fig.add_traces([
        go.Scatter(
            x=values.index, 
            y=(values["total"] / values["total"].cummax() - 1) * 100, 
            name="drawdown")
    ], rows=2, cols=1)
    fig.add_traces([
        go.Bar(
            x=values.index, 
            y=values["turnover"] * 100, 
            name="turnover"
        )
    ], rows=3, cols=1)
    st.plotly_chart(fig)

def display_evaluation(evaluation, trades):
    st.subheader("Evaluation")
    cols = st.columns(3, vertical_alignment="top")
    with cols[0]:
        st.metric("Total Return", f"{evaluation['total_return(%)']:.2f}%")
        st.metric("Max Drawdown", f"{evaluation['max_drawdown(%)']:.2f}%")
        st.metric("Alpha", f"{evaluation['alpha(%)']:.2f}%")
        st.metric("Trade Win Rate", f"{evaluation['trade_win_rate(%)']:.2}%")
        st.metric("Position Duration", f"{evaluation['position_duration(days)']} days"
        )
        st.metric("Trade Return", f"{evaluation['trade_return(%)']:.2}%")
    with cols[1]:
        st.metric("Annual Return", f"{evaluation['annual_return(%)']:.2f}%")
        st.metric("Max Drawdown Period", f"{evaluation['max_drawdown_period']} days")
        st.metric("Annual Volatility", f"{evaluation['annual_volatility(%)']:.2f}%")
        st.metric("Beta", f"{evaluation['beta']:.2f}")
        st.metric("Excess Return", f"{evaluation['excess_return(%)']:.2f}%")
        st.metric("VaR 5%", f"{evaluation['VaR_5%(%)']:.2f}%")
    with cols[2]:
        st.metric("Sharpe Ratio", f"{evaluation['sharpe_ratio']:.2f}")
        st.metric("Information Ratio", f"{evaluation['information_ratio']:.2f}")
        st.metric("Sortino Ratio", f"{evaluation['sortino_ratio(%)']:.2f}%")
        st.metric("Turnover Rate", f"{evaluation['turnover_ratio(%)']:.2}%")
        st.metric("Excess Volatility", f"{evaluation['excess_return(%)']:.2f}%")
        st.metric("CVaR 5%", f"{evaluation['CVaR_5%(%)']:.2f}%")
    
    st.subheader("Trades")
    st.dataframe(trades)

def display_performance(quotes_path: str):
    broker = st.session_state.broker
    ledger = broker.ledger
    if ledger.empty:
        st.error("No Transaction")
        return
    backadj = st.checkbox("Back-adjusted", value=True)
    with st.spinner("Loading market...", show_time=True):
        data = read_quotes(quotes_path, ledger["time"].min(), ledger["time"].max(), backadj=backadj)
    
    st.header("Performance")
    evaluation = broker.evaluate(data)
    display_curve(evaluation["values"])
    display_evaluation(evaluation["evaluation"], evaluation["trades"])

def layout(
    broker_path: Path = "app/broker", 
    quotes_path: str = "app/quotes",
    refresh_interval: int | str ="3s",
    keep_kline: int = 240
):
    broker_path = Path(broker_path)
    broker_path.mkdir(parents=True, exist_ok=True)
    st.title("METRIC")
    setup_market()
    setup_broker(broker_path=broker_path)
    placeholder = st.empty()
    broker = st.session_state.get("broker")
    if broker is None:
        st.warning("No broker selected")
        return
    display_monitor(refresh_interval=refresh_interval, placeholder=placeholder, broker_path=Path(broker_path))
    update_broker(broker_path=broker_path, refresh_interval=refresh_interval, keep_kline=keep_kline)
    st.divider()
    st.title("TRANSACTION")
    col1, col2 = st.columns(2)
    with col1:
        display_transact()
    with col2:
        display_transfer(broker_path=broker_path)
        display_cancel(refresh_interval=refresh_interval)()
        display_bracket_transact()
    st.divider()
    st.title("PERFORMANCE")
    display_performance(quotes_path=quotes_path)
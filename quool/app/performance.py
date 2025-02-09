import pandas as pd
import streamlit as st
import plotly.subplots as sp
import plotly.graph_objects as go
from quool.app import read_market, fetch_realtime


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
            y=values["total"] / values["total"].cummax() - 1, 
            name="drawdown")
    ], rows=2, cols=1)
    fig.add_traces([
        go.Bar(
            x=values.index, 
            y=values["turnover"], 
            name="turnover"
        )
    ], rows=3, cols=1)
    st.plotly_chart(fig)

def display_evaluation(evaluation, trades, broker):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Evaluation")
        evaluation.loc["max_drawdown_period"] = evaluation.loc["max_drawdown_period"].days
        evaluation.loc["position_duration(days)"] = (
            evaluation.loc["position_duration(days)"].days 
            if isinstance(evaluation.loc["position_duration(days)"], pd.Timedelta) 
            else evaluation.loc["position_duration(days)"]
        )
        st.dataframe(evaluation)
    with col2:
        st.subheader("Status")
        st.dataframe(pd.Series({
            "balance": broker.balance, 
            "value": broker.balance + (fetch_realtime()["close"] * pd.Series(broker.positions)).sum(),
            "pendings": len(broker.pendings),
            "orders": len(broker.orders),
        }, name="status").to_frame().T, hide_index=True)
        st.subheader("Trades")
        st.dataframe(trades, hide_index=True)

def display_performance():
    broker = st.session_state.broker
    ledger = broker.ledger
    if ledger.empty:
        st.error("No Transaction")
        return
    with st.spinner("Loading market...", show_time=True):
        broker.market = read_market(ledger["time"].min(), ledger["time"].max())
    
    st.header("Performance")
    evaluation = broker.evaluate()
    display_curve(evaluation["values"])
    display_evaluation(evaluation["evaluation"], evaluation["trades"], broker)

def layout():
    st.title("PERFORMANCE")
    display_performance()


if __name__ == "__page__":
    layout()

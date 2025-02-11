import pandas as pd
import streamlit as st
import plotly.subplots as sp
import plotly.graph_objects as go
from .tool import read_market


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
        st.metric("Position Duration", 
            f"{evaluation['position_duration(days)'].days 
            if isinstance(evaluation['position_duration(days)'], pd.Timedelta) else 0:.2f} days"
        )
        st.metric("Trade Return", f"{evaluation['trade_return(%)']:.2}%")
    with cols[1]:
        st.metric("Annual Return", f"{evaluation['annual_return(%)']:.2f}%")
        st.metric("Max Drawdown Period", f"{
            evaluation['max_drawdown_period'].days
            if isinstance(evaluation['max_drawdown_period'], pd.Timedelta) else 0:
        } days")
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

def display_performance():
    broker = st.session_state.broker
    ledger = broker.ledger
    if ledger.empty:
        st.error("No Transaction")
        return
    backadj = st.checkbox("Back-adjusted", value=True)
    with st.spinner("Loading market...", show_time=True):
        data = read_market(ledger["time"].min(), ledger["time"].max(), backadj=backadj)
    
    st.header("Performance")
    evaluation = broker.evaluate(data)
    display_curve(evaluation["values"])
    display_evaluation(evaluation["evaluation"], evaluation["trades"])

def layout():
    st.title("PERFORMANCE")
    display_performance()

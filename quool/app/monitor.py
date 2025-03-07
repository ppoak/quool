import pandas as pd
import streamlit as st
from .tool import BROKER_PATH, REFRESH_INTERVAL


@st.fragment(run_every=REFRESH_INTERVAL)
def display_monitor(placeholder):
    broker = st.session_state.broker
    market = st.session_state.market
    timepoints = st.session_state.timepoints
    value = broker.get_value(market.loc[timepoints[-1]])
    pre_value = broker.get_value(market.loc[timepoints[-2]]) if len(timepoints) > 1 else value
    orders = broker.orders
    pendings = broker.pendings
    if not orders.empty:
        orders["ordid"] = orders["ordid"].str.slice(0, 5)
    if not pendings.empty:
        pendings["ordid"] = pendings["ordid"].str.slice(0, 5)
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
        st.header("Positions")
        st.dataframe(pd.concat([broker.positions, market.loc[timepoints[-1]]], axis=1, join="inner"))
        st.header("Pending Orders")
        st.dataframe(pendings, hide_index=True)
        st.header("History Orders")
        st.dataframe(orders, hide_index=True)
    broker.store(BROKER_PATH / f"{broker.brokid}.json")

def layout():
    st.title("BROKER STATUS")
    placeholder = st.empty()
    broker = st.session_state.get("broker")
    if broker is None:
        st.warning("No broker selected")
        return
    display_monitor(placeholder)

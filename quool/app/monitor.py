import asyncio
import pandas as pd
import streamlit as st
import plotly.subplots as sp
import plotly.graph_objects as go
from quool.app import fetch_realtime, BROKER_PATH, REFRESH_INTERVAL, read_market


@st.fragment(run_every=REFRESH_INTERVAL)
def display_monitor(placeholder):
    market = fetch_realtime()
    broker = st.session_state.broker
    broker.update(None, market)
    placeholder.empty()
    with placeholder.container():
        orders = broker.orders
        pendings = broker.pendings
        if not orders.empty:
            orders["ordid"] = orders["ordid"].str.slice(0, 5)
        if not pendings.empty:
            pendings["ordid"] = pendings["ordid"].str.slice(0, 5)
        st.header("Status")
        st.dataframe(pd.Series({
            "balance": broker.balance, 
            "value": broker.balance + (market["close"] * pd.Series(broker.positions)).sum(),
            "pendings": len(broker.pendings),
            "orders": len(broker.orders),
        }, name="status").to_frame().T)
        st.header("Positions")
        if not broker.positions.empty:
            st.dataframe(pd.concat([broker.positions, market], axis=1, join="inner"))
        else:
            st.write("Empty")
        st.header("Pendings")
        if not pendings.empty:
            st.dataframe(pendings)
        else:
            st.write("Empty")
        st.header("History")
        if not orders.empty:
            st.dataframe(orders)
        else:
            st.write("Empty")
    broker.store(BROKER_PATH / f"{broker.brokid}.json")

def layout():
    st.title("BROKER STATUS")
    placeholder = st.empty()
    broker = st.session_state.get("broker")
    if broker is None:
        st.warning("No broker selected")
        return
    display_monitor(placeholder)


if __name__ == "__page__":
    layout()

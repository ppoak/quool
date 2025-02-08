import asyncio
import pandas as pd
import streamlit as st
from quool.app import fetch_realtime, BROKER_PATH, REFRESH_INTERVAL


@st.fragment(run_every=REFRESH_INTERVAL)
def display_monitor(_):
    market = fetch_realtime()
    broker = st.session_state.broker
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
    broker.store(st.session_state.bpath)

async def layout():
    st.title("BROKER STATUS")
    global placeholder
    placeholder = st.empty()
    broker = st.session_state.get("broker")
    if broker is None:
        st.warning("No broker selected")
        return
    task = asyncio.create_task(broker.realtime_update())
    task.add_done_callback(display_monitor)
    display_monitor(None)

if __name__ == "__page__":
    asyncio.run(layout())

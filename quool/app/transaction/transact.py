import pandas as pd
import streamlit as st
from quool.app import fetch_realtime, display_realtime, BROKER_PATH


def display_transact():
    st.header("Transact")
    broker = st.session_state.get("broker")
    if broker is None:
        st.warning("No broker selected")
        return
    code = st.text_input("*input symbol*")
    quantity = st.number_input("*input quantity*", step=100)
    limit = st.number_input("*limit price*", step=0.01, value=None)
    trigger = st.number_input("*trigger price*", step=0.01, value=None)
    exectype = st.selectbox("*execution type*", ["MARKET", "LIMIT", "STOP", "STOPLIMIT"], index=0)
    valid = pd.to_datetime(st.text_input("*valid time*", value=None))
    subcol1, subcol2 = st.columns(2)
    with subcol1:
        if st.button("buy", use_container_width=True):
            broker.buy(
                code=code, quantity=quantity, limit=limit, 
                trigger=trigger, exectype=exectype, valid=valid
            )
            broker.update(time=None, market=fetch_realtime())
            st.success("**order submited**", icon="✅")
    with subcol2:
        if st.button("sell", use_container_width=True):
            broker.sell(
                code=code, quantity=quantity, limit=limit,
                trigger=trigger, exectype=exectype, valid=valid
            )
            broker.update(time=None, market=fetch_realtime())
            st.success("**order submited**", icon="✅")

@st.fragment(run_every="3s")
def display_cancel():
    broker = st.session_state.get("broker")
    if broker is None:
        st.warning("No broker selected")
        return
    st.header("Cancel")
    ordid = st.selectbox("*select an order*", options=broker.pendings, format_func=str)
    if st.button("cancel", use_container_width=True):
        broker.cancel(ordid)
        broker.store(BROKER_PATH / f"{st.session_state.bname}.json")
        st.success("**order canceled**", icon="✅")

def display_transfer():
    amount = st.number_input("*transfer principle*", value=1000000, step=10000)
    broker = st.session_state.get("broker")
    if broker is None:
        st.warning("No broker selected")
        return
    if st.button("transfer", use_container_width=True):
        broker.transfer(time=None, amount=amount)
        broker.store(BROKER_PATH / f"{st.session_state.bname}.json")
        st.success("**transfered**", icon="✅")

def layout():
    st.title("TRANSACT")
    display_realtime()
    col1, col2 = st.columns(2)
    with col1:
        display_transact()
    with col2:
        display_cancel()
        display_transfer()


if __name__ == "__page__":
    layout()
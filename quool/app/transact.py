import asyncio
import pandas as pd
from io import BytesIO
import streamlit as st
from quool.app import display_realtime, fetch_realtime, BROKER_PATH, REFRESH_INTERVAL, TEMPLATE_PATH


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
            try:
                broker.buy(
                    code=code, quantity=quantity, limit=limit, 
                    trigger=trigger, exectype=exectype, valid=valid
                )
                broker.update(None, fetch_realtime())
                broker.store(BROKER_PATH / f"{broker.brokid}.json")
            except Exception as e:
                st.toast(f"**{e}**", icon="❌")
            else:
                st.toast("**order submited**", icon="✅")
                
    with subcol2:
        if st.button("sell", use_container_width=True):
            try:
                broker.sell(
                    code=code, quantity=quantity, limit=limit,
                    trigger=trigger, exectype=exectype, valid=valid
                )
                broker.update(None, fetch_realtime())
                broker.store(BROKER_PATH / f"{broker.brokid}.json")
            except Exception as e:
                st.toast(f"**{e}**", icon="❌")
            else:
                st.toast("**order submited**", icon="✅")

@st.fragment(run_every=REFRESH_INTERVAL)
def display_cancel():
    broker = st.session_state.broker
    st.header("Cancel")
    ordids = st.multiselect(
        "*select an order*", options=broker.pendings["ordid"].tolist(), 
        format_func=lambda x: x[:5]
    )
    if st.button("cancel", use_container_width=True):
        for ordid in ordids:
            broker.cancel(ordid)
        broker.store(st.session_state.bpath)
        st.toast("**order canceled**", icon="✅")

def display_transfer():
    amount = st.number_input("*transfer principle*", value=1000000, step=10000)
    broker = st.session_state.broker
    if st.button("transfer", use_container_width=True):
        broker.transfer(time=None, amount=amount)
        broker.store(BROKER_PATH / f"{broker.brokid}.json")
        st.toast("**transfered**", icon="✅")

def display_bracket_transact():
    st.header("Bracket Transact")
    broker = st.session_state.broker
    file = st.file_uploader("*upload bracket orders*", type="xlsx")
    subcol1, subcol2 = st.columns(2)
    with subcol1:
        st.download_button("*download template*", 
            data=(TEMPLATE_PATH / "bracket_template.xlsx").read_bytes(), 
            file_name="bracket_template.xlsx", use_container_width=True
        )
    with subcol2:
        if st.button("transact", use_container_width=True):
            if file is None:
                st.toast("**no file uploaded**", icon="❌")
                return
            with pd.ExcelFile(BytesIO(file.getvalue())) as f:
                for sheet in f.sheet_names:
                    if sheet == "buy":
                        for _, row in pd.read_excel(f, sheet_name=sheet, dtype={"code": "str"}).iterrows():
                            broker.buy(**row)
                    elif sheet == "sell":
                        for _, row in pd.read_excel(f, sheet_name=sheet, dtype={"code": "str"}).iterrows():
                            broker.sell(**row)
            broker.update(None, fetch_realtime())
            broker.store(BROKER_PATH / f"{broker.brokid}.json")

def layout():
    st.title("TRANSACT")
    broker = st.session_state.get("broker")
    if broker is None:
        st.warning("No broker selected")
        return
    display_realtime()
    col1, col2 = st.columns(2)
    with col1:
        display_transact()
    with col2:
        display_transfer()
        display_bracket_transact()


if __name__ == "__page__":
    layout()
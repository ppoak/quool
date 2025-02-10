import pandas as pd
from io import BytesIO
import streamlit as st
from quool.app import (
    display_realtime,
    BROKER_PATH, REFRESH_INTERVAL, TEMPLATE_PATH
)


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
            broker.buy(
                code=code, quantity=quantity, limit=limit, 
                trigger=trigger, exectype=exectype, valid=valid
            )
            st.toast("**order placed**", icon="✅")
                
    with subcol2:
        if st.button("sell", use_container_width=True):
            broker.sell(
                code=code, quantity=quantity, limit=limit,
                trigger=trigger, exectype=exectype, valid=valid
            )
            st.toast("**order placed**", icon="✅")

@st.fragment(run_every=REFRESH_INTERVAL)
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
        if file is None:
            return
        with pd.ExcelFile(BytesIO(file.getvalue())) as f:
            for sheet in f.sheet_names:
                if sheet == "buy":
                    for _, row in pd.read_excel(f, sheet_name=sheet, dtype={"code": "str"}).iterrows():
                        broker.buy(**row)
                elif sheet == "sell":
                    for _, row in pd.read_excel(f, sheet_name=sheet, dtype={"code": "str"}).iterrows():
                        broker.sell(**row)

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
        display_cancel()
        display_bracket_transact()


if __name__ == "__page__":
    layout()
import pandas as pd
import streamlit as st
from io import BytesIO
from pathlib import Path
from .tool import display_realtime


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

def display_transfer(app_path: str | Path):
    amount = st.number_input("*transfer principle*", value=1000000, step=10000)
    broker = st.session_state.broker
    if st.button("transfer", use_container_width=True):
        broker.transfer(time=None, amount=amount)
        broker.store((Path(app_path) / "broker") / f"{broker.brokid}.json")
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

def layout(app_path: str | Path = "app", refresh_interval: int | str = "30s"):
    st.title("TRANSACT")
    broker = st.session_state.get("broker")
    if broker is None:
        st.warning("No broker selected")
        return
    display_realtime(refresh_interval=refresh_interval)()
    col1, col2 = st.columns(2)
    with col1:
        display_transact()
    with col2:
        display_transfer(app_path=app_path)
        display_cancel(refresh_interval=refresh_interval)()
        display_bracket_transact()

import pandas as pd
import streamlit as st
from quool import ParquetManager, Broker
from quool.app import BROKER_PATH


def read_market(begin, end):
    if not (begin is None and end is None):
        begin = begin or pd.Timestamp("2015-01-01")
        end = end or pd.Timestamp.now()
        return ParquetManager("D:/Documents/DataBase/quotes_day").read(
            date__ge=begin, date__le=end,
        )
    else:
        return None

def display_creator(begin, end):
    st.header("Create One")
    name = st.text_input("*broker name*", value="default")

    if st.button("create"):
        market = read_market(begin, end)
        broker = Broker(market=market)
        broker.store(BROKER_PATH / f"{name}.json")
        st.session_state.bname = name
        st.session_state.broker = broker
        st.success("broker created", icon="✅")
        
def display_selector(begin, end):
    st.header("Select One")
    brokers = list(BROKER_PATH.glob("*.json"))
    name = st.selectbox("*Select an existing broker*", [b.stem for b in brokers])
    if st.button("select"):
        if name is not None:
            broker = Broker.restore(BROKER_PATH / f"{name}.json", market=read_market(begin, end))
            st.session_state.bname = name
            st.session_state.broker = broker
            st.success("broker selected", icon="✅")
        else:
            st.error("no broker selected", icon="❌")

def layout():
    st.title("BROKER SELECTOR")

    current_broker = st.empty()
    begin, end = None, None
    if st.button("Backtest"):
        begin = st.date_input("*select begin date for backtesting*", value="2015-01-01")
        end = st.date_input("*select end date for backtesting*")

    col1, col2 = st.columns(2)
    with col1:
        display_creator(begin, end)
    with col2:
        display_selector(begin, end)
    bname = st.session_state.get('bname', "*No Broker Selected*")
    current_broker.write(f"Current Broker: {bname}")

if __name__ == "__page__":
    layout()

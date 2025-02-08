import streamlit as st
from pathlib import Path
from quool.app import BROKER_PATH, Broker


def display_creator():
    st.header("Broker Creator")
    name = st.text_input("*broker name*", value="default")

    if st.button("create"):
        broker = Broker(market=None)
        broker.store(BROKER_PATH / f"{name}.json")
        st.session_state.bpath = Path(BROKER_PATH / f"{name}.json")
        st.session_state.broker = broker
        st.toast("broker created", icon="✅")
        
def display_selector():
    st.header("Broker Selector")
    brokers = list(BROKER_PATH.glob("*.json"))
    path = st.selectbox("*Select an existing broker*", brokers, format_func=lambda x: x.stem)
    if st.button("select"):
        if path is not None:
            broker = Broker.restore(path, market=None)
            st.session_state.bpath = path
            st.session_state.broker = broker
            st.toast("broker selected", icon="✅")
        else:
            st.toast("no broker selected", icon="❌")

def layout():
    st.title("BROKER SELECTOR")
    current_broker = st.empty()
    col1, col2 = st.columns(2)
    with col1:
        display_creator()
    with col2:
        display_selector()
    bpath = st.session_state.get('bpath', Path("*No Broker Selected*"))
    current_broker.write(f"Current Broker: **{bpath.stem}**")

if __name__ == "__page__":
    layout()

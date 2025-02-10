import sys
import streamlit as st
from streamlit import runtime
from streamlit.web import cli
from quool.app import (
    Broker, update_broker,
    ASSET_PATH, BROKER_PATH, LOG_PATH,
    TEMPLATE_PATH, STRATEGIES_PATH,
)


def main():
    st.set_page_config(
        page_title="TraderApp",
        page_icon="😊",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={
            'Get Help': 'https://www.extremelycoolapp.com/help',
            'Report a bug': "https://www.extremelycoolapp.com/bug",
            'About': "# This is a header. This is an *extremely* cool app!"
        }
    )

    monitor = st.Page("monitor.py", title="Monitor", icon="📈")
    transact = st.Page("transact.py", title="Transact", icon="💸")
    runner = st.Page("runner.py", title="Runner", icon="💡")
    performance = st.Page("performance.py", title="Performance", icon="📊")
    pg = st.navigation([monitor, transact, runner, performance])
    pg.run()

    selection = st.sidebar.selectbox(f"*select broker*", [broker.stem for broker in BROKER_PATH.glob("*.json")], index=0)
    if selection is not None:
        st.session_state.broker = Broker.restore(path=BROKER_PATH / f"{selection}.json")
    if st.session_state.get("broker") is None:
        st.sidebar.warning("No broker selected")
    else:
        st.sidebar.write(f"CURRENT BROKER: **{st.session_state.broker.brokid}**")
    
    name = st.sidebar.text_input("*input broker id*", value="default")
    col1, col2 = st.sidebar.columns(2)
    if col1.button("*create*", use_container_width=True):
        broker = Broker(brokid=name)
        broker.store(BROKER_PATH / f"{name}.json")
        st.session_state.broker = broker
        st.rerun()
    if col2.button("*save*", use_container_width=True):
        st.session_state.broker.store(BROKER_PATH / f"{name}.json")
    
    with st.sidebar.container():
        update_broker()


if __name__ == "__main__":
    ASSET_PATH.mkdir(parents=True, exist_ok=True)
    BROKER_PATH.mkdir(parents=True, exist_ok=True)
    STRATEGIES_PATH.mkdir(parents=True, exist_ok=True)
    TEMPLATE_PATH.mkdir(parents=True, exist_ok=True)
    LOG_PATH.mkdir(parents=True, exist_ok=True)
    if runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", __file__]
        cli.main()

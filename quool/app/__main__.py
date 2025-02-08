import sys
import streamlit as st
from streamlit import runtime
from streamlit.web import cli
from quool.app import ASSET_PATH, BROKER_PATH


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

    broker_selector = st.Page("brokers/selector.py", title="Selector", icon="📊")
    broker_monitor = st.Page("brokers/monitor.py", title="Monitor", icon="📈")
    transaction_transact = st.Page("transaction/transact.py", title="Transact", icon="💸")
    pg = st.navigation({
        "Broker": [broker_selector, broker_monitor],
        "Transaction": [transaction_transact],
    })
    pg.run()


if __name__ == "__main__":
    ASSET_PATH.mkdir(parents=True, exist_ok=True)
    BROKER_PATH.mkdir(parents=True, exist_ok=True)
    if runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", __file__]
        cli.main()

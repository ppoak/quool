import streamlit as st
import quool.app.tool as tool
from quool import Broker
from .monitor import layout as monitor_layout
from .performance import layout as performance_layout
from .runner import layout as runner_layout
from .strategy import layout as strategy_layout
from .transact import layout as transact_layout
from .tool import (
    fetch_realtime,
    update_broker, update_market, update_strategy,
    ASSET_PATH, BROKER_PATH, LOG_PATH, STRATEGIES_PATH,
)


def setup_styler():
    styler = st.sidebar.text_input("*input code styler*", value="raw2ricequant")
    styler = getattr(tool, styler, None)
    if styler is None:
        st.sidebar.warning(f"No styler named: {styler.__name__}")
    else:
        st.session_state.styler = styler

def setup_broker():
    selection = st.sidebar.selectbox(f"*select broker*", [broker.stem for broker in BROKER_PATH.glob("*.json")], index=0)
    if selection is not None:
        st.session_state.broker = Broker.restore(path=BROKER_PATH / f"{selection}.json")
    if st.session_state.get("broker") is None:
        st.sidebar.warning("No broker selected")
    else:
        st.sidebar.write(f"CURRENT BROKER: **{st.session_state.broker.brokid}**")

    name = st.sidebar.text_input("*input broker id*", value="default")
    col1, col2, col3 = st.sidebar.columns(3)
    if col1.button("*create*", use_container_width=True):
        broker = Broker(brokid=name)
        broker.store(BROKER_PATH / f"{name}.json")
        st.session_state.broker = broker
        st.rerun()
    if col2.button("*save*", use_container_width=True):
        st.session_state.broker.store(BROKER_PATH / f"{name}.json")
    if col3.button("*delete*", use_container_width=True):
        st.session_state.broker = None
        (BROKER_PATH / f"{name}.json").unlink()
        st.rerun()

def setup_page():
    monitor = st.Page(monitor_layout, title="Monitor", icon="📈", url_path="monitor")
    transact = st.Page(transact_layout, title="Transact", icon="💸", url_path="transact")
    runner = st.Page(runner_layout, title="Runner", icon="▶️", url_path="runner")
    performance = st.Page(performance_layout, title="Performance", icon="📊", url_path="performance")
    strategy = st.Page(strategy_layout, title="Strategy", icon="💡", url_path="strategy")
    pg = st.navigation([monitor, transact, strategy, runner, performance])
    pg.run()

def setup_market():
    if st.session_state.get("market") is None:
        st.session_state.market = fetch_realtime(code_styler=st.session_state.styler)
        st.session_state.timepoints = st.session_state.market.index.get_level_values(0).unique()

def setup_strategy():
    if st.session_state.get('broker') is None:
        st.sidebar.warning("No broker selected")
        return
    
    selection = st.sidebar.selectbox(f"*select strategy*", [strategy.stem for strategy in STRATEGIES_PATH.glob("*.py")], index=None)
    if selection is not None:
        try:
            update_strategy(selection)
        except Exception as e:
            st.sidebar.error(f"Error in strategy {selection}: {e}")
            return
    
    if st.session_state.get("strategy") is None:
        st.sidebar.warning("No strategy selected")
        return
    
    st.sidebar.write(f"CURRENT STRATEGY: **{st.session_state.strategy.__name__}**")
    with st.sidebar.container():
        st.session_state.strategy_kwargs = getattr(st.session_state.strategy, "params")()
    
    runonce = st.sidebar.checkbox("*run once*", value=True)
    col1, col2 = st.sidebar.columns(2)
    if col1.button("*run*", use_container_width=True):
        update_market()
        st.session_state.broker.data = st.session_state.market
        st.session_state.broker.timepoints = st.session_state.timepoints
        getattr(st.session_state.strategy, "init")(
            broker=st.session_state.broker, 
            time=st.session_state.market.index[0][0], 
            **st.session_state.strategy_kwargs
        )
        if runonce:
            st.session_state.strategy_stop = True
            getattr(st.session_state.strategy, "update")(
                broker=st.session_state.broker, 
                time=st.session_state.market.index[0][0], 
                **st.session_state.strategy_kwargs
            )
        else:
            st.session_state.strategy_stop = False
        st.rerun()
    if col2.button("*remove*", use_container_width=True):
        st.session_state.strategy = None
        (STRATEGIES_PATH / f"{selection}.py").unlink()
        st.rerun()

def layout():
    ASSET_PATH.mkdir(parents=True, exist_ok=True)
    BROKER_PATH.mkdir(parents=True, exist_ok=True)
    STRATEGIES_PATH.mkdir(parents=True, exist_ok=True)
    LOG_PATH.mkdir(parents=True, exist_ok=True)
    setup_styler()
    setup_broker()
    setup_market()
    setup_strategy()
    setup_page()
    with st.sidebar.container():
        update_broker()

import streamlit as st
from quool import Broker
from pathlib import Path
from functools import partial
from .monitor import layout as monitor_layout
from .performance import layout as performance_layout
from .runner import layout as runner_layout
from .strategy import layout as strategy_layout
from .transact import layout as transact_layout
from .tool import (
    fetch_realtime, raw2ricequant,
    update_broker, update_market, update_strategy,
)


def setup_broker(app_path: Path):
    selection = st.sidebar.selectbox(f"*select broker*", [broker.stem for broker in (Path(app_path) / "broker").glob("*.json")], index=0)
    if selection is not None:
        st.session_state.broker = Broker.restore(path=(Path(app_path) / "broker") / f"{selection}.json")
    if st.session_state.get("broker") is None:
        st.sidebar.warning("No broker selected")
    else:
        st.sidebar.write(f"CURRENT BROKER: **{st.session_state.broker.brokid}**")

    name = st.sidebar.text_input("*input broker id*", value="default")
    col1, col2, col3 = st.sidebar.columns(3)
    if col1.button("*create*", use_container_width=True):
        broker = Broker(brokid=name)
        broker.store((Path(app_path) / "broker") / f"{name}.json")
        st.session_state.broker = broker
        st.rerun()
    if col2.button("*save*", use_container_width=True):
        st.session_state.broker.store((Path(app_path) / "broker") / f"{name}.json")
    if col3.button("*delete*", use_container_width=True):
        st.session_state.broker = None
        ((Path(app_path) / "broker") / f"{name}.json").unlink()
        st.rerun()

def setup_page(app_path: str | Path = "app", refresh_interval: int | str = "30s"):
    monitor = st.Page(partial(monitor_layout, refresh_interval=refresh_interval, app_path=app_path), title="Monitor", icon="📈", url_path="monitor")
    transact = st.Page(partial(transact_layout, app_path=app_path, refresh_interval=refresh_interval), title="Transact", icon="💸", url_path="transact")
    runner = st.Page(partial(runner_layout, app_path=app_path), title="Runner", icon="▶️", url_path="runner")
    performance = st.Page(performance_layout, title="Performance", icon="📊", url_path="performance")
    strategy = st.Page(partial(strategy_layout, app_path=app_path), title="Strategy", icon="💡", url_path="strategy")
    pg = st.navigation([monitor, transact, strategy, runner, performance])
    pg.run()

def setup_market():
    if st.session_state.get("market") is None:
        st.session_state.market = fetch_realtime(code_styler=raw2ricequant)
        st.session_state.timepoints = st.session_state.market.index.get_level_values(0).unique()

def setup_strategy(strategy_path: Path, keep_kline: int):
    if st.session_state.get('broker') is None:
        st.sidebar.warning("No broker selected")
        return
    
    selection = st.sidebar.selectbox(f"*select strategy*", [strategy.stem for strategy in strategy_path.glob("*.py")], index=None)
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
        update_market(keep_kline=keep_kline)
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
        (strategy_path / f"{selection}.py").unlink()
        st.rerun()

def layout(app_path: str = "app", refresh_interval: str | int = "3s", keep_kline: int = 240):
    app_path = Path(app_path)
    broker_path = app_path / "broker"
    strategy_path = app_path / "strategy"
    log_path = app_path / "log"
    app_path.mkdir(parents=True, exist_ok=True)
    broker_path.mkdir(parents=True, exist_ok=True)
    strategy_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)
    setup_broker(app_path=app_path)
    setup_market()
    setup_strategy(strategy_path=strategy_path, keep_kline=keep_kline)
    setup_page(refresh_interval=refresh_interval, app_path=app_path)
    with st.sidebar.container():
        update_broker(app_path=app_path, refresh_interval=refresh_interval, keep_kline=keep_kline)()

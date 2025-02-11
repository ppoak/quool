import importlib
import traceback
import pandas as pd
import streamlit as st
from io import BytesIO
from pathlib import Path
from quool import Broker
from joblib import Parallel, delayed
from quool.app import (
    read_market, display_editor,
    STRATEGIES_PATH, LOG_PATH, BROKER_PATH, TEMPLATE_PATH, REFRESH_INTERVAL,
)


@delayed
def run_strategy(name, market, init, update, stop, since, params):
    timepoints = market.index.get_level_values(0).unique()
    broker = Broker(market, brokid=name)
    if init is not None:
        init(broker=broker, time=timepoints[0], **params)
    for tp in timepoints:
        broker.update(tp)
        update(time=tp, broker=broker, **params)
    if stop is not None:
        stop(broker=broker, time=timepoints[0], **params)
    broker.store(BROKER_PATH / f"{broker.brokid}.json", since)

def display_market():
    error_nomarket = st.empty()
    if st.session_state.get("market") is None:
        error_nomarket.error("No market selected")
    begin = st.date_input("*select begin date for backtesting*", value=pd.to_datetime("now") - pd.Timedelta(days=30))
    end = st.date_input("*select end date for backtesting*", value='today')
    extra = st.text_input("extra fields for market loading", value=None)
    with st.spinner("Loading market...", show_time=True):
        try:
            market = read_market(begin, end, extra=extra)
            st.session_state.market = market
        except Exception as e:
            st.toast(f"Error loading market: {e}", icon="❌")
    error_nomarket.empty()

def display_selector():
    st.header("Strategies Selector")
    strategies = list(STRATEGIES_PATH.glob("*.py"))
    strategy = st.selectbox("*Select an existing strategy*", strategies, format_func=lambda x: x.stem)
    if strategy is not None:
        st.session_state.strat = strategy
    else:
        st.error("No strategy selected", icon="❌")

def display_strategy():
    st.header(f"Strategy")
    if st.session_state.get("market") is None:
        st.warning("No market selected")
        return
    strategy = st.session_state.get("strat")
    if strategy is None:
        st.warning("No strategy selected")
        return
    else:
        market = st.session_state.market
        try:
            module = importlib.import_module(
                str(strategy).replace('/', '.').replace('\\', '.')[:-3]
            )
            importlib.reload(module)
        except Exception as e:
            st.error(f"Error loading strategy: {e}")
            st.error('\n'.join([trace.replace("^", "") for trace in traceback.format_exception(type(e), e, e.__traceback__)]))
            return
        if (
            not hasattr(module, "update") or not hasattr(module, "params")
            or not callable(getattr(module, "update")) or not callable(getattr(module, "params"))
        ):
            st.error("Invalid Strategy")
        params = getattr(module, "params")
        init = getattr(module, "init", None)
        update = getattr(module, "update")
        stop = getattr(module, "stop", None)
        st.write(module.__doc__ or "User is too lazy to write a docstring")
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            param = params()
        with subcol2:
            pd.DataFrame([param], index=[st.session_state.strat.stem]).to_excel(TEMPLATE_PATH / "param_template.xlsx")
            st.download_button(
                "*download param template*", 
                data=Path(TEMPLATE_PATH / "param_template.xlsx").read_bytes(),
                file_name="param_template.xlsx"
            )
            file = st.file_uploader("*upload param file*", type="xlsx")
        since = st.date_input("*save since*", value=market.index.levels[0].min())
        if st.button("Run", use_container_width=True):
            if file is not None:
                st.session_state.param = pd.read_excel(BytesIO(file.getvalue()), index_col=0).to_dict(orient="index")
            else:
                st.session_state.param = {st.session_state.strat.stem: param}
            status = st.empty()
            with status.container():
                display_status()
            with st.spinner("Running...", show_time=True):
                Parallel(
                    n_jobs=min(len(st.session_state.param), 4), 
                    backend="loky"
                )(run_strategy(
                    name=name,
                    market=market,
                    init=init,
                    update=update,
                    stop=stop,
                    since=since,
                    params=para,
                ) for name, para in st.session_state.param.items())
            status.empty()
            with status.container():
                display_status()
            st.toast(f"strategy {st.session_state.strat.stem} executed", icon="✅")

@st.fragment(run_every=REFRESH_INTERVAL)
def display_status():
    brokers = [b.stem for b in list(BROKER_PATH.glob("*.json"))]
    param = st.session_state.param
    finished = 0
    progress = st.progress(0)
    placeholders = {name: st.code("no logs") for name in param.keys()}
    for name in param.keys():
        if name in brokers:
            finished += 1
            log = Path(LOG_PATH) / f"{name}.log"
            if log.exists():
                placeholders[name].code(log.read_text(), language="json")
    progress.progress(finished / len(param), f"finished: {finished}/{len(param)}")

def layout():
    st.title("STRATEGIES RUNNER")
    display_market()
    display_selector()
    if st.button("edit", use_container_width=True):
        display_editor(st.session_state.strat.read_text())
    display_strategy()


if __name__ == "__page__":
    layout()


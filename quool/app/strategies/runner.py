import importlib
import pandas as pd
import streamlit as st
from joblib import Parallel, delayed
from io import BytesIO
from pathlib import Path
from quool import Broker
from quool.app import STRATEGIES_PATH, BROKER_PATH, TEMPLATE_PATH, read_market


@delayed
def run_strategy(name, market, preupdate, update, postupdate, history, params):
    timepoints = market.index.get_level_values(0).unique()
    broker = Broker(market)
    for tp in timepoints:
        if preupdate is not None:
            preupdate(broker=broker, time=tp, **params)
        update(time=tp, broker=broker, **params)
        if postupdate is not None:
            postupdate(broker=broker, time=tp, **params)
    broker.store(BROKER_PATH / f"{name}.json", history)

def display_market():
    error_nomarket = st.empty()
    if st.session_state.get("market") is None:
        error_nomarket.error("No market selected")
    begin = st.date_input("*select begin date for backtesting*", value="2015-01-01")
    end = st.date_input("*select end date for backtesting*")
    if st.button("Load", use_container_width=True):
        market = read_market(begin, end)
        st.session_state.market = market
        st.toast("Market loaded", icon="✅")
        error_nomarket.empty()

def display_creator():
    st.header("Strategies Creator")

def display_selector():
    st.header("Strategies Selector")
    strategies = list(STRATEGIES_PATH.glob("*.py"))
    strategy = st.selectbox("*Select an existing strategy*", strategies, format_func=lambda x: x.stem)
    if st.button("select"):
        if strategy is not None:
            st.toast("strategy selected", icon="✅")
            st.session_state.spath = strategy
        else:
            st.error("no strategy selected", icon="❌")

def display_strategy():
    st.header(f"Strategy")
    if st.session_state.get("market") is None:
        st.warning("No market selected")
        return
    strategy = st.session_state.get("spath")
    if strategy is None:
        st.warning("No strategy selected")
        return
    else:
        module = importlib.import_module(
            str(strategy).replace('/', '.').replace('\\', '.')[:-3]
        )
        importlib.reload(module)
        if (
            not hasattr(module, "update") or not hasattr(module, "params")
            or not callable(getattr(module, "update")) or not callable(getattr(module, "params"))
        ):
            st.error("Invalid Strategy")
        params = getattr(module, "params")
        preupdate = getattr(module, "preupdate", None)
        update = getattr(module, "update")
        postupdate = getattr(module, "postupdate", None)
        st.write(module.__doc__ or "User is too lazy to write a docstring")
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            param = params()
        with subcol2:
            pd.DataFrame([param], index=[st.session_state.spath.stem]).to_excel(TEMPLATE_PATH / "param_template.xlsx")
            st.download_button(
                "*download param template*", 
                data=Path(TEMPLATE_PATH / "param_template.xlsx").read_bytes(),
                file_name="param_template.xlsx"
            )
            file = st.file_uploader("*upload param file*", type="xlsx")
        history = st.checkbox("*save history*", value=True)
        if st.button("Run", use_container_width=True):
            if file is not None:
                param = pd.read_excel(BytesIO(file.getvalue()), index_col=0).to_dict(orient="index")
            else:
                param = {st.session_state.spath.stem: param}
            Parallel(n_jobs=min(len(param), 4), backend="loky")(run_strategy(
                name=name,
                market=st.session_state.market,
                preupdate=preupdate,
                update=update,
                postupdate=postupdate,
                history=history,
                params=para,
            ) for name, para in param.items())
            st.toast(f"strategy {st.session_state.spath.stem} executed", icon="✅")

def layout():
    st.title("STRATEGIES RUNNER")
    display_market()
    col1, col2 = st.columns(2)
    with col1:
        display_creator()
    with col2:
        display_selector()
    display_strategy()


if __name__ == "__page__":
    layout()


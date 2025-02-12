import pandas as pd
import streamlit as st
from io import BytesIO
from pathlib import Path
from joblib import Parallel, delayed
from quool import Broker, setup_logger
from .tool import (
    read_market, update_strategy,
)


@delayed
def run_strategy(app_path, name, data, init, update, stop, since, params):
    timepoints = data.index.get_level_values(0).unique()
    broker = Broker(brokid=name)
    broker.logger = setup_logger(
        name=name, file=(Path(app_path) / "log") / f"{name}.log", 
        stream=False, clear=True, replace=True
    )
    broker.data = data
    broker.timepoints = timepoints
    if init is not None:
        init(broker=broker, time=timepoints[0], **params)
    for tp in timepoints:
        broker.update(time=tp, market=data.loc[tp])
        broker.data = data.loc[:tp]
        broker.timepoints = timepoints[timepoints <= tp]
        update(time=tp, broker=broker, **params)
    if stop is not None:
        stop(broker=broker, time=timepoints[-1], **params)
    broker.store((Path(app_path) / "broker") / f"{broker.brokid}.json", since)

def display_market():
    error_nomarket = st.empty()
    if st.session_state.get("market") is None:
        error_nomarket.error("No market selected")
    begin = st.date_input("*select begin date for backtesting*", value=pd.to_datetime("now") - pd.Timedelta(days=30))
    end = st.date_input("*select end date for backtesting*", value='today')
    col1, col2 = st.columns([0.3, 0.7])
    with col1:
        backadj = st.checkbox("backadjust data", value=True)
    with col2:
        extra = st.text_input("extra fields for market loading", value="")
    with st.spinner("Loading market...", show_time=True):
        try:
            data = read_market(begin, end, backadj=backadj, extra=extra)
            st.session_state.data = data
        except Exception as e:
            st.toast(f"Error loading market: {e}", icon="❌")
    error_nomarket.empty()

def display_selector(app_path: str | Path):
    st.header("Strategies Selector")
    strategies = list((Path(app_path) / "strategy").glob("*.py"))
    strategy = st.selectbox("*Select an existing strategy*", [strat.stem for strat in strategies])
    if strategy is not None:
        update_strategy(strategy)
    else:
        st.error("No strategy selected", icon="❌")

def display_strategy(app_path: str | Path):
    st.header(f"Strategy")
    if st.session_state.get("market") is None:
        st.warning("No market selected")
        return
    strategy = st.session_state.get("strategy")
    if strategy is None:
        st.warning("No strategy selected")
        return
    else:
        data = st.session_state.data
        strategy = st.session_state.strategy
        params = getattr(strategy, "params")
        init = getattr(strategy, "init", None)
        update = getattr(strategy, "update")
        stop = getattr(strategy, "stop", None)
        st.write(strategy.__doc__ or "User is too lazy to write a docstring")
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            param = params()
        with subcol2:
            bio = BytesIO()
            writer = pd.ExcelWriter(bio, engine="xlsxwriter")
            pd.DataFrame([param], index=[strategy.__name__]).to_excel(writer, sheet_name="params")
            writer._save()
            bio.seek(0)
            st.download_button(
                "*download param template*", 
                data=bio.read(),
                file_name="param_template.xlsx"
            )
            file = st.file_uploader("*upload param file*", type="xlsx")
        since = st.date_input("*save since*", value=data.index.levels[0].min())
        if st.button("Run", use_container_width=True):
            if file is not None:
                st.session_state.param = pd.read_excel(BytesIO(file.getvalue()), index_col=0).to_dict(orient="index")
            else:
                st.session_state.param = {st.session_state.strategy.__name__: param}
            
            for name in st.session_state.param.keys():
                path = (Path(app_path) / "log") / f"{name}.log"
                if path.exists():
                    path.unlink()
            with st.spinner("Running...", show_time=True):
                Parallel(
                    n_jobs=min(len(st.session_state.param), 4), 
                    backend="loky"
                )(run_strategy(
                    name=name,
                    data=data,
                    init=init,
                    update=update,
                    stop=stop,
                    since=since,
                    params=para,
                ) for name, para in st.session_state.param.items())
            st.toast(f"strategy {st.session_state.strategy.__name__} executed", icon="✅")

def layout(app_path: str | Path = "app"):
    st.title("STRATEGIES RUNNER")
    display_market()
    display_strategy(app_path=app_path)

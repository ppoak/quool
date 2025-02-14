import re
import pandas as pd
import streamlit as st
import plotly.subplots as sp
import plotly.graph_objects as go
from pathlib import Path
from .backend import App
from io import BytesIO, StringIO
from joblib import delayed, Parallel
from quool import Broker, Order, setup_logger


@st.dialog("Edit your strategy", width="large")
def display_editor(module):
    code = Path(module.__file__).read_text()
    height = max(len(code.split("\n")) * 20, 68)
    code = st.text_area(label="*edit your strategy*", value=code, height=height)
    if st.button("save", use_container_width=True):
        Path(module.__file__).write_text(code)
        st.rerun()


class BrokerPage(App):

    def select_broker(self):
        selection = st.sidebar.selectbox(f"*select broker*", [broker.stem for broker in Path(self.broker_path).glob("*.json")], index=0)
        if selection is not None:
            self.set_broker(selection)
        if not hasattr(self, "broker"):
            st.sidebar.error("No broker selected")

    def activate_broker(self):
        act, deact = st.sidebar.columns(2)
        with act:
            if act.button("Activate", use_container_width=True):
                if self.get_status("_refresh_broker"):
                    st.toast(f"broker {self.broker.brokid} is already activated", icon="❌")
                else:
                    self.refresh_broker()
                    st.toast(f"broker {self.broker.brokid} activated", icon="✅")
        with deact:
            if deact.button("Deactivate", use_container_width=True):
                if self.get_status("_refresh_broker"):
                    self.stop_task("_refresh_broker")
                    st.toast(f"broker {self.broker.brokid} deactivated", icon="✅")
                else:
                    st.toast(f"broker {self.broker.brokid} is not activated", icon="❌")

    def manage_broker(self):
        name = st.sidebar.text_input("*input broker id*", value="default")
        col1, col2, col3 = st.sidebar.columns(3)
        if col1.button("*create*", use_container_width=True):
            self.set_broker(name)
            self.store_broker()
            st.rerun()
        if col2.button("*save*", use_container_width=True):
            self.store_broker()
        if col3.button("*delete*", use_container_width=True):
            self.delete_broker()
            st.rerun()

    @st.fragment(run_every="3s")
    def display_monitor(self, placeholder):
        value_now = self.broker.get_value(self.market.loc[self.timepoint[-1]])
        value_pre = self.broker.get_value(self.market.loc[self.timepoint[0]])
        pendings = self.broker.pendings
        orders = self.broker.orders
        if not pendings.empty:
            pendings["ordid"] = pendings["ordid"].str[:5]
        if not orders.empty:
            orders["ordid"] = orders["ordid"].str[:5]
        placeholder.empty()
        with placeholder.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Value", value=round(value_now, 3), delta=value_now - value_pre)
            with col2:
                st.metric("Balance", value=round(self.broker.balance, 3))
            with col3:
                st.metric("Market", value=round(value_now - self.broker.balance, 3))
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Number of Pendings", value=len(self.broker.pendings))
                with st.expander("Pendings"):
                    st.dataframe(pendings, hide_index=True)
            with col2:
                st.metric("Number of Orders", value=len(self.broker.orders))
                with st.expander("Orders"):
                    st.dataframe(orders, hide_index=True)
            with st.expander("Positions"):
                st.dataframe(pd.concat([
                    self.broker.positions, self.market.loc[self.timepoint[-1]]
                ], axis=1, join="inner"))

    @st.fragment(run_every="3s")
    def display_cancel(self):
        options = self.broker.pendings["ordid"].tolist() if not self.broker.pendings.empty else []
        ordids = st.multiselect(
            "*cancel some order(s)*", options=sorted(options), 
            format_func=lambda x: str(self.broker.get_order(x))
        )
        for ordid in ordids:
            self.broker.cancel(ordid)
            st.toast(f"**order {ordid[:5]} canceled**", icon="✅")

    def display_transfer(self):
        amount = st.number_input("*transfer principle*", value=1000000, step=10000)
        if st.button("transfer", use_container_width=True):
            self.broker.transfer(time=None, amount=amount)
            self.store_broker()
            st.toast("**transfered**", icon="✅")

    def display_bracket_transact(self):
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
                    self.broker.buy(**item)
                elif item["side"] == "sell":
                    item.pop("side")
                    self.broker.sell(**item)

    def display_code_transact(self):
        exectype = st.selectbox("*execution type*", ["MARKET", "LIMIT", "STOP", "STOPLIMIT"], index=0)
        code = st.text_input("*input symbol*")
        quantity = st.number_input("*input quantity*", step=100)
        if exectype in ["LIMIT", "STOPLIMIT"]:
            limit = st.number_input("*limit price*", step=0.01, value=None)
        else:
            limit = None
        if exectype in ["STOP", "STOPLIMIT"]:
            trigger = st.number_input("*trigger price*", step=0.01, value=None)
        else:
            trigger = None
        valid = pd.to_datetime(st.text_input("*valid time*", value=None))
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            if st.button("buy", use_container_width=True):
                order = self.broker.buy(
                    code=code, quantity=quantity, limit=limit, 
                    trigger=trigger, exectype=exectype, valid=valid
                )
                self.store_broker()
                st.toast(f"**{order}**", icon="✅")
                    
        with subcol2:
            if st.button("sell", use_container_width=True):
                order = self.broker.sell(
                    code=code, quantity=quantity, limit=limit,
                    trigger=trigger, exectype=exectype, valid=valid
                )
                self.store_broker()
                st.toast(f"**{order}**", icon="✅")

    def display_transact(self):
        col1, col2 = st.columns(2)
        with col1:
            self.display_code_transact()
        with col2:
            self.display_bracket_transact()
            self.display_cancel()
            self.display_transfer()
    
    def display_curve(self, values):
        fig = sp.make_subplots(3, 1, shared_xaxes=True, row_heights=[0.7, 0.15, 0.15])
        fig.add_traces([
            go.Scatter(
                x=values.index, 
                y=values["total"], 
                name="value"
            ),
            go.Scatter(
                x=values.index,
                y=values["market"],
                name="market"
            ),
        ], rows=1, cols=1)
        fig.add_traces([
            go.Scatter(
                x=values.index, 
                y=(values["total"] / values["total"].cummax() - 1) * 100, 
                name="drawdown")
        ], rows=2, cols=1)
        fig.add_traces([
            go.Bar(
                x=values.index, 
                y=values["turnover"] * 100, 
                name="turnover"
            )
        ], rows=3, cols=1)
        st.plotly_chart(fig)

    def display_evaluation(self, evaluation, trades):
        cols = st.columns(3, vertical_alignment="top")
        with cols[0]:
            st.metric("Total Return", f"{evaluation['total_return(%)']:.2f}%")
            st.metric("Max Drawdown", f"{evaluation['max_drawdown(%)']:.2f}%")
            st.metric("Alpha", f"{evaluation['alpha(%)']:.2f}%")
            st.metric("Trade Win Rate", f"{evaluation['trade_win_rate(%)']:.2}%")
            st.metric("Position Duration", f"{evaluation['position_duration(days)']} days"
            )
            st.metric("Trade Return", f"{evaluation['trade_return(%)']:.2}%")
        with cols[1]:
            st.metric("Annual Return", f"{evaluation['annual_return(%)']:.2f}%")
            st.metric("Max Drawdown Period", f"{evaluation['max_drawdown_period']} days")
            st.metric("Annual Volatility", f"{evaluation['annual_volatility(%)']:.2f}%")
            st.metric("Beta", f"{evaluation['beta']:.2f}")
            st.metric("Excess Return", f"{evaluation['excess_return(%)']:.2f}%")
            st.metric("VaR 5%", f"{evaluation['VaR_5%(%)']:.2f}%")
        with cols[2]:
            st.metric("Sharpe Ratio", f"{evaluation['sharpe_ratio']:.2f}")
            st.metric("Information Ratio", f"{evaluation['information_ratio']:.2f}")
            st.metric("Sortino Ratio", f"{evaluation['sortino_ratio(%)']:.2f}%")
            st.metric("Turnover Rate", f"{evaluation['turnover_ratio(%)']:.2}%")
            st.metric("Excess Volatility", f"{evaluation['excess_return(%)']:.2f}%")
            st.metric("CVaR 5%", f"{evaluation['CVaR_5%(%)']:.2f}%")
        
        st.subheader("Trades")
        st.dataframe(trades)

    def display_performance(self):
        ledger = self.broker.ledger
        if ledger.empty:
            st.error("no transaction")
            return
        backadj = st.checkbox("back-adjusted", value=True)
        with st.spinner("Loading market...", show_time=True):
            data = self.read_quotes(ledger["time"].min(), ledger["time"].max(), backadj=backadj)
            data = pd.concat([data, self.market[["open", "high", "low", "close", "volume"]]])
        
        if data is not None:
            evaluation = self.broker.evaluate(data)
            self.display_curve(evaluation["values"])
            self.display_evaluation(evaluation["evaluation"], evaluation["trades"])
        else:
            st.error("no market data available")

    def run(self):
        self.select_broker()
        self.set_market()
        self.refresh_market()
        self.activate_broker()
        self.manage_broker()

        self.display_monitor(st.empty())
        st.divider()
        self.display_transact()
        st.divider()
        self.display_performance()


class StrategyPage(BrokerPage):
    
    def generate_strategy(self, prompt: str):
        def _generate():
            nonlocal prompt
            refs = re.findall(r"<(.*)>", prompt)
            for ref in refs:
                ref_path = self.strategy_path / f"{ref}.py"
                if ref_path.exists():
                    prompt = prompt.replace(f"<{ref}>", f"<reference>\n```\n{ref_path.read_text()}\n```\n</reference>")
            full_prompt = """
            Here is the docstring of `Broker` and `Order` in quantitive package `quool`:

            <order_doc>
            {order_doc}
            </order_doc>

            <broker_doc>
            {broker_doc}
            </broker_doc>

            Now, you are a strategy developer who is skilled in the quant tool package `quool`.
            In the following steps, user will submit a strategy development requirement to you.
            What you need to do is generating a strategy according to the requirement. 
            And the strategy must contain the following content:

            <requirements>
            1. Strategy Docstring: A brief description of the strategy, including the purpose and principles of the strategy.
            2. Strategy Code: Generate strategy code that meets the user's requirements. The code contains the following content:
                a. function params: signature `params() -> dict`，the function does not accept parameter input and returns a dictionary, which contains the parameters of the strategy. The function content needs to be provided to the user through the streamlit input box for the strategy needed parameters interface.
                b. function update: signature `update(broker, time, **kwargs)`，the function accepts a Broker object, a time point, and user-provided parameters, and updates the strategy status. The specific implementation of the function inside needs to be implemented according to the strategy idea provided by the user.
                c. (optional) function init: signature `init(broker, time, **kwargs)`，the function accepts a Broker object and user-provided parameters, and initializes the strategy status. The specific implementation of the function inside needs to be implemented according to the strategy idea provided by the user.
                d. (optional) function stop: signature `stop(broker, time, **kwargs)`，the function accepts a Broker object and user-provided parameters, and stops the strategy from running. The specific implementation of the function inside needs to be implemented according to the strategy idea provided by the user.
            </requirements>
            
            The following is a output template:
            
            <template>
            ```python
            '''
            This is a strategy example.
            In this example, we log every trading time upon update.
            '''
            # import streamlit to create parameter on UI
            import streamlit as st
            # ParquetManager is a class to manage parquet files
            # setup_logger is a function to setup logger
            from quool import ParquetManager, Broker, setup_logger
            # other imports can be imported
            import pandas as pd

            def params():
                param = {{}}
                param["param"] = st.number_input("slots")
                return param

            def init(broker: Broker, time: pd.Timestamp, **kwargs):
                '''
                This function runs before the entire backtesting loop
                Broker is a global variable, any variable can be attached to it and used in different functions
                There are some preset for broker:
                    1. logger (logging.Logger): you can use it to log information
                    2. data (pd.DataFrame): market data. 
                        In `init`, broker.data is total historical data used for backtesting
                    3. timepoints (pd.DatetimeIndex): timepoints of the market data. 
                        In `init`, broker.timepoints is total historical timepoints of the market data
                '''

            def update(broker: Broker, time: pd.Timestamp, **kwargs):
                '''
                This function runs in every trading time
                In this function, `broker.logger` stays the same with `init`,
                but `broker.data` and `broker.timepoints` are updated to the current time point.
                And `broker.data_` is the crossection market data.
                In the following code, we display some basic manipulations in broker.
                '''
                # time is a pd.Timestamp object, indicator current time point in backtesting
                broker.logger.info(f"update at {{time}}")
                logger.info(f"kwargs: {{kwargs}}")
                # get market data
                market_at_time = broker.data.loc[time]
                # or
                market_at_time = broker.data_
                # get position
                position = broker.positions
                # get value
                value = broker.get_value(broker.data_)
                # place buy/sell order
                code = "000001.XSHE"
                broker.buy(code=code, quantity=quantity)
                # get indicator
                indicator = kwargs["indicator"].loc[time]
            
            def stop(broker: Broker, time: pd.Timestamp, **kwargs):
                # this function runs after the entire backtesting loop
                broker.logger.info("stop")
            ```
            </template>
            
            In the tamplate above, every comment indicate the aim of the code. You need to write the code according to the task requirement, 
            and add comments to explain the important steps. You can refer to <order_doc> and <broker_doc> tags to understand the usage of Broker and Order.

            Now, please complete the task of strategy development according to the requirements provided to you and the example template. 
            In your reply, please strictly follow the format of the example template within the <template> tag, and do not add any other information.
            The task is shown as follows:

            <task>
            {prompt}
            </task>
            """.format(broker_doc=Broker.__doc__, order_doc=Order.__doc__, prompt=prompt)
            return st.session_state.model.stream(full_prompt)
        return _generate()

    def select_strategy(self):
        strategy = st.sidebar.selectbox("*select an existing strategy*", [strat.stem for strat in self.strategy_path.glob("*.py")])
        if strategy is not None:
            try:
                self.set_strategy(strategy)
            except Exception as e:
                st.toast(str(e), icon="❌")
        else:
            st.sidebar.error("no strategy selected", icon="❌")

    def display_docstring(self):
        if not hasattr(self, "strategy"):
            st.error("no strategy selected", icon="❌")
        else:
            code_placeholder = st.empty()
            code_placeholder.code(self.strategy.__doc__ or "User is too lazy to write docstring")
            if Path(self.strategy.__file__).exists():
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("edit", use_container_width=True):
                        display_editor(self.strategy)
                with col2:
                    if st.button("discard", use_container_width=True):
                        self.delete_strategy()
                        code_placeholder.empty()
                        st.toast("strategy deleted", icon="✅")

    def create_strategy(self):
        name = st.text_input("*strategy name*", value="test")
        file = st.file_uploader("update strategy file", type=["py"])
        if file is not None:
            (self.strategy_path / f"{name}.py").write_text(StringIO(file.getvalue().decode("utf-8")))

    def select_data(self):
        error_nomarket = st.empty()
        if not hasattr(self, "data"):
            error_nomarket.error("no data selected")
        begin, end = st.slider(
            "*select date range for backtesting*", 
            max_value=pd.to_datetime("now").date(),
            min_value=(pd.to_datetime("now") - pd.Timedelta(days=365 * 20)).date(),
            value=((pd.to_datetime("now") - pd.Timedelta(days=365)).date(), pd.to_datetime("now").date()),
            format="YYYY-MM-DD", 
        )
        col1, col2 = st.columns([0.3, 0.7])
        with col1:
            backadj = st.checkbox("backadjust data", value=True)
        with col2:
            extra = st.text_input("extra fields for market loading", value="")
        with st.spinner("Loading market...", show_time=True):
            try:
                self.set_data(begin, end, backadj, extra)
            except Exception as e:
                st.toast(f"Error loading market: {e}", icon="❌")
        error_nomarket.empty()

    @delayed
    def run_strategy(self, name, data, init, update, stop, since, params):
        timepoints = data.index.get_level_values(0).unique()
        broker = Broker(brokid=name)
        broker.logger = setup_logger(
            name=name, file=self.log_path / f"{name}.log", 
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
        broker.store(self.broker_path / f"{broker.brokid}.json", since)
    
    def backtest_strategy(self):
        if not hasattr(self, "data"):
            st.error("no data selected")
            return
        if not hasattr(self, "strategy"):
            st.error("no strategy selected")
            return
        else:
            init = getattr(self.strategy, "init", None)
            update = getattr(self.strategy, "update")
            stop = getattr(self.strategy, "stop", None)
            bio = BytesIO()
            writer = pd.ExcelWriter(bio, engine="xlsxwriter")
            pd.DataFrame([self.strategy_kwargs], index=[self.strategy.__name__]).to_excel(writer, sheet_name="params")
            writer._save()
            bio.seek(0)
            st.download_button(
                "*download param template*", 
                data=bio.read(),
                file_name="param_template.xlsx"
            )
            file = st.file_uploader("*upload param file*", type="xlsx")
            since = st.date_input("*save since*", value=self.data.index.levels[0].min())
            if st.button("Backtest", use_container_width=True):
                if file is not None:
                    self.strategy_kwargs = pd.read_excel(BytesIO(file.getvalue()), index_col=0).to_dict(orient="index")
                else:
                    self.strategy_kwargs = {self.strategy.__name__: self.strategy_kwargs}
                
                for name in st.session_state.param.keys():
                    path = self.log_path / f"{name}.log"
                    if path.exists():
                        path.unlink()
                with st.spinner("Backtesting...", show_time=True):
                    Parallel(
                        n_jobs=min(len(st.session_state.param), 4), 
                        backend="loky"
                    )(self.run_strategy(
                        name=name,
                        data=self.data,
                        init=init,
                        update=update,
                        stop=stop,
                        since=since,
                        params=para,
                    ) for name, para in self.strategy_kwargs.items())
                st.toast(f"strategy {self.strategy.__name__} executed", icon="✅")

    def run(self):
        self.set_market()
        self.select_broker()
        self.select_strategy()

        self.display_docstring()
        self.create_strategy()
        st.divider()
        self.select_data()
        st.divider()
        self.backtest_strategy()


class ChatPage(App):
    
    def select_model(self):
        name = st.sidebar.text_input("*input a model*", value="deepseek-r1:7b")
        base = st.sidebar.text_input("*input a base model*", value="http://localhost:11434")
        key = st.sidebar.text_input("*input an api key*", value="ollama", type="password")
        self.set_model(name, base, key)
    
    def display_chat(self):
        if not hasattr(self, "model"):
            st.error("No model selected. Please select a model first.")
            return
        
        for message in st.session_state.history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        if prompt := st.chat_input("say something ..."):
            with st.chat_message("user"):
                st.write(prompt)
            st.session_state.history.append({"role": "user", "content": prompt})
            with st.chat_message("assistant"):
                response = st.write_stream(st.session_state.model.stream(st.session_state.history))
            st.session_state.history.append({"role": "assistant", "content": response})

    def run(self):
        self.select_model()
        self.display_chat()


class SchedulePage(App):

    def select_schedule(self):
        selection = st.sidebar.selectbox("*input schedule name*", [schedule.stem for schedule in self.schedule_path.glob("*.py")])
        if selection is not None:
            self.set_schedule(selection)
        if not hasattr(self, "schedule"):
            st.sidebar.error("no schedule selected")

    def display_launcher(self):
        if not hasattr(self, "schedule"):
            st.error("*no schedule selected*")
            return
        
        params = self.schedule.params()
        immdiate_start = st.checkbox("*immediate start*", value=True)
        time_delta = st.text_input("*time delta*", value="1day")
        max_iters = st.number_input("*max iterations*", value=-1, step=1, min_value=-1)
        sender = st.text_input("*input sender email*", value=None)
        password = st.text_input("*input password*", value=None, type="password")
        receiver = st.text_input("*input receiver email*", value=None)
        cc = st.text_input("*input cc email*", value=None)

        if not self.get_status(self.schedule.__name__):
            if  st.button("start", use_container_width=True):
                self.start_task(
                    taskid=self.schedule.__name__, 
                    task=self.schedule.run, 
                    time_delta=time_delta,
                    immediate_start=immdiate_start,
                    max_iters=max_iters,
                    sender=sender,
                    password=password,
                    receiver=receiver,
                    cc=cc,
                    **params,
                )
                st.toast(f"schedule {self.schedule.__name__} started", icon="✅")
        else:
            if st.button("stop", use_container_width=True):
                self.stop_task(self.schedule.__name__)
                st.toast(f"schedule {self.schedule.__name__} stopped", icon="❌")

    def run(self):
        self.select_schedule()
        self.display_launcher()

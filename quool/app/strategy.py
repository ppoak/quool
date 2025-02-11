import re
import ollama
import streamlit as st
from pathlib import Path
from quool import Order, Broker
from .tool import STRATEGIES_PATH, display_editor, update_strategy


def generate_strategy(model: str, prompt: str):
    def _generate():
        nonlocal prompt
        refs = re.findall(r"<(.*)>", prompt)
        for ref in refs:
            ref_path = STRATEGIES_PATH / f"{ref}.py"
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
        # some basic parameters exist in quool.app, you can import them if needed
        from quool.app import LOG_PATH
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
        stream = ollama.chat(
            model=model,
            messages=[
                {'role': 'user', 'content': full_prompt},
            ],
            stream=True,
        )
        for word in stream:
            yield word["message"]["content"]
    return _generate()

def display_creator():
    name = st.text_input("*strategy name*", value="test")
    text = st.text_area("*Write your strategy here*")
    col1, col2 = st.columns(2)
    with col1:
        try:
            model = st.selectbox("*select a model*", [model.model for model in ollama.list().models])
        except:
            st.error("no model available")
            return
    with col2:
        clicked = st.button("generate", use_container_width=True)
    strategy_placeholder = st.empty()
    if clicked:
        response = strategy_placeholder.write_stream(generate_strategy(model, text))
        code = re.findall(r"```(python)?\s*([\s\S]*)\s*```", response)[0][-1]
        (STRATEGIES_PATH / f"{name}.py").write_text(code)
        try:
            update_strategy(name)
        except Exception as e:
            st.error(e)
    if st.session_state.get("strategy") is None:
        st.warning("no strategy selected")
    elif not clicked:
        code_placeholder = st.empty()
        code_placeholder.code(Path(st.session_state.strategy.__file__).read_text())
    if (STRATEGIES_PATH / f"{name}.py").exists():
        col1, col2 = st.columns(2)
        with col1:
            if st.button("edit", use_container_width=True):
                display_editor((STRATEGIES_PATH / f"{name}.py").read_text(), name)
        with col2:
            if st.button("discard", use_container_width=True):
                Path(STRATEGIES_PATH / f"{name}.py").unlink()
                st.session_state.strategy = None
                code_placeholder.empty()
                st.toast("strategy deleted", icon="✅")
        
def layout():
    st.title("STRATEGY CREATEOR")
    display_creator()

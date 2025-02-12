try:
    from pathlib import Path
    from functools import partial
    import streamlit as st
    import quool.app.task as task
    import quool.app.tool as tool
    import quool.app.broker as broker
    import quool.app.strategy as strategy


    def layout(
        quotes_path: str | Path,
        app_path: str | Path = "app", 
        refresh_interval: int | str = "5s", 
        keep_kline: int = 240
    ):
        app_path = Path(app_path)
        broker_path = app_path / "broker"
        strategy_path = app_path / "strategy"
        log_path = app_path / "log"
        task_path = app_path / "task"
        app_path.mkdir(parents=True, exist_ok=True)
        broker_path.mkdir(parents=True, exist_ok=True)
        strategy_path.mkdir(parents=True, exist_ok=True)
        log_path.mkdir(parents=True, exist_ok=True)
        task_path.mkdir(parents=True, exist_ok=True)

        broker_page = st.Page(partial(broker.layout, quotes_path=quotes_path, broker_path=broker_path, refresh_interval=refresh_interval, keep_kline=keep_kline), title="Broker", icon="📊", url_path="broker")
        strategy_page = st.Page(partial(strategy.layout, quotes_path=quotes_path, broker_path=broker_path, strategy_path=strategy_path, keep_kline=keep_kline), title="Strategy", icon="💡", url_path="strategy")
        task_page = st.Page(partial(task.layout, task_path=task_path), title="Task", icon="📅", url_path="task")
        pg = st.navigation([broker_page, strategy_page, task_page])
        pg.run()

except ImportError as e:
    print(e)


try:
    import streamlit as st
    import quool.app.backend as backend
    import quool.app.frontend as frontend
    from pathlib import Path

    def run(
        quotes_path: str | Path = None,
        app_path: str | Path = "app",
        refresh_interval: int | str = "5s",
        max_timepoints: int = 240,
        log_level: str = "INFO",
    ):
        broker = frontend.BrokerPage(
            quotes_path=quotes_path, 
            app_path=app_path, 
            refresh_interval=refresh_interval, 
            max_timepoints=max_timepoints, 
            log_level=log_level
        )
        strategy = frontend.StrategyPage(
            quotes_path=quotes_path, 
            app_path=app_path, 
            refresh_interval=refresh_interval, 
            max_timepoints=max_timepoints, 
            log_level=log_level
        )
        schedule = frontend.SchedulePage(
            quotes_path=quotes_path, 
            app_path=app_path, 
            refresh_interval=refresh_interval, 
            max_timepoints=max_timepoints, 
            log_level=log_level
        )
        chat = frontend.ChatPage(
            quotes_path=quotes_path, 
            app_path=app_path, 
            refresh_interval=refresh_interval, 
            max_timepoints=max_timepoints, 
            log_level=log_level
        )
        broker_page = st.Page(broker.run, title="Broker", icon="📊", url_path="broker")
        strategy_page = st.Page(strategy.run, title="Strategy", icon="📈", url_path="strategy")
        schedule_page = st.Page(schedule.run, title="Schedule", icon="📅", url_path="schedule")
        chat_page = st.Page(chat.run, title="Chat", icon="💬", url_path="chat")

        nav = st.navigation([broker_page, strategy_page, schedule_page, chat_page])
        nav.run()
        

except ImportError as e:
    pass

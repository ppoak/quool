import time
import json
import importlib
import threading
import pandas as pd
import akshare as ak
from pathlib import Path
from langchain_openai import OpenAI
from quool.contrib import ParquetManager
from quool import Broker, setup_logger, Emailer


def raw2ricequant(code: str):
    if code.startswith("6"):
        return code + ".XSHG"
    else:
        return code + ".XSHE"

def reimport_module(path: Path, name: str):
    filepath = str(path / f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def fetch_realtime(code_styler: callable = None):
    data = ak.stock_zh_a_spot_em().set_index("代码", drop=True).drop(columns="序号")
    data["open"] = data["最新价"]
    data["high"] = data["最新价"]
    data["low"] = data["最新价"]
    data["close"] = data["最新价"]
    data["volume"] = data["成交量"] * 100
    if code_styler is not None:
        data.index = pd.MultiIndex.from_product([
            [pd.to_datetime("now")], data.index.map(code_styler)
        ], names=["datetime", "code"])
    return data

def fetch_kline(symbol: str, format: bool = True):
    data = ak.stock_zh_a_hist(symbol=symbol)
    if format:
        data = data.drop(columns="股票代码")
        data = data.rename(columns={
            "日期": "datetime", "开盘": "open", "最高": "high", 
            "最低": "low", "收盘": "close", "成交量": "volume"
        })
        data["datetime"] = pd.to_datetime(data["datetime"])
        data.set_index("datetime", inplace=True)
    return data


class App:

    def __init__(
        self,
        quotes_path: str | Path = None,
        app_path: str | Path = "app",
        refresh_interval: int | str = "5s",
        max_timepoints: int = 240,
        log_level: str = "INFO",
    ):
        self.quotes_path = quotes_path
        self.app_path = Path(app_path)
        self.broker_path = self.app_path / "broker"
        self.strategy_path = self.app_path / "strategy"
        self.chat_path = self.app_path / "chat"
        self.schedule_path = self.app_path / "schedule"
        self.status_path = self.app_path / "status"
        self.log_path = self.app_path / "log"
        self.broker_path.mkdir(parents=True, exist_ok=True)
        self.strategy_path.mkdir(parents=True, exist_ok=True)
        self.chat_path.mkdir(parents=True, exist_ok=True)
        self.schedule_path.mkdir(parents=True, exist_ok=True)
        self.status_path.mkdir(parents=True, exist_ok=True)
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.logger = setup_logger("AppManager", file=self.log_path / "app.log", level=log_level)
        self.refresh_interval = refresh_interval
        self.max_timepoint = max_timepoints
        self.logger.debug("AppManager initialized")
    
    def is_trading_time(self):
        if not hasattr(self, "timepoint"):
            return False
        time = self.timepoint[-1]
        if (
            time.date() in ak.tool_trade_date_hist_sina().squeeze().to_list()
            and (
                (time.time() >= pd.to_datetime("09:30:00").time() and time.time() <= pd.to_datetime("11:30:00").time()) 
                or (time.time() >= pd.to_datetime("13:00:00").time() and time.time() <= pd.to_datetime("15:00:00").time())
            )
        ):
            return True
        return False

    def set_data(
        self,
        begin: str, 
        end: str, 
        backadj: bool = True, 
        extra: str = None
    ):
        if self.quotes_path is not None and not (begin is None and end is None):
            begin = begin or pd.Timestamp("2015-01-01")
            end = end or pd.Timestamp.now()
            quotes_day = ParquetManager(self.quotes_path)
            data = quotes_day.read(
                date__ge=begin, date__le=end, index=["date", "code"],
                columns=["open", "high", "low", "close", "volume"] + (extra.split(',') if extra else [])
            )
            if backadj:
                adj = quotes_day.read(
                    date__ge=begin, date__le=end, index=["date", "code"], columns=["adjfactor"]
                )
                data.loc[:, ["open", "high", "low", "close"]] = data[["open", "high", "low", "close"]].mul(adj["adjfactor"], axis=0)
            self.data = data
        
    def set_market(self):
        self.market = fetch_realtime(raw2ricequant)
        self.timepoint = self.market.index.get_level_values(0).unique()
        self.logger.debug("market data initialized")

    def set_broker(self, name: str):
        self.broker = Broker.restore(self.broker_path / f"{name}.json")
        self.logger.debug(f"broker {name} initialized")
    
    def delete_broker(self):
        (self.broker_path / f"{self.broker.brokid}.json").unlink()
        self.logger.debug(f"broker {self.broker.brokid} deleted")
        del self.broker
    
    def store_broker(self):
        self.broker.store(self.broker_path / f"{self.broker.brokid}.json")
        self.logger.debug(f"broker {self.broker.brokid} saved")
    
    def set_strategy(self, name: str = None):
        if name is not None:
            self.strategy = reimport_module(self.app_path / "strategy", name)
            if not hasattr(self.strategy, "update") or not hasattr(self.strategy, "params"):
                raise ValueError("strategy must have `update` and `params` methods.")
            self.strategy_kwargs = self.strategy.params()
            self.logger.debug(f"strategy {name} initialized")
            return
        self.logger.debug(f"no strategy set")
    
    def delete_strategy(self):
        Path(self.strategy.__file__).unlink()
        self.strategy = None
        self.strategy_kwargs = {}
        self.logger.debug(f"strategy deleted")

    def edit_strategy(self, code: str):
        Path(self.strategy.__file__).write_text(code)
        self.logger.debug(f"strategy {self.strategy.__name__} edited")

    def set_model(self, name: str = "gpt-3.5-turbo", base: str = None, key: str = None):
        self.model = OpenAI(name=name, base_url=base, api_key=key)
        self.logger.debug(f"model {name} on {base} with {key} initialized")
    
    def set_chat(self, chatid: str):
        if not hasattr(self, "chat"):
            self.chat = {chatid: []}
        else:
            self.chat[chatid] = []
        self.logger.debug(f"chat {chatid} initialized")
    
    def store_chat(self, chatid: str = None):
        if chatid is None:
            for chatid, message in self.chat.items():
                with open(self.chat_path / f"{chatid}.json", "w") as f:
                    json.dump({chatid: message}, f, ensure_ascii=False, indent=4)
        else:
            with open(self.chat_path / f"{chatid}.json", "w") as f:
                json.dump({chatid: self.chat[chatid]}, f, ensure_ascii=False, indent=4)
        self.logger.debug(f"chat {chatid} saved")
    
    def delete_chat(self, chatid: str):
        del self.chat[chatid]
        (self.chat_path / f"{chatid}.json").unlink()
        self.logger.debug(f"chat {chatid} deleted")

    def set_schedule(self, module: str):
        module = reimport_module(self.app_path / "schedule", module)
        if not hasattr(module, "run") or not hasattr(module, "params"):
            raise ValueError("schedule must have `run` method.")
        self.schedule = module
        self.logger.debug(f"schedule {module} initialized")
    
    def get_status(self, name: str):
        status_file = self.status_path / f"{name}.id"
        if status_file.exists():
            for tid in threading.enumerate():
                if tid.ident == int(status_file.read_text()):
                    return True
        return False
    
    def stop_task(self, name: str):
        status_file = self.status_path / f"{name}.id"
        if status_file.exists():
            for tid in threading.enumerate():
                if tid.ident == int(status_file.read_text()):
                    status_file.unlink()
                    self.logger.debug(f"task {name} stopped")
                    return True
        self.logger.debug(f"task {name} not found")
        return False

    def start_task(
        self, 
        taskid: str,
        task, 
        time_delta: str, 
        immediate_start: bool = True,
        max_iters: int = -1,
        sender: str = None,
        password: str = None,
        receiver: str = None,
        cc: str = None,
        *args, **kwargs
    ):
        status_file = self.status_path / f"{taskid}.id"
        if self.get_status(taskid):
            self.logger.debug(f"skipped task {taskid}: already running")
            return
        
        def task_wrap(*args, **kwargs):
            nonlocal immediate_start, max_iters
            num_iter = 0
            notify_task = task
            if sender and password and receiver:
                notify_task = Emailer.notify(sender, password, receiver, cc)(task)
                self.logger.debug(f"task {taskid} initiated with email notification {sender} ({password}) -> {receiver} ({cc})")
            latest_time = pd.Timestamp('now')
            if callable(time_delta):
                time_interval = time_delta(latest_time)
            else:
                time_interval = pd.Timedelta(time_delta)
            while status_file.exists() and (max_iters < 0 or num_iter < max_iters):
                if immediate_start:
                    latest_time = pd.Timestamp("now")
                    notify_task(*args, **kwargs)
                    num_iter += 1
                    immediate_start = False
                    self.logger.debug(f"task {taskid} runned with immediate start")
                if pd.Timestamp("now") >= latest_time + time_interval:
                    latest_time = pd.Timestamp("now")
                    notify_task(*args, **kwargs)
                    num_iter += 1
                    self.logger.debug(f"task {taskid} runned")
                time.sleep((latest_time + time_interval - pd.Timestamp("now")).total_seconds())
                
        t = threading.Thread(target=task_wrap, args=args, kwargs=kwargs)
        t.daemon = True
        t.start()
        status_file.write_text(str(t.ident))
        return t
    
    def refresh_market(self):
        def _refresh_market():
            if self.timepoint.size >= self.max_timepoint:
                self.market.drop(self.timepoint[0], level=0, inplace=True)
            self.market = pd.concat([self.market, fetch_realtime(raw2ricequant)], axis=0)
            self.timepoint = self.market.index.get_level_values(0).unique()
        t = self.start_task("refresh_market", _refresh_market, time_delta=self.refresh_interval)
        return t

    def refresh_broker(self, run_strats: bool = False):
        def _refresh_broker():
            if self.is_trading_time():
                market_now = self.market.loc[self.timepoint[-1]]
                market_pre = self.market.loc[self.timepoint[-2]]
                market_delta = market_now.copy()
                market_delta["open"] = market_pre["close"]
                market_delta["volmue"] = market_now["volume"] - market_pre["volume"]
                if run_strats:
                    self.strategy.update(self.broker, self.timepoint[-1], **self.strategy_kwargs)
                self.broker.update(self.timepoint[-1], market_delta)
            else:
                self.broker.update(self.timepoint[-1], pd.DataFrame())
            self.store_broker()
        t = self.start_task("refresh_broker", _refresh_broker, time_delta=self.refresh_interval)
        return t

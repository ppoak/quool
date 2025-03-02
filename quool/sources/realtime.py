import pandas as pd
from collections import deque
from quool import Source, proxy_request


def is_trading_time(time: str) -> bool:
    time = pd.to_datetime(time)
    trading_hours = [("09:30:00", "11:30:00"), ("13:00:00", "15:00:00")]

    if time.weekday() >= 5:
        return False
    for start, end in trading_hours:
        if start <= time <= end:
            return True
    return False


def read_realtime(proxies: list[dict] = None):
    url = "https://82.push2.eastmoney.com/api/qt/clist/get"
    params = {
        "pn": "1",
        "pz": "50000",
        "po": "1",
        "np": "1",
        "ut": "bd1d9ddb04089700cf9c27f6f7426281",
        "fltt": "2",
        "invt": "2",
        "fid": "f3",
        "fs": "m:0 t:6,m:0 t:80,m:1 t:2,m:1 t:23,m:0 t:81 s:2048",
        "fields": "f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,"
        "f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152",
        "_": "1623833739532",
    }
    r = proxy_request(url, proxies=proxies, timeout=1, params=params)
    data_json = r.json()
    if not data_json["data"]["diff"]:
        return pd.DataFrame()
    temp_df = pd.DataFrame(data_json["data"]["diff"])
    temp_df.columns = [
        "_",
        "close",
        "change_pct",
        "change_amt",
        "volume",
        "turnover",
        "amplitude",
        "turnover_rate",
        "pe_ratio",
        "volume_ratio",
        "5min_change",
        "code",
        "_",
        "name",
        "high",
        "low",
        "open",
        "prev_close",
        "market_cap",
        "float_market_cap",
        "rise_speed",
        "pb_ratio",
        "60day_change",
        "ytd_change",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
        "-",
    ]
    temp_df.reset_index(inplace=True)
    temp_df["index"] = temp_df.index + 1
    temp_df.rename(columns={"index": "serial_num"}, inplace=True)
    temp_df = temp_df[
        [
            "serial_num",
            "code",
            "name",
            "close",
            "change_pct",
            "change_amt",
            "volume",
            "turnover",
            "amplitude",
            "high",
            "low",
            "open",
            "prev_close",
            "volume_ratio",
            "turnover_rate",
            "pe_ratio",
            "pb_ratio",
            "market_cap",
            "float_market_cap",
            "rise_speed",
            "5min_change",
            "60day_change",
            "ytd_change",
        ]
    ]
    temp_df["close"] = pd.to_numeric(temp_df["close"], errors="coerce")
    temp_df["change_pct"] = pd.to_numeric(temp_df["change_pct"], errors="coerce")
    temp_df["change_amt"] = pd.to_numeric(temp_df["change_amt"], errors="coerce")
    temp_df["volume"] = pd.to_numeric(temp_df["volume"], errors="coerce")
    temp_df["turnover"] = pd.to_numeric(temp_df["turnover"], errors="coerce")
    temp_df["amplitude"] = pd.to_numeric(temp_df["amplitude"], errors="coerce")
    temp_df["high"] = pd.to_numeric(temp_df["high"], errors="coerce")
    temp_df["low"] = pd.to_numeric(temp_df["low"], errors="coerce")
    temp_df["open"] = pd.to_numeric(temp_df["open"], errors="coerce")
    temp_df["prev_close"] = pd.to_numeric(temp_df["close"], errors="coerce")
    temp_df["volume_ratio"] = pd.to_numeric(temp_df["volume_ratio"], errors="coerce")
    temp_df["turnover_rate"] = pd.to_numeric(temp_df["turnover_rate"], errors="coerce")
    temp_df["pe_ratio"] = pd.to_numeric(temp_df["pe_ratio"], errors="coerce")
    temp_df["pb_ratio"] = pd.to_numeric(temp_df["pb_ratio"], errors="coerce")
    temp_df["market_cap"] = pd.to_numeric(temp_df["market_cap"], errors="coerce")
    temp_df["float_market_cap"] = pd.to_numeric(
        temp_df["float_market_cap"], errors="coerce"
    )
    temp_df["rise_speed"] = pd.to_numeric(temp_df["rise_speed"], errors="coerce")
    temp_df["5min_change"] = pd.to_numeric(temp_df["5min_change"], errors="coerce")
    temp_df["60day_change"] = pd.to_numeric(temp_df["60day_change"], errors="coerce")
    temp_df["ytd_change"] = pd.to_numeric(temp_df["ytd_change"], errors="coerce")
    return temp_df.set_index("code")


class RealtimeSource(Source):

    def __init__(self, proxies: list | dict = None, limit: int = 3000):
        self.limit = limit
        self.proxies = proxies
        self._datas = deque([read_realtime(self.proxies)], maxlen=self.limit)
        now = pd.Timestamp("now")
        self._times = deque([now], maxlen=self.limit)

    @property
    def times(self):
        return self._times

    @property
    def time(self):
        return self._times[-1]

    @property
    def datas(self):
        return pd.concat(
            self._datas, axis=0, keys=self._times, names=["datetime", "code"]
        )

    @property
    def data(self):
        return self._datas[-1]

    def update(self):
        now = pd.Timestamp("now")
        if not is_trading_time(self._times[-1]):
            return None
        self._times.append(now)
        self._datas.append(read_realtime(self.proxies))
        return self._datas[-1]

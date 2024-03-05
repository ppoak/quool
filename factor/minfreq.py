import quool
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed


qtmf = quool.Factor("data/quotes-min")
qtdf = quool.Factor("data/quotes-day")

class MinFreqFactor(quool.Factor):

    def get_factor(
        self, name: str, 
        start: str, stop: str,
    ):
        trading_days = qtdf.get_trading_days(start, stop)
        return Parallel(n_jobs=-1, backend='loky')(
            getattr(self, name)(date) for date in trading_days
        )

    def tail_volume_percent(
        self, date: pd.Timestamp
    ) -> pd.DataFrame:
        pass
from .base import (
    fqtd, ffin, BaseFactor
)

class EvaluationFactor(BaseFactor):

    def get_barra_bp_ratio(self, date: str):
        rollback = fqtd.get_trading_days_rollback(date, rollback=252)
        trading_days = fqtd.get_trading_days(start=rollback, stop=date)
        equity = ffin.read('total_equity', start=rollback, stop=date)
        equity = equity.reindex(trading_days).ffill()
        equity = equity.loc[date]
        price = fqtd.read('close', start=date, stop=date).loc[date]
        adjfactor = fqtd.read('adjfactor', start=date, stop=date).loc[date]
        shares = fqtd.read('circulation_a', start=date, stop=date).loc[date]
        size = price * adjfactor * shares
        res = equity / size
        res.name = date
        return res

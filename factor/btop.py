import numpy as np
import pandas as pd
import statsmodels.api as sm
from base import (
    fqtd,fina, BaseFactor
)


class Btop(BaseFactor):

    def get_marketcap(self, date: str):
        shares = fqtd.read("circulation_a", start=date, stop=date)
        price = fqtd.read("close", start=date, stop=date)
        adjfactor = fqtd.read("adjfactor", start=date, stop=date)
        res = (shares * price * adjfactor).loc[date]
        return res
    
    def get_book_to_price(self, date: str):
        bv = fina.read('total_assets,total_liabilities',start=date, stop=date)
        bv['book_value'] = bv['total_assets'] - bv['total_liabilities']
        bv = bv.reset_index().set_index('order_book_id')['book_value']
        marketcap = self.get_marketcap(date)
        res = bv / marketcap
        return res
    

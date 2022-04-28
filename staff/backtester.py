import pandas as pd
from ..tools import *

@pd.api.extensions.register_dataframe_accessor("relocator")
@pd.api.extensions.register_series_accessor("relocator")
class Relocator(Worker):

    def profit(self, weight: pd.Series = None, weight_col: str = None, forward_col: str = None):
        '''calculate profit from weight and forward
        ---------------------------------------------

        weight_col: str, the column name of weight
        forward_col: str, the column name of forward
        '''
        if self.type_ == Worker.TS:
            raise TypeError('Please transform your data into multiindex data')
        
        elif self.type_ == Worker.CS:
            raise TypeError('We cannot calculate the profit by cross section data')
        
        elif self.type_ == Worker.PN and self.is_frame:
            if weight_col is None or forward_col is None:
                raise ValueError('Please specify the weight and forward column')
            return self.data.groupby(level=0).apply(lambda x:
                (x.loc[:, weight_col] * x.loc[:, forward_col]).sum()
                / x.loc[:, weight_col].sum()
            )
        
        elif self.type_ == Worker.PN and not self.is_frame:
            # if you pass a Series in a panel form without weight, we assume that 
            # value is the forward return and weights are all equal
            if weight is None:
                return self.data.groupby(level=0).mean()
            # if other is not None, and data is a Series, we assume that weight is
            # the weight and data is the forward return
            else:
                data = pd.merge(self.data, weight, left_index=True, right_index=True)
                return data.groupby(level=0).apply(
                    lambda x: (x.iloc[:, 0] * x.iloc[:, 1]).sum() / x.iloc[:, 1].sum()
                )

@pd.api.extensions.register_dataframe_accessor("backtester")
class Backtester(Worker):
    pass

import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as st
from ..tools import *


class AnalystError(FrameWorkError):
    pass

@pd.api.extensions.register_dataframe_accessor("regressor")
@pd.api.extensions.register_series_accessor("regressor")
class Regressor(Worker):
    '''Regressor is a staff worker in pandasquant, used for a dataframe
    to perform regression analysis in multiple ways, like ols, logic,
    and so on.
    '''
    
    def ols(self, y: pd.Series = None, x_col: 'str | list' = None, y_col: str = None):
        '''OLS Regression Function
        ---------------------------

        other: Series, assigned y value in a series form
        x_col: list, a list of column names used for regression x values
        y_col: str, the column name used for regression y values
        '''
        def _reg(data):
            y = data.iloc[:, -1]
            x = data.iloc[:, :-1]
            x = sm.add_constant(x)
            model = sm.OLS(y, x).fit()
            t = pd.Series(model.tvalues)
            p = pd.Series(model.pvalues)
            coef = pd.Series(model.params)
            res = pd.concat([coef, t, p], axis=1)
            res.columns = ['coef', 't', 'p']
            return res

        param_status = (y is not None, x_col is not None, y_col is not None)

        if param_status == (True, True, True):
            raise AnalystError('ols', "You can assign either other or x_col and y_col, but not both.")
        elif param_status == (True, False, False):
            if self.type_ == Worker.PN:
                y.index.names = self.data.index.names
            else:
                y.index.name = self.data.index.name
            if not self.is_frame and self.data.name is None:
                self.data.name = 'ols_x'
            if y.name is None:
                y.name = 'ols_y'
            data = pd.merge(self.data, y, left_index=True, right_index=True).dropna()
        elif param_status == (False, True, True):
            data = self.data.loc[:, item2list(x_col) + [y_col]].dropna()
        else:
            raise AnalystError('ols', "You need to assign x_col and y_col both.")

        if self.type_ == Worker.PN:
            return data.groupby(level=0).apply(_reg)
        else:
            return _reg(data)
        
    def logistic(self, y: pd.Series = None, x_col: 'str | list' = None, y_col: str = None):
        '''Logistics Regression Function
        ---------------------------

        y: Series, assigned y value in a series form
        x_col: list or str, a list of column names used for regression x values
        y_col: str, the column name used for regression y values
        '''
        def _reg(data):
            y = data.iloc[:, -1]
            x = data.iloc[:, :-1]
            x = sm.add_constant(x)
            model = sm.Logit(y, x).fit()
            t = pd.Series(model.tvalues)
            p = pd.Series(model.pvalues)
            coef = pd.Series(model.params)
            res = pd.concat([coef, t, p], axis=1)
            res.columns = ['coef', 't', 'p']
            return res

        param_status = (y is not None, x_col is not None, y_col is not None)

        if param_status == (True, True, True):
            raise AnalystError('logistics', "You can assign either other or x_col and y_col, but not both.")
            
        elif param_status == (True, False, False):
            if self.type_ == Worker.PN:
                y.index.names = self.data.index.names
            else:
                y.index.name = self.data.index.name
            if not self.is_frame and self.data.name is None:
                self.data.name = 'logistics_x'
            if y.name is None:
                y.name = 'logistics_y'
            data = pd.merge(self.data, y, left_index=True, right_index=True).dropna()
            
        elif param_status == (False, True, True):
            data = self.data.loc[:, item2list(x_col) + [y_col]].dropna()

        else:
            raise AnalystError('ols', "You need to assign x_col and y_col both.")

        if self.type_ == Worker.PN:
            return data.groupby(level=0).apply(_reg)
        else:
            return _reg(data)            
            
@pd.api.extensions.register_dataframe_accessor("describer")
@pd.api.extensions.register_series_accessor("describer")
class Describer(Worker):
    '''Describer is a staff worker in pandasquant, used for a dataframe
    or a series to perform a series of descriptive analysis, like
    correlation analysis, and so on.
    '''

    def corr(self, other: pd.Series = None, method: str = 'spearman', tvalue = False):
        '''Calculation for correlation matrix
        -------------------------------------

        method: str, the method for calculating correlation function
        tvalue: bool, whether to return t-value of a time-seriesed correlation coefficient
        '''
        if other is not None:
            other = other.copy()
            if self.type_ == Worker.PN:
                other.index.names = self.data.index.names
            else:
                other.index.name = self.data.name
            
            if self.data.name is None:
                self.data.name = 'corr_x'
            if other.name is None:
                other.name = 'corr_y'
            
            data = pd.merge(self.data, other, left_index=True, right_index=True)
        else:
            data = self.data

        if self.type_ == Worker.PN:
            corr = data.groupby(level=0).corr(method=method)
            if tvalue:
                n = corr.index.levels[0].size
                mean = corr.groupby(level=1).mean()
                std = corr.groupby(level=1).std()
                t = mean / std * np.sqrt(n)
                t = t.loc[t.columns, t.columns].replace(np.inf, np.nan).replace(-np.inf, np.nan)
                return t
            return corr
        else:
            return data.corr(method=method)

    def ic(self, forward: pd.Series = None, grouper = None, method: str = 'spearman'):
        '''To calculate ic value
        ------------------------

        other: series, the forward column
        method: str, 'spearman' means rank ic
        '''
        if forward is not None:
            forward = forward.copy()
            if self.type_ == Worker.PN:
                forward.index.names = self.data.index.names
            else:
                forward.index.name = self.data.name
            
            if not self.is_frame and self.data.name is None:
                self.data.name = 'factor'
            if isinstance(forward, pd.Series) and forward.name is None:
                forward.name = 'forward'
            data = pd.merge(self.data, forward, left_index=True, right_index=True)
        else:
            data = self.data.copy()
        
        groupers = [pd.Grouper(level=0)]
        if grouper is not None:
            groupers += item2list(grouper)
        groupers_num = len(groupers)
            
        if self.type_ == Worker.PN:
            ic = data.groupby(groupers).corr(method=method)
            idx = (slice(None),) * groupers_num + (ic.columns[-1],)
            ic = ic.loc[idx, ic.columns[:-1]].droplevel(groupers_num)
            return ic

        elif self.type_ == Worker.CS:
            if groupers_num < 2:
                ic = data.corr(method=method)
            else:
                ic = data.groupby(groupers[1:]).corr(method=method)
            idx = (slice(None),) * (groupers_num - 1) + (ic.columns[-1],)
            if groupers_num < 2:
                return ic.loc[idx, ic.columns[:-1]]
            ic = ic.loc[idx, ic.columns[:-1]].droplevel(groupers_num - 1)
            return ic
        
        else:
            raise AnalystError('ic', 'Timeseries data cannot be used to calculate ic value!')

@pd.api.extensions.register_dataframe_accessor("tester")
@pd.api.extensions.register_series_accessor("tester")
class Tester(Worker):

    def sigtest(self, h0: 'float | pd.Series' = 0):
        '''To apply significant test (t-test, p-value) to see if the data is significant
        -------------------------------------------------------------------------

        h0: float or Series, the hypothesized value
        '''
        def _t(data):
            mean = data.mean(axis=0)
            std = data.std(axis=0)
            size = data.shape[0] - 1
            t = (mean - h0) / std * np.sqrt(size)
            p = st.t.sf(np.abs(t), size) * 2
            return pd.Series({'t': t.values[0], 'p': p[0]})
        
        if self.type_ == Worker.PN:
            return self.data.groupby(level=1).apply(_t)
        
        else:
            return _t(self.data)


if __name__ == "__main__":
    panelframe = pd.DataFrame(np.random.rand(500, 5), index=pd.MultiIndex.from_product(
        [pd.date_range('20100101', periods=100), list('abcde')]
    ), columns=['id1', 'id2', 'id3', 'id4', 'id5'])
    panelseries = pd.Series(np.random.rand(500), index=pd.MultiIndex.from_product(
        [pd.date_range('20100101', periods=100), list('abcde')]
    ), name='id1')
    tsframe = pd.DataFrame(np.random.rand(500, 5), index=pd.date_range('20100101', periods=500),
        columns=['id1', 'id2', 'id3', 'id4', 'id5'])
    tsseries = pd.Series(np.random.rand(500), index=pd.date_range('20100101', periods=500), name='id1')
    csframe = pd.DataFrame(np.random.rand(5, 5), index=list('abcde'), 
        columns=['id1', 'id2', 'id3', 'id4', 'id5'])
    csseries = pd.Series(np.random.rand(5), index=list('abcde'), name='id1')
    print(csseries.describer.corr(csseries))

import quool
from typing import Any

class Regressor(quool.base.Worker):
    """Regressor
    ============

    Regressor is a staff quool.base.Worker in quool, used for a dataframe
    to perform regression analysis in multiple ways
    
    now it supports:
    OLS, linear logistic, WLS
    """

    def ols(
        self,
        y: quool.Series, 
        intercept: bool = True,
        backend: str = 'statsmodels',
        **kwargs,
    ) -> 'quool.Series | Any':
        """OLS Regression Function
        ---------------------------

        y: Series, assigned y value in a series form
        intercept: bool, whether to add a intercept value
        kwargs: some other kwargs passed to backend
        """

    def logistics(
        self,
        y: quool.Series, 
        intercept: bool = True,
        backend: str = 'statsmodels',
        **kwargs,
    ) -> 'quool.Series | Any':
        """Logistics Regression Function
        ---------------------------

        y: Series, assigned y value in a series form
        intercept: bool, whether to add a intercept value
        backend: str, choose between statsmodels and sklearn
        kwargs: some other kwargs passed to backend
        """

    def wls(
        self, 
        y: quool.Series,
        weights: quool.Series,
        intercept: bool = True,
        backend: str = 'statsmodels',
        **kwargs
    ) -> 'quool.Series | Any':
        """WLS(weighted least squares) Regression Function
        ---------------------------

        y: Series, assigned y value in a series form
        weights: Series, higher the weight, higher the proportion of
        intercept: bool, whether to add a intercept value
        kwargs: some other kwargs passed to backend
        """



class Decompositer(quool.base.Worker):
    """Decompositer is a staff quool.base.Worker in bearalpha, used for a dataframe
    or a series to perform a series of diminishing the dimensions, like pca
    """

    def pca(
        self, 
        ncomp: int, 
        backend: str = 'statsmodels', **kwargs
    ) -> 'quool.Series | Any':
        """PCA decomposite
        --------------------
        
        ncomp: int, number of components after decomposition
        backend: str, choice between 'sklearn' and 'statsmodels'
        """


    
class Describer(quool.base.Worker):
    """Describer is a staff quool.base.Worker in bearalpha, used for a dataframe
    or a series to perform a series of descriptive analysis, like
    correlation analysis, and so on.
    """

    def corr(
        self, 
        other: quool.Series = None, 
        method: str = 'spearman', 
        tvalue = False,
    ) -> 'quool.Series | quool.DataFrame':
        """Calculation for correlation matrix
        -------------------------------------

        method: str, the method for calculating correlation function
        tvalue: bool, whether to return t-value of a time-seriesed correlation coefficient
        """

    def ic(
        self, 
        forward: quool.Series = None, 
        grouper = None, 
        method: str = 'spearman'
    ) -> 'quool.Series | quool.DataFrame':
        """To calculate ic value
        ------------------------

        other: series, the forward column
        method: str, 'spearman' means rank ic
        """


class SigTester(quool.base.Worker):

    def ttest(
        self, 
        h0: 'float | quool.Series' = 0
    ) -> 'quool.DataFrame | quool.Series': 
        """To apply significant test (t-test, p-value) to see if the data is significant
        -------------------------------------------------------------------------

        h0: float or Series, the hypothesized value
        """

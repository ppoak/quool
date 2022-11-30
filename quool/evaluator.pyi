import quool


class Evaluator(quool.base.Worker):

    def sharpe(
        self, 
        rf: 'int | float | quool.Series' = 0.04, 
        period: 'int | str' = 'a'
    ) -> 'quool.DataFrame | quool.Series':
        """To Calculate sharpe ratio for the net value curve
        -----------------------------------------------------
        
        rf: int, float or pd.Series, risk free rate, default to 4%,
        period: freqstr or dateoffset, the resample or rolling period
        """

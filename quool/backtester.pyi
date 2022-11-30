import quool
import backtrader as bt


class Relocator(quool.base.Worker):

    def profit(
        self, 
        ret: quool.Series, 
        portfolio: quool.Series = None,
    ) -> quool.Series:
        """calculate profit from weight and forward
        ---------------------------------------------

        ret: pd.Series, the return data in either PN series or TS frame form
        portfolio: pd.Series, the portfolio tag marked by a series, 
            only available when passing a PN
        """

    def networth(
        self, 
        price: 'quool.Series | quool.DataFrame',
    ) -> quool.Series:
        """Calculate the networth curve using normal price data
        --------------------------------------------------------

        price: pd.Series or pd.DataFrame, the price data either in
            MultiIndex form or the TS Matrix form
        return: pd.Series, the networth curve
        """

    def turnover(self, side: str = 'both') -> quool.Series:
        """calculate turnover
        ---------------------

        side: str, choice between "buy", "short" or "both"
        """


class BackTrader(quool.base.Worker):

    def run(
        self, 
        strategy: quool.Strategy = None, 
        cash: float = 1000000,
        indicators: 'quool.Indicator | list' = None,
        analyzers: 'quool.Analyzer | list' = None,
        observers: 'quool.Observer | list' = None,
        coc: bool = False,
        image_path: str = None,
        data_path: str = None,
        show: bool = True,
    ) -> None: 
        """Run a strategy using backtrader backend
        -----------------------------------------
        
        strategy: bt.Strategy
        cash: int, initial cash
        spt: int, stock per trade, defining the least stocks in on trade
        indicators: bt.Indicator or list, a indicator or a list of them
        analyzers: bt.Analyzer or list, an analyzer or a list of them
        observers: bt.Observer or list, a observer or a list of them
        coc: bool, to set whether cheat on close
        image_path: str, path to save backtest image
        data_path: str, path to save backtest data
        show: bool, whether to show the result
        """


    def relocate(
        self,
        portfolio: 'quool.DataFrame | quool.Series' = None,
        cash: float = 1000000,
        analyzers: 'bt.Analyzer | list' = None,
        observers: 'bt.Observer | list' = None,
        coc: bool = False,
        image_path: str = None,
        data_path: str = None,
        show: bool = True,
    ):
        """Test directly from dataframe position information
        -----------------------------------------
        
        portfolio: pd.DataFrame or pd.Series, position information
        spt: int, stock per trade, defining the least stocks in on trade
        ratio: float, retention ration for cash, incase of failure in order
        cash: int, initial cash
        indicators: bt.Indicator or list, a indicator or a list of them
        analyzers: bt.Analyzer or list, an analyzer or a list of them
        observers: bt.Observer or list, a observer or a list of them
        coc: bool, to set whether cheat on close
        image_path: str, path to save backtest image
        data_path: str, path to save backtest data
        show: bool, whether to show the result
        """


class Factester(quool.base.Worker):

    def analyze(
        self,
        price: 'quool.Series | quool.DataFrame',
        marketcap: 'quool.Series | quool.DataFrame' = None,
        grouper: 'quool.Series | quool.DataFrame | dict' = None, 
        benchmark: quool.Series = None,
        periods: 'list | int' = [5, 10, 15], 
        q: int = 5, 
        commission: float = 0.001, 
        commission_type: str = 'both', 
        plot_period: 'int | str' = -1, 
        data_path: str = None, 
        image_path: str = None, 
        show: bool = True
    ): ...
import quool


class Converter(quool.base.Worker):

    def price2ret(
        self, 
        period: 'str | int', 
        open_col: str = 'close', 
        close_col: str = 'close', 
        method: str = 'algret',
        lag: int = 1,
    ) -> 'quool.DataFrame | quool.Series':
        """Convert the price information to return information
        ------------------------------------------------------
        
        period: str or int or DateOffset, if in str and DateOffset format,
            return will be in like resample format, otherwise, you can get rolling
            return formatted data
        open_col: str, if you pass a dataframe, you need to assign which column
            represents open price
        colse_col: str, the same as open_col, but to assign close price
        method: str, choose between 'algret' and 'logret'
        lag: int, define how many day as lagged after the day of calculation forward return
        """

    def cum2diff(
        self,
        grouper = None, 
        period: int = 1, 
        axis: int = 0, 
        keep: bool = True
    ) -> 'quool.DataFrame | quool.Series': ...

    def dummy2category(
        self, 
        dummy_col: list = None, 
        name: str = 'group'
    ) -> 'quool.DataFrame | quool.Series': ...

    def logret2algret(self) -> 'quool.DataFrame | quool.Series': ...

    def resample(self, rule: str, **kwargs) -> 'quool.DataFrame | quool.Series': ...

    def spdatetime(self, level: int = None, axis: int = 0) -> 'quool.DataFrame | quool.Series':
        """Split data with datetime into date and time formatted index
        ------------------------------------------------------------

        level: int, the level the datetime index exists, only available when not matching standard data types
        axis: int, the axis the datetime index exists, only available when not matching standard data types
        """

    def panelize(self) -> 'quool.DataFrame | quool.Series':
        """Panelize a dataframe
        ------------------------
        
        Specifically used for imbalanced panel data, this
        function will help you deal with that
        """

    def reduce_memory(self) -> quool.DataFrame:
        """Reduce the memory usage for a dataframe
        -----------------------------------------

        iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
        """


class PreProcessor(quool.base.Worker):

    def standarize(
        self, 
        method: str = 'zscore', 
        grouper = None
    )  -> 'quool.DataFrame | quool.Series': ...

    def deextreme(
        self,
        method = 'mad', 
        grouper = None, 
        n = None
    ) -> 'quool.DataFrame | quool.Series': ...

    def fillna(
        self, 
        method = 'pad_zero', 
        grouper = None
    ) -> 'quool.DataFrame | quool.Series': ...
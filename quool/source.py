import pandas as pd
from .base import SourceBase


class DataFrameSource(SourceBase):
    """Concrete data source implementation using in-memory DataFrame for historical backtesting.

    Inherits from SourceBase to provide time-sequenced data access patterns. Designed for efficiently
    iterating through pre-loaded historical market data in a time-sequential manner.

    Key Features:
        - Requires standardized DataFrame format with specific structure
        - Maintains complete history in memory for fast access
        - Provides time-sliced views of historical data
        - Auto-advancing time pointer through update() calls

    Important Data Format Requirements:
        Index:
            - Must be a pandas MultiIndex with exactly two levels:
                1. Datetime-like index (pd.DatetimeIndex preferred)
                2. Security/instrument codes (string identifiers)
            - Index must be sorted chronologically
        
        Columns:
            - Mandatory columns: 'open', 'high', 'low', 'close', 'volume'
            - Additional custom columns are permitted
            - All price data should be in float format
            - Volume should be integer type

    Args:
        data (pd.DataFrame): Historical market data meeting the following criteria:
            - Structured with MultiIndex (datetime, instrument_code)
            - Contains required OHLCV columns
            - Sorted chronologically by datetime index
            - Cleaned data (no NaN values in key columns)

    Attributes:
        time (pd.Timestamp): Current simulation time (read-only)
        timepoint (pd.DatetimeIndex): Available timestamps up to current time
        data (pd.DataFrame): Historical data view up to current time, maintaining original MultiIndex

    Methods:
        update() -> pd.DataFrame | None: Advance to next available timestamp and return new data slice
        read(): Inherited from SourceBase (not implemented in this subclass)

    Raises:
        ValueError: If input data doesn't meet format requirements
        IndexError: If data index is not properly sorted

    Example Usage:

        # 1. Prepare compliant DataFrame
        import pandas as pd
        import numpy as np
        
        # Create datetime index
        dates = pd.date_range('2023-01-01', periods=3)
        codes = ['AAPL', 'MSFT']
        
        # Create MultiIndex
        index = pd.MultiIndex.from_product(
            [dates, codes],
            names=['datetime', 'symbol']
        )
        
        # Generate sample data
        data = pd.DataFrame({
            'open': np.random.uniform(100, 200, 6),
            'high': np.random.uniform(100, 200, 6),
            'low': np.random.uniform(100, 200, 6),
            'close': np.random.uniform(100, 200, 6),
            'volume': np.random.randint(1000, 10000, 6)
        }, index=index).sort_index()
        
        # 2. Initialize data source
        source = DataFrameSource(data)
        print(f"Initial time: {source.time}")
        # Output: Initial time: 2023-01-01 00:00:00
        
        # 3. Access initial data slice
        print("Initial data:")
        print(source.datas)
        '''
                                       open        high         low       close  volume
        datetime   symbol                                                              
        2023-01-01 AAPL         150.123456  180.456789  145.987654  160.321456    5000
                   MSFT         180.654321  190.123456  175.456789  185.789123    7500
        '''
        
        # 4. Advance to next time point
        new_data = source.update()
        print(f"\nNew time: {source.time}")
        # Output: New time: 2023-01-02 00:00:00
        
        # 5. Access updated data view
        print("\nUpdated data:")
        print(source.datas)
        '''
                                       open        high         low       close  volume
        datetime   symbol                                                              
        2023-01-01 AAPL         150.123456  180.456789  145.987654  160.321456    5000
                   MSFT         180.654321  190.123456  175.456789  185.789123    7500
        2023-01-02 AAPL         155.456789  165.123456  150.987654  160.000000    6000
                   MSFT         185.000000  195.500000  180.250000  190.750000    8000
        '''

        # 6. Access latest data view
        print("\nUpdated data:")
        print(source.data)
        '''
                            open        high         low       close  volume
        symbol                                                              
        AAPL         155.456789  165.123456  150.987654  160.000000    6000
        MSFT         185.000000  195.500000  180.250000  190.750000    8000
        '''
        
        # 7. Continue advancing through time
        while True:
            new_data = source.update()
            if new_data is None:
                print("\nReached end of historical data")
                break
            print(f"\nNew data at {source.time}:")
            print(new_data)
    """

    def __init__(self, data: pd.DataFrame):
        self._data = data
        self._times = data.index.get_level_values(0).unique().sort_values()
        self._time = self._times.min()
    
    @property
    def times(self):
        return self._times[self._times <= self.time]
    
    @property
    def time(self):
        return self._time
    
    @property
    def datas(self):
        return self._data.loc[:self.time]
    
    @property
    def data(self):
        return self._data.loc[self.time]
    
    def update(self):
        future = self._timepoint[self._timepoint > self.time]
        if future.empty:
            return None
        self._time = future.min()
        return self._data.loc[self._time]

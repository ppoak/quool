import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mplfinance as mpf
from matplotlib.widgets import Cursor
from ..tools import *


class ArtistError(FrameWorkError):
    pass


@pd.api.extensions.register_dataframe_accessor("drawer")
@pd.api.extensions.register_series_accessor("drawer")
class Drawer(Worker):
    '''Drawer is a staff of pandasquant for visulaizing data'''

    def draw(self, kind: str, 
             datetime: str = slice(None), 
             asset: str = slice(None), 
             indicator: str = slice(None), 
             **kwargs):
        '''Draw a image of the given slice of data
        ------------------------------------------

        kind: str, the kind of the plot
        datetime: str, the slice of datetime, default to all time period
        asset: str, the slice of asset, default to all assets
        indicator: str, the slice of indicator, default to all indicators
        kwargs: dict, the kwargs for the plot function
        '''
        plotwised = self._flat(datetime, asset, indicator)
        
        if not isinstance(plotwised, (pd.Series, pd.DataFrame)):
            raise ArtistError('draw', 'Your slice data seems not to be a plotable data')
        
        ax = kwargs.pop('ax', None)
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(12, 8))

        # bar plot
        if isinstance(plotwised, pd.Series) and isinstance(plotwised.index, 
            pd.DatetimeIndex) and kind == "bar":
            ax.bar(plotwised.index, plotwised, **kwargs)
        elif isinstance(plotwised, pd.DataFrame) and isinstance(plotwised.index,
            pd.DatetimeIndex) and kind == "bar":
            bot = 0
            for col in plotwised.columns:
                ax.bar(plotwised.index, plotwised[col], label=col, bottom=bot, **kwargs)
                bot += plotwised[col]
        
        # candle plot
        elif isinstance(plotwised, pd.DataFrame) and isinstance(plotwised.index,
            pd.DatetimeIndex) and kind == "candle" and \
            pd.Index(['open', 'high', 'low', 'close']).isin(plotwised.columns).all():
            mpf.plot(plotwised, ax=ax, style='charles')
                    
        else:
            plotwised.plot(kind=kind, ax=ax, **kwargs)

        Cursor(ax, useblit=False, color='grey', lw=0.5, horizOn=True, vertOn=True)


@pd.api.extensions.register_dataframe_accessor("printer")
@pd.api.extensions.register_series_accessor("printer")
class Printer(Worker):
    
    def display(self, datetime: str = slice(None), asset: str = slice(None),
        indicator: str = slice(None), maxdisplay_length: int = 10, 
        maxdisplay_width: int = 3, title: str = "Table"):
        """Print the dataframe or series in a terminal
        ------------------------------------------

        formatter: pd.DataFrame, the formatter for the dataframe
        """
        printwised = self._flat(datetime, asset, indicator)

        printwised = printwised.reset_index().astype('str')
        
        # the table is too long (over 100 lines), the first and last can be printed
        if printwised.shape[0] >= 100:
            printwised = printwised.iloc[[i for i in range(maxdisplay_length + 1)] + [i for i in range(-maxdisplay_length, 0)]]
            printwised.iloc[maxdisplay_length] = '...'
        # the table is too wide (over 10 columns), the first and last can be printed
        if printwised.shape[1] >= 10:
            printwised = printwised.iloc[:, [i for i in range(maxdisplay_width + 1)] + [i for i in range(-maxdisplay_width, 0)]]
            printwised.iloc[:, maxdisplay_width] = '...'
        
        table = Table(title=title)
        for col in printwised.columns:
            table.add_column(str(col), justify="center", no_wrap=True)
        for row in printwised.index:
            table.add_row(*printwised.loc[row].tolist())

        CONSOLE.print(table)

if __name__ == "__main__":
    pass
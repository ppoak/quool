import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from ..tools import *


class ArtistError(FrameWorkError):
    pass

class Gallery():
    '''Gallery is a context manager, so you can use it like this:
    
    >>> with Gallery(nrows=2, ncols=3, figsize=(12, 8), show=True, path='/tmp/test.png') as (fig, axes):
            axes[0, 0].plot(range(10))
            axes[0, 1].plot(range(10))
            axes[0, 2].plot(range(10))
    
    it will automatically create a figure with assigned columns and rows in figsize,
    and after plotting, it will automatically save the figure to path or show it in a window,
    and at the same time, will set all timeseries index to be displayed in a human-readable format.
    '''
    
    def __init__(self, nrows: int, ncols: int, figsize: tuple = None,
        show: bool = True, path: str = None) -> None:
        self.nrows = nrows
        self.ncols = ncols
        self.figsize = (12 * ncols, 8 * nrows)
        self.show = show
        self.path = path

    def __enter__(self):
        fig, axes = plt.subplots(self.nrows, self.ncols, figsize=self.figsize)
        axes = np.array(axes).reshape((self.nrows, self.ncols))
        self.fig = fig
        self.axes = axes
        return (fig, axes)
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for ax in self.axes.reshape(-1):
            ax.xaxis.set_major_locator(mticker.MaxNLocator())

        if self.path:
            plt.savefig(self.path)
        if self.show:
            plt.show()
            
        return False

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
        if self.type_ == Worker.PN:
            plotwised = self._flat(datetime, asset, indicator)
        
        else:
            if not self.is_frame:
                if self.type_ == Worker.TS:
                    plotwised = self.data.copy().loc[datetime]
                elif self.type_ == Worker.CS:
                    plotwised = self.data.copy().loc[asset]
            else:
                if self.type == Worker.TS:
                    plotwised = self.data.copy().loc[(datetime, indicator)]
                elif self.type_ == Worker.CS:
                    plotwised = self.data.copy().loc[(asset, indicator)]
        
        if not isinstance(plotwised, (pd.Series, pd.DataFrame)):
            raise ArtistError('draw', 'Your slice data seems not to be a plotable data')
        
        if isinstance(plotwised.index, pd.DatetimeIndex):
            plotwised.index = plotwised.index.strftime(r'%Y-%m-%d')

        plotwised.plot(kind=kind, **kwargs)


if __name__ == "__main__":
    tsseries = pd.Series(np.random.rand(100), index=pd.date_range('20200101', periods=100), name='id8')
    panelframe = pd.DataFrame(np.random.rand(500, 5), index=pd.MultiIndex.from_product(
        [pd.date_range('20100101', periods=100), list('abcde')]
    ), columns=['id1', 'id2', 'id3', 'id4', 'id5'])
    with Gallery(1, 2, path='test.png') as (_, axes):
        panelframe['id1'].drawer.draw('line', asset='a', color='red', ax=axes[0, 0])
        tsseries.drawer.draw('line', ax=axes[0, 0].twinx())
        panelframe.drawer.draw('bar', asset='c', ax=axes[0, 1], stacked=True)
    
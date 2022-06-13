import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.widgets import MultiCursor
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
        show: bool = True, path: str = None, xaxis_keep_mask: list = None) -> None:
        self.xaxis_keep_mask = xaxis_keep_mask
        self.nrows = nrows
        self.ncols = ncols
        self.figsize = figsize or (12 * ncols, 8 * nrows)
        self.show = show
        self.path = path

    def __enter__(self):
        fig, axes = plt.subplots(self.nrows, self.ncols, figsize=self.figsize)
        axes = np.array(axes).reshape((self.nrows, self.ncols))
        self.fig = fig
        self.axes = axes
        self.cursor = MultiCursor(fig.canvas, tuple(axes.reshape(-1)), 
            useblit=True, color='grey', lw=0.5, horizOn=True, vertOn=True)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # if self.xaxis_keep_mask is not None:
        #     for i, ax in enumerate(self.axes.reshape(-1)):
        #         if self.xaxis_keep_mask[i]:
        #             ax.xaxis.label.set_visible(False)
        #             ax.tick_params(labelrotation=90)
        #         else:
        #             ax.xaxis.set_major_locator(mticker.MaxNLocator())
        # else:
        #     for ax in self.axes.reshape(-1):
        #         ax.xaxis.set_major_locator(mticker.MaxNLocator())

        if self.path:
            plt.savefig(self.path, pad_inches=0.0, 
                dpi=self.fig.dpi, bbox_inches='tight')
        if self.show:
            plt.show()
        
        plt.close(self.fig)
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
                if self.type_ == Worker.TS:
                    plotwised = self.data.copy().loc[(datetime, indicator)]
                elif self.type_ == Worker.CS:
                    plotwised = self.data.copy().loc[(asset, indicator)]
        
        if not isinstance(plotwised, (pd.Series, pd.DataFrame)):
            raise ArtistError('draw', 'Your slice data seems not to be a plotable data')
        
        if kind == 'bar':
            ax = kwargs.pop('ax', None)
            if ax is not None:
                ax.bar(plotwised.index, plotwised.values, **kwargs)
            else:
                plt.bar(plotwised.index, plotwised.values, **kwargs)
        else:
            plotwised.plot(kind=kind, **kwargs)


@pd.api.extensions.register_dataframe_accessor("printer")
@pd.api.extensions.register_series_accessor("printer")
class Printer(Worker):
    
    def display(self, datetime: str = slice(None), asset: str = slice(None),
        indicator: str = slice(None), maxdisplay: int = 10, title: str = "Table"):
        """Print the dataframe or series in a terminal
        ------------------------------------------

        formatter: pd.DataFrame, the formatter for the dataframe
        """
        if self.type_ == Worker.PN:
            printwised = self._flat(datetime, asset, indicator)
        
        else:
            if not self.is_frame:
                if self.type_ == Worker.TS:
                    printwised = self.data.copy().loc[datetime]
                elif self.type_ == Worker.CS:
                    printwised = self.data.copy().loc[asset]
            else:
                if self.type_ == Worker.TS:
                    printwised = self.data.copy().loc[(datetime, indicator)]
                elif self.type_ == Worker.CS:
                    printwised = self.data.copy().loc[(asset, indicator)]

        printwised = printwised.reset_index()
        
        # the table is too long (over 100 lines), the first and last can be printed
        if printwised.shape[0] >= 100:
            printwised = printwised.iloc[[i for i in range(maxdisplay + 1)] + [i for i in range(-maxdisplay, 0)]]
            printwised.iloc[maxdisplay] = '...'
        
        table = Table(title=title)
        for col in printwised.columns:
            table.add_column(str(col), justify="center", no_wrap=True)
        for row in printwised.index:
            table.add_row(*printwised.loc[row].tolist())

        Console.print(table)

if __name__ == "__main__":
    tsseries = pd.Series(np.random.randint(0, 4, size=500), index=pd.MultiIndex.from_product(
        [pd.date_range('20200101', periods=100), list('abcde')]), name='id8')
    panelframe = pd.DataFrame(np.random.rand(500, 5), index=pd.MultiIndex.from_product(
        [pd.date_range('20200101', periods=100), list('abcde')]
    ), columns=['id1', 'id2', 'id3', 'id4', 'id5'])
    panelframe['group'] = tsseries
    with Gallery(1, 2, path='test.png') as g:
        panelframe.drawer.draw('box', ax=g.axes[0, 0], datetime='20200101')
        panelframe['id1'].drawer.draw('line', asset='a', color='red', ax=g.axes[0, 0])
        tsseries.drawer.draw('line', ax=g.axes[0, 0].twinx())
        panelframe.drawer.draw('bar', asset='c', ax=g.axes[0, 1], stacked=True)
    
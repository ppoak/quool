import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
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

        printwised = printwised.reset_index().astype('str')
        
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
    pass
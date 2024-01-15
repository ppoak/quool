import abc
import numpy as np
import pandas as pd
from .exception import NotRequiredDimError


class Data(abc.ABC):

    def __init__(self, data: pd.DataFrame | pd.Series) -> None:
        if not isinstance(data, (pd.DataFrame, pd.Series)):
            self._data = data
        else:
            if isinstance(data, pd.DataFrame) and data.shape[1] == 1:
                data = data.squeeze()
            self._data = data
    
    def __str__(self) -> str:
        return str(self._data)
    
    def __repr__(self) -> str:
        return self.__str__()

    def __getitem__(self, key):
        return self._data.__getitem__(key)
    
    def __setitem__(self, key, value):
        self._data.__setitem__(key, value)
    
    def __getattr__(self, name):
        if (name.startswith('_') or name.endswith('_') or
            name.startswith('__') or name.endswith('__')):
            if name in self.__dict__:
                return getattr(self, name)
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")                
        return getattr(self._data, name)
    
    def __setattr__(self, name, value):
        if (name.startswith('_') or name.endswith('_') or
            name.startswith('__') or name.endswith('__')):
            super().__setattr__(name, value)
        else:
            setattr(self._data, name, value)

    def __len__(self):
        return len(self._data)
    
    def __iter__(self):
        return iter(self._data)
    
    def __call__(self) -> pd.DataFrame | pd.Series:
        return self._data
    
    def __add__(self, other):
        return Data(self._data.__add__(Data(other)._data))
    
    def __sub__(self, other):
        return Data(self._data.__sub__(Data(other)._data))
    
    def __mul__(self, other):
        return Data(self._data.__mul__(Data(other)._data))

    def __truediv__(self, other):
        return Data(self._data.__truediv__(Data(other)._data))
    
    def __floordiv__(self, other):
        return Data(self._data.__floordiv__(Data(other)._data))
    
    def __mod__(self, other):
        return Data(self._data.__mod__(Data(other)._data))
    
    def __pow__(self, other):
        return Data(self._data.__pow__(Data(other)._data))

    def __and__(self, other):
        return Data(self._data.__and__(Data(other)._data))

    def __xor__(self, other):
        return Data(self._data.__xor__(Data(other)._data))

    def __or__(self, other):
        return Data(self._data.__or__(Data(other)._data))

    @property
    def dimshape(self):
        if isinstance(self._data, pd.Series):
            return (self._data.index.nlevels, )
        return (self._data.index.nlevels, self._data.columns.nlevels)
    
    @property
    def naxes(self):
        return len(self._data.shape)
    
    @property
    def ndims(self):
        return sum(self.dimshape)
    
    @property
    def rowdim(self):
        return self.dimshape[0]
    
    @property
    def coldim(self):
        if isinstance(self._data, pd.Series):
            return None
        return self.dimshape[1]
    
    @property
    def dimnames(self):
        if isinstance(self._data, pd.Series):
            return self._data.index.names
        return self._data.index.names + self._data.columns.names

    @property
    def rowname(self):
        return self._data.index.names

    @property
    def colname(self):
        if isinstance(self._data, pd.Series):
            return None
        return self._data.columns.names

    def swapdim(self, fromdim: int | str, todim: int | str):
        rowdim = self.rowdim
        if self.naxes > 1:
            fromdim = self.dimnames.index(fromdim) if isinstance(fromdim, str) else fromdim
            fromdim = fromdim + self.ndims if fromdim < 0 else fromdim
            todim = self.dimnames.index(todim) if isinstance(todim, str) else todim
            todim = todim + self.ndims if todim < 0 else todim
            if fromdim < rowdim and todim < rowdim:
                # this is on axis 0
                self._data = self._data.swaplevel(i=fromdim, j=todim, axis=0)
            elif fromdim >= rowdim and todim >= rowdim:
                # this is on axis 1
                self._data = self._data.swaplevel(i=fromdim, j=todim, axis=1)
            elif fromdim < rowdim and todim >= rowdim:
                # this is from axis 0 to axis 1
                self._data = self._data.unstack(level=fromdim)
                self._data = self._data.swaplevel(i=-1, j=todim - rowdim, axis=1)
            elif fromdim >= rowdim and todim < rowdim:
                # this is from axis 1 to axis 0
                self._data = self._data.stack(level=int(fromdim - rowdim))
                if self.naxes > 1:
                    self._data = self._data.swaplevel(i=-1, j=todim, axis=0)
                else:
                    self._data = self._data.swaplevel(i=-1, j=todim)
        else:
            if todim < 0:
                # when todim < 0, meaning data needs to be unstacked to extend axes
                self._data = self._data.unstack(level=fromdim)
            else:
                # when todim > 0 or todim is string type (changed to int), naively reorder
                self._data = self._data.swaplevel(i=fromdim, j=todim)

        return self

    def panelize(self):
        if self.rowdim > 1:
            levels = [self._data.index.get_level_values(i).unique() for i in range(self.rowdim)]
            if self._data.shape[0] < np.prod([level.size for level in levels]):
                self._data = self._data.reindex(pd.MultiIndex.from_product(levels), axis=0)
        if self.coldim > 1:
            levels = [self._data.columns.get_level_values(i).unique() for i in range(self.coldim)]
            if self._data.shape[1] < np.prod([level.size for level in levels]):
                self._data = self._data.reindex(pd.MultiIndex.from_product(levels), axis=1)
        return self

    def rank(self, axis=0):
        self._data = self._data.rank(axis=axis)
        return self
    
    def corr(
        self, 
        other: pd.DataFrame | pd.Series | int | str = None, 
        axis: int = 0, 
        level: int | str = None,
        method: str = 'pearson'
    ):
        # DataFrame with Index
        if self.rowdim == 1 and self.naxes > 1 and other is not None:
            result = self._data.corrwith(Data(other)._data, axis=axis, method=method)
        elif self.rowdim == 1 and self.naxes > 1 and other is None:
            result = self._data.corr(method=method)
        # Series with Index
        elif self.rowdim == 1 and self.naxes == 1:
            result = self._data.corr(Data(other)._data, method=method)
        # DataFrame with MultiIndex
        elif self.rowdim > 1 and self.naxes > 1 and other is None and level is not None:
            result = self._data.groupby(level=level).corr(method=method)
        elif self.rowdim > 1 and self.naxes > 1 and other is None and level is None:
            result = self._data.corr(method=method)
        elif self.rowdim > 1 and self.naxes > 1 and other is not None and level is None:
            result = self._data.corrwith(Data(other)._data, axis=axis, method=method)
        elif self.rowdim > 1 and self.naxes > 1 and other is not None and level is not None:
            result = self._data.groupby(level=level).apply(lambda x: x.corrwith(
                Data(other)._data.loc[x.index], axis=axis, method=method))
        # Series with MultiIndex
        elif self.rowdim > 1 and self.naxes == 1 and other is None and level is not None:
            result = self._data.unstack(level=level).corr(method=method)
        elif self.rowdim > 1 and self.naxes == 1 and other is not None and level is None:
            result = self._data.corr(Data(other)._data, method=method)
        elif self.rowdim > 1 and self.naxes == 1 and other is not None and level is not None:
            result = self._data.unstack(level=level).corrwith(
                Data(other)._data.unstack(level=level), method=method, axis=axis)
        
        return Data(result)

    def shift(self, n: int = 1, level: int | str = 0):
        if self.rowdim == 1:
            return Data(self._data.shift(n))
        return Data(self._data.groupby(level=level).shift(n))


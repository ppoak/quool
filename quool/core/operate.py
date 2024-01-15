import abc
import numpy as np
from .data import Data
from .exception import NotRequiredDimError


class OpBase(abc.ABC):

    @abc.abstractmethod
    def __call__(self) -> Data:
        raise NotImplementedError


class NpBase(OpBase):

    def __call__(self, data: Data, func: str, **kwargs):
        return Data(getattr(np, func)(data(), **kwargs))


class NpPairBase(OpBase):

    def __call__(
        self, 
        left: Data, 
        right: Data | int | float, 
        func: str,
        **kwargs
    ):
        if isinstance(right, Data):
            return Data(getattr(np, func)(left(), right(), **kwargs))
        else:
            return Data(getattr(np, func)(left(), right, **kwargs))


class LevelBase(OpBase):
    
    def __call__(self, data: Data, level: int | str, func: str, **kwargs):
        if data.rowdim > 1:
            return Data(getattr(data().groupby(level=level), func)(**kwargs))
        else:
            raise NotRequiredDimError


class AxBase(OpBase):

    def __call__(self, data: Data, axis: int, func: str, **kwargs) -> Data:
        return Data(getattr(data(), func)(axis=axis, **kwargs))


class AxPairBase(OpBase):

    def __call__(
        self, 
        left: Data, 
        right: Data | int | float, 
        axis: int, 
        func: str,
        **kwargs
    ) -> Data:
        if isinstance(right, Data):
            return Data(getattr(left(), func)(axis=axis, other=right(), **kwargs))
        else:
            return Data(getattr(left(), func)(axis=axis, other=right, **kwargs))


class RollBase(OpBase):

    def __call__(self, data: Data, func: str, window: int | float, **kwargs) -> Data:
        if window == 0:
            return Data(getattr(data().expanding(window), func)(**kwargs))
        elif window < 1 and window > 0:
            return Data(getattr(data().ewm(window), func)(**kwargs))
        elif window > 1:
            return Data(getattr(data().rolling(window), func)(**kwargs))
        else:
            raise ValueError('window must be greater than 0')

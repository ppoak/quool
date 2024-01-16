import abc
import numpy as np
from .data import PdData
from .exception import NotRequiredDimError


class OpBase(abc.ABC):

    @abc.abstractmethod
    def __call__(self) -> PdData:
        raise NotImplementedError


class NpBase(OpBase):

    def __call__(self, data: PdData, func: str, **kwargs):
        return PdData(getattr(np, func)(data(), **kwargs))


class NpPairBase(OpBase):

    def __call__(
        self, 
        left: PdData, 
        right: PdData | int | float, 
        func: str,
        **kwargs
    ):
        if isinstance(right, PdData):
            return PdData(getattr(np, func)(left(), right(), **kwargs))
        else:
            return PdData(getattr(np, func)(left(), right, **kwargs))


class LevelBase(OpBase):
    
    def __call__(self, data: PdData, level: int | str, func: str, **kwargs):
        if data.rowdim > 1:
            return PdData(getattr(data().groupby(level=level), func)(**kwargs))
        else:
            raise NotRequiredDimError


class AxBase(OpBase):

    def __call__(self, data: PdData, axis: int, func: str, **kwargs) -> PdData:
        return PdData(getattr(data(), func)(axis=axis, **kwargs))


class AxPairBase(OpBase):

    def __call__(
        self, 
        left: PdData, 
        right: PdData | int | float, 
        axis: int, 
        func: str,
        **kwargs
    ) -> PdData:
        if isinstance(right, PdData):
            return PdData(getattr(left(), func)(axis=axis, other=right(), **kwargs))
        else:
            return PdData(getattr(left(), func)(axis=axis, other=right, **kwargs))


class RollBase(OpBase):

    def __call__(self, data: PdData, func: str, window: int | float, **kwargs) -> PdData:
        if window == 0:
            return PdData(getattr(data().expanding(window), func)(**kwargs))
        elif window < 1 and window > 0:
            return PdData(getattr(data().ewm(window), func)(**kwargs))
        elif window > 1:
            return PdData(getattr(data().rolling(window), func)(**kwargs))
        else:
            raise ValueError('window must be greater than 0')

import abc
import numpy as np
import pandas as pd
from numba import njit
from .database import Dim2Frame
from .exception import NotRequiredDimError


class Factor(Dim2Frame):

    def __init__(
        self, 
        data: pd.DataFrame | pd.Series,
        level: int | str = 0,
    ) -> None:
        super().__init__(data, level)

    def __str__(self) -> str:
        return (f'{self.__class__.__name__} '
                f'from {self.data.index.min().strftime(f"%Y-%m-%d")} '
                f'to {self.data.index.max().strftime(f"%Y-%m-%d")}')
    
    def __repr__(self) -> str:
        return self.__str__()

    def __add__(self, other: 'Factor') -> 'Factor':
        return Add()(self, other)
    
    def __sub__(self, other: 'Factor') -> 'Factor':
        return Sub()(self, other)

    def __mul__(self, other: 'Factor') -> 'Factor':
        return Mul()(self, other)

    def __truediv__(self, other: 'Factor') -> 'Factor':
        return Div()(self, other)

    def __pow__(self, other: 'Factor') -> 'Factor':
        return Pow()(self, other)


class OpBase(abc.ABC):

    @abc.abstractmethod
    def __call__(self) -> Factor:
        raise NotImplementedError


class NpBase(OpBase):

    def __call__(self, factor: Factor, func: str) -> Factor:
        return Factor(getattr(np, func)(factor.data))


class NpPairBase(OpBase):

    def __call__(self, left: Factor, right: Factor | int | float, func: str) -> Factor:
        if isinstance(right, Factor):
            return Factor(getattr(np, func)(left.data, right.data))
        else:
            return Factor(getattr(np, func)(left.data, right))


class AxBase(OpBase):

    def __call__(self, factor: Factor, axis: int, func: str) -> Factor:
        return Factor(getattr(factor.data, func)(axis=axis))


class AxPairBase(OpBase):

    def __call__(self, left: Factor, right: Factor | int | float, axis: int, func: str) -> Factor:
        if isinstance(right, Factor):
            return Factor(getattr(left.data, func)(axis=axis, other=right.data))
        else:
            return Factor(getattr(left.data, func)(axis=axis, other=right))


class RollBase(OpBase):

    def __call__(self, factor: Factor, func: str, window: int | float) -> Factor:
        if window == 0:
            return Factor(getattr(factor.data.expanding(window), func)())
        elif window < 1 and window > 0:
            return Factor(getattr(factor.data.ewm(window), func)())
        elif window > 1:
            return Factor(getattr(factor.data.rolling(window), func)())
        else:
            raise ValueError('window must be greater than 0')


class Abs(NpBase):

    def __call__(self, factor: Factor):
        super().__call__(factor, 'abs')


class Sign(NpBase):

    def __call__(self, factor: Factor):
        super().__call__(factor, 'sign')


class Sqrt(NpBase):

    def __call__(self, factor: Factor):
        super().__call__(factor, 'sqrt')


class Log(NpBase):

    def __call__(self, factor: Factor):
        super().__call__(factor, 'log')


class Add(NpPairBase):

    def __call__(self, left: Factor, right: Factor | int | float) -> Factor:
        return super().__call__(left, right, 'add')

class Sub(NpPairBase):

    def __call__(self, left: Factor, right: Factor | int | float) -> Factor:
        return super().__call__(left, right, 'sub')

class Mul(NpPairBase):

    def __call__(self, left: Factor, right: Factor | int | float) -> Factor:
        return super().__call__(left, right, 'mul')


class Div(NpPairBase):

    def __call__(self, left: Factor, right: Factor | int | float) -> Factor:
        return super().__call__(left, right, 'div')


class Pow(NpPairBase):

    def __call__(self, left: Factor, right: Factor | int | float) -> Factor:
        return super().__call__(left, right, 'pow')


class Corr(AxPairBase):

    def __call__(self, left: Factor, right: Factor) -> Factor:
        return super().__call__(left, right, 1, 'corrwith')


class Layer(AxBase):

    def __call__(self, factor: Factor, ngroup: int) -> Factor:
        return factor.apply(lambda x: pd.qcut(x, ngroup, labels=False), axis=1) + 1


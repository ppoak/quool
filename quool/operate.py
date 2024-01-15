from .core.operate import (
    NpBase, 
    NpPairBase, 
    LevelBase,
    AxBase,
    AxPairBase, 
    RollBase,
    Data
)


class Abs(NpBase):

    def __call__(self, data: Data):
        super().__call__(data, 'abs')

class Sign(NpBase):

    def __call__(self, data: Data):
        super().__call__(data, 'sign')

class Sqrt(NpBase):

    def __call__(self, data: Data):
        super().__call__(data, 'sqrt')

class Log(NpBase):

    def __call__(self, data: Data):
        super().__call__(data, 'log')

class Add(NpPairBase):

    def __call__(self, left: Data, right: Data | int | float):
        return super().__call__(left, right, 'add')

class Sub(NpPairBase):

    def __call__(self, left: Data, right: Data | int | float):
        return super().__call__(left, right, 'sub')

class Mul(NpPairBase):

    def __call__(self, left: Data, right: Data | int | float):
        return super().__call__(left, right, 'mul')

class Div(NpPairBase):

    def __call__(self, left: Data, right: Data | int | float):
        return super().__call__(left, right, 'div')

class Pow(NpPairBase):

    def __call__(self, left: Data, right: Data | int | float):
        return super().__call__(left, right, 'pow')

class LevelShift(LevelBase):

    def __call__(self, data: Data, level: int ,n: int):
        return super().__call__(data, level, 'shift', periods=n)

class Shift(AxBase):

    def __call__(self, data: Data, n: int):
        return super().__call__(data, 0, 'shift', periods=n)

class CsCorr(AxPairBase):

    def __call__(self, left: Data, right: Data, method: str = 'pearson'):
        return super().__call__(left, right, 1, 'corrwith', method=method)

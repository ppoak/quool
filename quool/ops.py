import numpy as np


def abs(x: np.ndarray):
    return np.abs(x)

def neg(x: np.ndarray):
    return np.negative(x)

def sign(a: np.ndarray):
    return np.sign(a)

def sqrt(x: 'np.ndarray | int | float'):
    if isinstance(x, (int, float)):
        if x < 0:
            x = -x
    else:
        x = np.abs(x)
    return np.sqrt(x)

def ssqrt(x: 'np.ndarray | int | float'):
    if isinstance(x, (int, float)):
        if x < 0:
            x = -x
            sign = -1
        else:
            sign = 1
    else:
        sign = np.ones_like(x)
        sign[x<0] = -1
        x = np.abs(x)
    return sign * np.sqrt(x)

def square(x: np.ndarray):
    return x ** 2

def csrank(x: np.ndarray):
    x = x.astype('float64')
    x[np.isinf(x)] = np.nan
    x = np.nan_to_num(x)
    x = x.argsort(axis=1).argsort(axis=1)
    for row in x:
        if np.count_nonzero(row==0) == len(row):
            row[row==0] = np.nan
    return x

def csnorm(x: np.ndarray) -> np.ndarray:
    res = np.full_like(x, fill_value=np.nan)
    x = x.astype('float64')
    x[np.isinf(x)] = np.nan
    x = np.nan_to_num(x)
    begin = 0
    for idx, row in enumerate(x):
        if np.count_nonzero(row==0) == len(row):
            continue
        else:
           begin = idx 
           break
    for row in range(begin, x.shape[0]):
        cs_mean = np.nanmean(x[row, :])
        cs_std = np.nanstd(x[row, :])
        res[row, :] = (x[row, :] - cs_mean) / cs_std
    return res

def add(x: np.ndarray, b: 'np.ndarray | float'):
    return x + b

def sub(x: np.ndarray, b: 'np.ndarray | float'):
    return x - b

def mul(x: np.ndarray, b: 'np.ndarray | float'):
    if isinstance(b, (float)):
        if abs(b-0) < 1e-2:
            b = 1e-1
    return x * b

def div(x: np.ndarray, b: 'np.ndarray | float'):
    if isinstance(b, (int, float)):
        if b >= 0:
            b = max(1e-1, b)
        else:
            b = min(1e-1, b)
    return x / b

def power(a: np.array, b: 'float|int'):
    if b < 0:
        b = -b
    return np.power(a, b)

def maximum(x: np.ndarray, d: 'float|int'):
    x[x<d] = d
    x = x.astype('float64')
    x[np.isinf(x)] = np.nan
    x = np.nan_to_num(x)
    for row in x:
        if np.count_nonzero(row==0) == len(row):
            row[row==0] = np.nan
    return x

def minimum(x: np.ndarray, d: 'float|int'):
    x[x>d] = d
    x = x.astype('float64')
    x[np.isinf(x)] = np.nan
    x = np.nan_to_num(x)
    for row in x:
        if np.count_nonzero(row==0) == len(row):
            row[row==0] = np.nan
    return x

def log(x: np.array, d: float):
    return np.log(abs(x)) / np.log(d)

def sum(x: np.ndarray, d: int):
    mat = np.zeros((x.shape[0], x.shape[0]-d+1))
    x = x.astype('float64')
    x[np.isinf(x)] = np.nan
    x = np.nan_to_num(x)
    for i in range(mat.shape[0]-d+1):
        mat[i:(i + d), i] = 1
    res =  x.T @ mat
    res = np.hstack((np.full((x.shape[1], d-1), fill_value=np.nan), res))
    for row in res:
        if np.count_nonzero(row==0) == len(row):
            row[row==0] = np.nan
    return res.T

def mean(x: np.ndarray, d: int) -> np.ndarray:
    ma = np.zeros((x.shape[0], x.shape[0]-d+1))
    x = x.astype('float64')
    x[np.isinf(x)] = np.nan
    x = np.nan_to_num(x)
    for i in range(ma.shape[0]-d+1):
        ma[i:(i + d), i] = 1/d
    res =  x.T @ ma
    res = np.hstack((np.full((x.shape[1], d-1), fill_value=np.nan), res))
    for row in res:
        if np.count_nonzero(row==0) == len(row):
            row[row==0] = np.nan
    return res.T

def wma(x: np.ndarray, d: int) -> np.ndarray:
    wma = np.zeros((x.shape[0], x.shape[0]-d+1))
    x = x.astype('float64')
    x[np.isinf(x)] = np.nan
    x = np.nan_to_num(x)
    denominator = np.sum([i for i in range(1, d+1)])
    for i in range(wma.shape[0]-d+1):
        for j in range(1, d+1):
            wma[(i + j - 1):(i + j), i] = j / denominator
    res =  x.T @ wma
    res = np.hstack((np.full((x.shape[1], d-1), fill_value=np.nan), res))
    for row in res:
        if np.count_nonzero(row==0) == len(row):
            row[row==0] = np.nan
    return res.T

def ema(x: np.ndarray, d: int) -> np.ndarray:
    ema = np.zeros((x.shape[0], x.shape[0]-d+1))
    x = x.astype('float64')
    x[np.isinf(x)] = np.nan
    x = np.nan_to_num(x)
    denominator = np.sum(np.exp(np.arange(1, d + 1)))
    for i in range(ema.shape[0]-d+1):
        for j in range(1, d+1):
            ema[(i + j - 1):(i + j), i] = np.exp(j) / denominator
    res =  x.T @ ema
    res = np.hstack((np.full((x.shape[1], d-1), fill_value=np.nan), res))
    for row in res:
        if np.count_nonzero(row==0) == len(row):
            row[row==0] = np.nan
    return res.T

def var(x: np.ndarray, d: int) -> np.ndarray:
    res = np.full_like(x, fill_value=np.nan)
    x = x.astype('float64')
    x[np.isinf(x)] = np.nan
    x = np.nan_to_num(x)
    begin = 0
    for idx, row in enumerate(x):
        if np.count_nonzero(row==0) == len(row):
            continue
        else:
           begin = idx 
           break
    for row in range(begin, x.shape[0] - d + 1):
        v = np.nanvar(x[row:row + d], axis=0)
        res[row + d - 1, :] = v
    return res

def skew(x: np.ndarray, d: int) -> np.ndarray:
    res = np.full_like(x, fill_value=np.nan)
    x = x.astype('float64')
    x[np.isinf(x)] = np.nan
    x = np.nan_to_num(x)
    begin = 0
    for idx, row in enumerate(x):
        if np.count_nonzero(row==0) == len(row):
            continue
        else:
           begin = idx 
           break
    for row in range(begin, x.shape[0] - d + 1):
        tmp = x[row:row + d]
        s = np.sum((tmp - np.nanmean(tmp, axis=0)) ** 3, axis=0) / \
                (np.count_nonzero(~np.isnan(tmp), axis=0) * np.nanstd(tmp, axis=0) ** 3)
        res[row + d - 1, :] = s
    return res

def kurt(x: np.ndarray, d: int) -> np.ndarray:
    res = np.full_like(x, fill_value=np.nan)
    x = x.astype('float64')
    x[np.isinf(x)] = np.nan
    x = np.nan_to_num(x)
    begin = 0
    for idx, row in enumerate(x):
        if np.count_nonzero(row==0) == len(row):
            continue
        else:
           begin = idx 
           break
    for row in range(begin, x.shape[0] - d + 1):
        tmp = x[row:row + d]
        tmpm = tmp.mean(axis=0)
        tmps = tmp.std(axis=0)
        tmp = (tmp - tmpm) / tmps
        k = (tmp ** 4).mean(axis=0) - 3
        res[row + d - 1, :] = k
    return res

def max(x: np.array, d: int) -> np.ndarray:
    res = np.full_like(x, fill_value=np.nan)
    x = x.astype('float64')
    x[np.isinf(x)] = np.nan
    x = np.nan_to_num(x)
    begin = 0
    for idx, row in enumerate(x):
        if np.count_nonzero(row==0) == len(row):
            continue
        else:
           begin = idx 
           break
    for row in range(begin, x.shape[0] - d + 1):
        m = np.nanmax(x[row:row + d], axis=0)
        res[row + d - 1, :] = m
    return res

def min(x: np.array, d: int) -> np.ndarray:
    res = np.full_like(x, fill_value=np.nan)
    x = x.astype('float64')
    x[np.isinf(x)] = np.nan
    x = np.nan_to_num(x)
    begin = 0
    for idx, row in enumerate(x):
        if np.count_nonzero(row==0) == len(row):
            continue
        else:
           begin = idx 
           break
    for row in range(begin, x.shape[0] - d + 1):
        m = np.nanmin(x[row:row + d], axis=0)
        res[row + d - 1, :] = m
    return res

def delta(x: np.array, d: int) -> np.ndarray:
    res = np.full_like(x, fill_value=np.nan)
    x = x.astype('float64')
    x[np.isinf(x)] = np.nan
    x = np.nan_to_num(x)
    begin = 0
    for idx, row in enumerate(x):
        if np.count_nonzero(row==0) == len(row):
            continue
        else:
           begin = idx 
           break
    for row in range(begin, x.shape[0] - d + 1):
        tmp = x[row:row + d]
        res[row + d - 1, :] = tmp[-1, :] - tmp[0, :]
    return res

def delay(x: np.ndarray, d: int) -> np.ndarray:
    x = x.astype('float64')
    x[np.isinf(x)] = np.nan
    x = np.nan_to_num(x)
    res = np.roll(x, d, axis=0)
    res[:d, :] = np.nan
    for row in res:
        if np.count_nonzero(row==0) == len(row):
            row[:] = np.nan
    return res

def rank(x: np.ndarray, d: int) -> np.ndarray:
    res = np.full_like(x, fill_value=np.nan)
    x = x.astype('float64')
    x[np.isinf(x)] = np.nan
    x = np.nan_to_num(x)
    begin = 0
    for idx, row in enumerate(x):
        if np.count_nonzero(row==0) == len(row):
            continue
        else:
           begin = idx 
           break
    for row in range(begin, x.shape[0] - d + 1):
        rank = x[row:row + d].argsort(axis=0).argsort(axis=0)
        res[row + d - 1, :] = rank[-1, :]
    return res

def scale(x: np.ndarray, a: int = 1) -> np.ndarray:
    x = x.astype('float64')
    x[np.isinf(x)] = np.nan
    x = np.nan_to_num(x)
    s = np.sum(x, axis=1, keepdims=1)
    x = (x/s) * a
    return x

def product(x: np.ndarray, d: int) -> np.ndarray:
    res = np.full_like(x, fill_value=np.nan)
    x = x.astype('float64')
    x[np.isinf(x)] = np.nan
    x = np.nan_to_num(x)
    begin = 0
    for idx, row in enumerate(x):
        if np.count_nonzero(row==0) == len(row):
            continue
        else:
           begin = idx 
           break
    for row in range(begin, x.shape[0] - d + 1):
        tmp = x[row:row + d]
        m = np.prod(tmp, axis=0)
        res[row + d - 1, :] = m
    return res

def decay_linear(x: np.ndarray, d: int) -> np.ndarray:
    res = np.full_like(x, fill_value=np.nan)
    x = x.astype('float64')
    x[np.isinf(x)] = np.nan
    x = np.nan_to_num(x)
    begin = 0
    for idx, row in enumerate(x):
        if np.count_nonzero(row==0) == len(row):
            continue
        else:
           begin = idx 
           break
    for row in range(begin, x.shape[0] - d + 1):
        tmp = x[row:row + d]
        s = np.sum(tmp, axis=0)
        res[row + d - 1, :] = x[row + d - 1, :] / s
    return res

def std(x: np.ndarray, d: int):
    res = np.full_like(x, fill_value=np.nan)
    x = x.astype('float64')
    x[np.isinf(x)] = np.nan
    x = np.nan_to_num(x)
    begin = 0
    for idx, row in enumerate(x):
        if np.count_nonzero(row==0) == len(row):
            continue
        else:
           begin = idx 
           break
    for row in range(begin, x.shape[0] - d + 1):
        res[row + d - 1, :] = np.nanstd(x[row:row + d], axis=0)
    return res

def tsnorm(x: np.ndarray, d: int) -> np.ndarray:
    res = np.full_like(x, fill_value=np.nan)
    x = x.astype('float64')
    x[np.isinf(x)] = np.nan
    x = np.nan_to_num(x)
    begin = 0
    for idx, row in enumerate(x):
        if np.count_nonzero(row==0) == len(row):
            continue
        else:
           begin = idx 
           break
    for row in range(begin, x.shape[0] - d + 1):
        ts_mean = np.nanmean(x[row:row+d], axis=0)
        ts_std = np.nanstd(x[row:row+d], axis=0)
        res[row + d - 1, :] = (x[row + d - 1] - ts_mean) / ts_std
    return res

def ifelse(x: np.ndarray, a: 'np.ndarray|int|float', cond: str, y: 'np.ndarray|int|float', z: 'np.ndarray|int|float'):
    if isinstance(a, (int, float)) and isinstance(y, np.ndarray):
        mask = eval(f'x {cond} a')
        res = np.where(mask, y, z)
        return res
    if isinstance(a, (int, float)) and isinstance(y, (int, float)) and isinstance(z, np.ndarray):
        y = np.full_like(x, y)
        mask = eval(f'x {cond} a')
        res = np.where(mask, y, z)
        return res
    if isinstance(a, np.ndarray) and isinstance(y, np.ndarray) and isinstance(z, (int, float)):
        z = np.full_like(x, z)
        mask = eval(f'x {cond} a')
        res = np.where(mask, y, z)
        return res

def correlation(x: np.ndarray, y: np.ndarray, d: int):
    x = x.astype(np.float64); y = y.astype(np.float64)
    res = np.full_like(x, fill_value=np.nan)
    for row in range(x.shape[0] - d + 1):
        x_demean = np.nan_to_num(x - np.nanmean(x[row:row + d], axis=0).repeat(x.shape[0]).reshape(x.shape[1], x.shape[0]).T)
        y_demean = np.nan_to_num(y - np.nanmean(y[row:row + d], axis=0).repeat(x.shape[0]).reshape(x.shape[1], x.shape[0]).T)
        res[row + d - 1, :] =(x_demean.T @ y_demean).diagonal() \
                / (np.linalg.norm(x_demean, axis=0) * np.linalg.norm(y_demean, axis=0))
    return res

def covariance(x: np.ndarray, y: np.ndarray,d: int):
    res = np.full_like(x, fill_value=np.nan)
    for row in range(x.shape[0] - d + 1):
        x_demean = np.nan_to_num(x - np.nanmean(x[row:row + d], axis=0).repeat(x.shape[0]).reshape(x.shape[1], x.shape[0]).T)
        y_demean = np.nan_to_num(y - np.nanmean(y[row:row + d], axis=0).repeat(x.shape[0]).reshape(x.shape[1], x.shape[0]).T)
        res[row + d - 1, :] = (x_demean.T @ y_demean).diagonal() / (d - 1)
    return res


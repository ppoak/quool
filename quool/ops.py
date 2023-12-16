import numpy as np


# one-element operators
def abs(a: np.ndarray) -> np.ndarray:
    return np.abs(a)

def neg(a: np.ndarray) -> np.ndarray:
    return np.negative(a)

def sign(a: np.ndarray) -> np.ndarray:
    return np.sign(a)

def sqrt(a: 'np.ndarray | int | float') -> np.ndarray:
    return np.sqrt(np.abs(a))

def ssqrt(a: 'np.ndarray | int | float') -> np.ndarray:
    return np.sign(a) * np.sqrt(np.abs(a))

def square(a: np.ndarray) -> np.ndarray:
    return a ** 2

def log(a: np.ndarray) -> np.ndarray:
    return np.log(a)

def abslog(a: np.ndarray) -> np.ndarray:
    return np.log(np.abs(a))

def csrank(a: np.ndarray) -> np.ndarray:
    return a.argsort(axis=1).argsort(axis=1)

def csnorm(a: np.ndarray) -> np.ndarray:
    mean = np.mean(a, axis=1).reshape(-1, 1)
    std = np.std(a, axis=1).reshape(-1, 1)
    return (a - mean) / std

# two-element operators
def add(a: np.ndarray, b: 'np.ndarray | float') -> np.ndarray:
    return a + b

def sub(a: np.ndarray, b: 'np.ndarray | float') -> np.ndarray:
    return a - b

def mul(a: np.ndarray, b: 'np.ndarray | float') -> np.ndarray:
    return a * b

def div(a: np.ndarray, b: 'np.ndarray | float') -> np.ndarray:
    b = np.clip(b, 1e-4, np.inf)
    return a / b

def power(a: np.array, b: float | int) -> np.ndarray:
    a = np.clip(a, 1e-4, np.inf)
    return np.power(a, b)

def maximum(a: np.ndarray, b: np.ndarray | float | int) -> np.ndarray:
    return np.clip(a, 0, b)

def minimum(a: np.ndarray, b: np.ndarray | float | int) -> np.ndarray:
    return np.clip(a, b, 0)

# one-rolling operators
def rsum(a: np.ndarray, w: int) -> np.ndarray:
    res = np.full_like(a, np.nan)
    for i in range(w, a.shape[0]):
        s = (i - w) % i
        res[i - 1] = np.nansum(a[s:i], axis=0)
    return res

def rmean(a: np.ndarray, w: int) -> np.ndarray:
    res = np.full_like(a, np.nan)
    for i in range(w, a.shape[0]):
        s = (i - w) % i
        res[i - 1] = np.nanmean(a[s:i], axis=0)
    return res

def var(a: np.ndarray, w: int) -> np.ndarray:
    res = np.full_like(a, np.nan)
    for i in range(w, a.shape[0]):
        s = (i - w) % i
        res[i - 1] = np.nanvar(a[s:i], axis=0)
    return res

def skew(a: np.ndarray, w: int) -> np.ndarray:
    res = np.full_like(a, np.nan)
    for i in range(w, a.shape[0]):
        s = (i - w) % i
        mean = np.nanmean(a[s:i], axis=0)
        std = np.nanstd(a[s:i], axis=0, ddof=1)
        res[i - 1] = (np.sum((a[s:i] - mean) ** 3) / w) / (std ** 3)
    return res

def kurt(a: np.ndarray, w: int) -> np.ndarray:
    res = np.full_like(a, np.nan)
    for i in range(w, a.shape[0]):
        s = (i - w) % i
        mean = np.nanmean(a[s:i], axis=0)
        std = np.nanstd(a[s:i], axis=0, ddof=1)
        res[i - 1] = (np.sum((a[s:i] - mean) ** 4) / w) / (std ** 4) - 3
    return res

def wma(x: np.ndarray, d: int) -> np.ndarray:
    wma = np.zeros((x.shape[0], x.shape[0]-d+1))
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

def ema(a: np.ndarray, w: int) -> np.ndarray:
    res = np.full_like(a, np.nan)
    res[0] = a[0]
    for i in range(1, a.shape[0]):
        res[i] = 2 / (1 + w) * a[i] + (w - 1) / (1 + w) * res[i - 1]
    return res

def max(x: np.array, d: int) -> np.ndarray:
    res = np.full_like(x, fill_value=np.nan)
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
    res = np.roll(x, d, axis=0)
    res[:d, :] = np.nan
    return res

def rank(x: np.ndarray, d: int) -> np.ndarray:
    res = np.full_like(x, fill_value=np.nan)
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
    x[np.isinf(x)] = np.nan
    x = np.nan_to_num(x)
    s = np.sum(x, axis=1, keepdims=1)
    x = (x/s) * a
    return x

def product(x: np.ndarray, d: int) -> np.ndarray:
    res = np.full_like(x, fill_value=np.nan)
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


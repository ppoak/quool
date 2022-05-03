import numpy as np
import pandas as pd
from ..tools import *


class ProcessorError(FrameWorkError):
    pass

@pd.api.extensions.register_dataframe_accessor("converter")
@pd.api.extensions.register_series_accessor("converter")
class Converter(Worker):
    
    def price2ret(self, period: str, open_column: str = 'close', close_column: str = 'close'):
        if self.type_ == Worker.PN and self.is_frame:
            # https://pandas.pydata.org/docs/reference/api/pandas.Grouper.html
            # https://stackoverflow.com/questions/15799162/
            close_price = self.data.groupby([
                pd.Grouper(level=0, freq=period, label='right'),
                pd.Grouper(level=1)
            ]).last().loc[:, close_column]
            open_price = self.data.groupby([
                pd.Grouper(level=0, freq=period, label='right'),
                pd.Grouper(level=1)
            ]).first().loc[:, open_column]

        elif self.type_ == Worker.PN and not self.is_frame:
            # if passing a series in panel form, assuming that
            # it is the only way to figure out a return
            close_price = self.data.groupby([
                pd.Grouper(level=0, freq=period, label='right'),
                pd.Grouper(level=1)
            ]).last()
            open_price = self.data.groupby([
                pd.Grouper(level=0, freq=period, label='right'),
                pd.Grouper(level=1)
            ]).first()

        elif self.type_ == Worker.TS:
            close_price = self.data.\
                resample(period, label='right').last()
            open_price = self.data.\
                resample(period, label='right').first()
            
        else:
            raise ProcessorError('price2ret', 'Can only convert time series data to return')

        return (close_price - open_price) / open_price

    def price2fwd(self, period: str, open_column: str = 'open', close_column: str = 'close'):
        if self.type_ == Worker.PN and self.is_frame:
            # https://pandas.pydata.org/docs/reference/api/pandas.Grouper.html
            # https://stackoverflow.com/questions/15799162/
            close_price = self.data.groupby([
                pd.Grouper(level=0, freq=period, label='left'),
                pd.Grouper(level=1)
            ]).last().loc[:, close_column]
            open_price = self.data.groupby([
                pd.Grouper(level=0, freq=period, label='left'),
                pd.Grouper(level=1)
            ]).first().loc[:, open_column]

        elif self.type_ == Worker.PN and not self.is_frame:
            # if passing a series in panel form, assuming that
            # it is the only way to figure out a return
            close_price = self.data.groupby([
                pd.Grouper(level=0, freq=period, label='right'),
                pd.Grouper(level=1)
            ]).last()
            open_price = self.data.groupby([
                pd.Grouper(level=0, freq=period, label='right'),
                pd.Grouper(level=1)
            ]).first()
        
        elif self.type_ == Worker.TS:
            close_price = self.data.\
                resample(period, label='left').last()
            open_price = self.data.\
                resample(period, label='left').first()
        else:
            raise ProcessorError('price2fwd', 'Can only convert time series data to forward')

        return (close_price - open_price) / open_price
        
    def cum2diff(self, grouper = None, period: int = 1, axis: int = 0, keep: bool = True):
        def _diff(data):
            diff = data.diff(period, axis=axis)
            if keep:
                diff.iloc[:period] = data.iloc[:period]
            return diff
        
        if grouper is None:
            diff = _diff(self.data)
        else:
            diff = self.data.groupby(grouper).apply(lambda x: x.groupby(level=1).apply(_diff))
            
        return diff

    def dummy2category(self, dummy_col: list = None, name: str = 'group'):
        if not self.is_frame:
            raise ProcessorError('dummy2category', 'Can only convert dataframe to category')
            
        if dummy_col is None:
            dummy_col = self.data.columns
        
        columns = pd.DataFrame(
            dummy_col.values.reshape((1, -1))\
            .repeat(self.data.shape[0], axis=0),
            index=self.data.index, columns=dummy_col
        )
        # fill nan value with 0, because bool(np.nan) is true
        category = columns[self.data.loc[:, dummy_col].fillna(0).astype('bool')]\
            .replace(np.nan, '').astype('str').sum(axis=1)
        category.name = name
        return category

    def logret2algret(self):
        return np.exp(self.data) - 1
    
    def algret2logret(self):
        return np.log(self.data)

    def resample(self, rule: str, **kwargs):
        if self.type_ == Worker.TS:
            return self.data.resample(rule, **kwargs)
        elif self.type_ == Worker.PN:
            return self.data.groupby([pd.Grouper(level=0, freq=rule, **kwargs), pd.Grouper(level=1)])

@pd.api.extensions.register_dataframe_accessor("preprocessor")
@pd.api.extensions.register_series_accessor("preprocessor")
class PreProcessor(Worker):
    
    def standarize(self, method: str = 'zscore', grouper = None):
        def _zscore(data):
            data = data.copy()
            mean = data.mean()
            std = data.std()
            zscore = (data - mean) / std
            return zscore

        def _minmax(data):
            data = data.copy()
            min_ = data.min()
            max_ = data.max()
            minmax = (data - min_) / (max_ - min_)
            return minmax
        
        groupers = [pd.Grouper(level=0)]

        if self.type_ == Worker.PN and grouper is not None:
            groupers += [grouper]

        if method == 'zscore':
            return self.data.groupby(groupers).apply(_zscore)

        elif method == 'minmax':
            return self.data.groupby(groupers).apply(_minmax)

    def deextreme(self, method = 'md_correct', grouper = None, n = None):

        def _md_correct(data):
            data = data.copy()
            median = data.median()
            mad = (data - median).abs().median()
            mad = mad.values.reshape((1, -1)).repeat(len(data), axis=0).reshape(data.shape)
            mad = pd.DataFrame(mad, index=data.index, columns=data.columns)
            madup = mad + n * mad
            maddown = mad - n * mad
            data[data > madup] = madup
            data[data < maddown] = maddown
            return data
            
        def _std_correct(data):
            data = data.copy()
            mean = data.mean()
            mean = mean.values.reshape((1, -1)).repeat(len(data), axis=0).reshape(data.shape)
            mean = pd.DataFrame(mean, index=data.index, columns=data.columns)
            std = data.std()
            up = mean + n * std
            down = mean - n * std
            data[data > up] = up
            data[data < down] = down
            return data
        
        def _drop_odd(data):
            data = data.copy()
            if not isinstance(n, (list, tuple)):
                min_, max_ = n / 2, 1 - n /2
            else:
                min_, max_ = n[0], n[1]
            down = data.quantile(min_)
            up = data.quantile(max_)
            down = down.values.reshape((1, -1)).repeat(len(data), axis=0).reshape(data.shape)
            up = up.values.reshape((1, -1)).repeat(len(data), axis=0).reshape(data.shape)
            data[data > up] = up
            data[data < down] = down
            return data

        groupers = [pd.Grouper(level=0)]
        
        if self.type_ == Worker.PN and grouper is not None:
            groupers += [grouper]

        if method == 'md_correct':
            if n is None:
                n = 5
            return self.data.groupby(groupers).apply(_md_correct)

        elif method == 'std_correct':
            if n is None:
                n = 3
            return self.data.groupby(groupers).apply(_std_correct)
        
        elif method == 'drop_odd':
            if n is None:
                n = 0.1
            return self.data.groupby(groupers).apply(_drop_odd)
    
    def fill_miss(self, method = 'pad_zero', grouper = None):

        def _pad_with_zero(data):
            data = data.copy()
            data = data.fillna(0)
            return data
            
        def _pad_with_mean(data):
            data = data.copy()
            mean = data.mean(axis=0)
            mean = mean.values.reshape((1, -1)).repeat(len(data), axis=0).reshape(data.shape)
            mean = pd.DataFrame(mean, columns=data.columns, index=data.index)
            data = data.fillna(mean)
            return data

        def _pad_with_median(data):
            data = data.copy()
            median = data.median(axis=0)
            median = median.values.reshape((1, -1)).repeat(len(data), axis=0).reshape(data.shape)
            median = pd.DataFrame(median, columns=data.columns, index=data.index)
            data = data.fillna(median)
            return data

        groupers = [pd.Grouper(level=0)]
        
        if self.type_ == Worker.PN and grouper is not None:
            groupers += [grouper]

        if method == 'pad_zero':
            return self.data.groupby(groupers).apply(_pad_with_zero)

        elif method == 'pad_mean':
            return self.data.groupby(groupers).apply(_pad_with_mean)
        
        elif method == 'pad_median':
            return self.data.groupby(groupers).apply(_pad_with_median)

if __name__ == "__main__":
    import numpy as np
    price = pd.DataFrame(np.random.rand(100, 4), columns=['open', 'high', 'low', 'close'],
        index=pd.MultiIndex.from_product([pd.date_range('20100101', periods=20), list('abced')]))
    price.iloc[0, 2] = np.nan
    print(price.preprocessor.fill_miss('pad_zero'))
    
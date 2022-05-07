# 金融工程、量化投资通用框架

## 简介

使用pandas进行量化研究，需要首先明确量化研究的指标，根据指标的计算方式得到需要的原始数据。原始数据可能需要进行进一步的处理，例如从价格数据转换为过去的收益率，或是未来收益率。最后通过处理好的数据其进行计算，得到目标的指标。最后分析目标指标与对应标的收益率或是其它相关指标的关系，得到初步的结论。

又或是对量化策略进行分析，其本质是根据一系列的指标数据的大小或方向得出一系列的买卖信号，利用买卖信号和收盘价开盘价等信息进行回测，最终得到相关的组合收益时间序列，回撤，夏普比率等衡量组合效果的指标。

综上所述，这些研究都是基于面板数据的，即在任何时间点（datetime）上都有一个全市场（asset）的各项指标（indicator）信息，从这三个维度我们可以构建一个面板数据分析的插件，统一整合到pandas中，让pandas直接调用我们常用的量化分析方法的接口，让量化分析更加简单方便，高效快捷。

![process](./process.svg)

## analyst

analyst是pandasquant中负责对整理好的数据进行分析的“成员”，在经过数据整理、数据清洗、去极值标准化等一系列操作后，就需要对数据进行进一步的分析，例如计算因子的相关性、计算因子的IC或是使用Barra模型对因子进行回归分析等等，analyst为这一系列操作提供了便捷的接口。

### regressor

analyst中的回归分析器，可以对传入的数据进行回归分析，目前(v0.1)版本仅支持ols的回归分析，后续版本将支持其它的回归分析方法。

> ols(y: pd.Series = None, x_col: str = None, y_col: str = None):
> OLS Regression Function
> other: Series, assigned y value in a series form
> x_col: list, a list of column names used for regression x values
> y_col: str, the column name used for regression y values

```python
# data is a dataframe
data.regressor.ols(x_col=['x0', 'x1'], y_col='y')
# data is a series
data.regressor.ols(y=pd.Series([1, 2, 3, 4, 5, 6]))
```

### describer

analyst中对数据进行描述性统计分析、计算IC的分析器，后续版本可能加入其他进行描述统计分析的方法。

> corr(self, other: pd.Series = None, method: str = 'spearman', tvalue = False):
> Calculation for correlation matrix
method: str, the method for calculating correlation > function
> tvalue: bool, whether to return t-value of a time-seriesed correlation coefficient

```python
# data is a dataframe
data.describer.corr()
# data is a series
data.describer.corr(corr_target: pd.Series)
```

> ic(forward: pd.Series = None, grouper = None, method: str = 'spearman'):
> To calculate ic value
> other: series, the forward column
> method: str, 'spearman' means rank ic

```python
# data is a dataframe
data.describer.ic(grouper=indstry_group) # if you want to analyse by industry
# data is a series
data.describer.ic(forward: pd.Series)
```

## calculator

pandasquant中主要负责窗口计算的“成员”，在我们计算指标的过程中，使用到的往往不是整个面板数据，为了避免引入未来信息，有时往往还需要限制时间窗口。所以在给定的时间窗口上对面板数据进行滚动的计算便显得尤为重要。但是由于pandas中提供的rolling方法返回的并不是一个可供滚动的DataFrame类，只有在使用iter方法转为迭代器后才可对滚动窗口的DataFrame进行计算，同时由于面板数据有两个维度的索引，如果每个日期对应的资产数量不一致，还会出现滚动窗口大小不断变化的问题，因此calculator中的rolling方法为这一类计算需求提供了便利。

在计算一个指标时，我们需要明确的只是在一个计算单元的时间窗口内的计算方法即可。例如过去20日的动量因子计算，是对过去20日内股票池中股票收益率进行计算，只需要选取过去20个交易日的面板数据，计算最后一日的收盘价相对于第一日收盘价的涨跌幅，对每一个滚动窗口都是如此。所以我们定义的计算单元函数就应该是用于计算如此一个时间窗口内的动量计算方法。

需要特别注意的是，计算结果返回应该是DataFrame、Series、数值、列表等等，pandasquant会自动的将计算结果是为一个整体，并且为结果标记上当前时间窗口的最晚时间作为计算结果的索引。

> rolling(window: int, func, *args, grouper = None, offset: int = 0, **kwargs):
> Provide rolling window func apply for pandas dataframe
> window: int, the rolling window length
> func: unit calculation function
> args: arguments apply to func
> grouper: the grouper applied in func
> offset: int, the offset of the index, default 0 is the latest time
> kwargs: the keyword argument applied in func

```python
def calc_unit(data):
    start = data.index.levels[0][0]
    end = data.index.levels[0][1]
    result = data.loc[end] / data[start] - 1
    result.index = pd.MultiIndex.from_product([[end], result.index])
    return result

result = indicators.calculator.rolling(window=20, func=calc_unit)
```

## TODO

- 对输入数据进行约束，变成双行索引与双列索引的标准化数据，同时添加构造函数，更加方便、精确的控制输入的数据
- 加入机器学习算法至analyze中，可以使用机器学习的回归与分类方法进行计算

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


class __TimeFormatter(logging.Formatter):

    def __init__(
        self, 
        display_time: bool = True,
        display_name: str = True,
        fmt: str | None = None, 
        datefmt: str | None = None, 
        style: str = "%", 
        validate: bool = True, *, 
        defaults = None
    ) -> None:
        super().__init__(fmt, datefmt, style, validate, defaults=defaults)
        self.display_time = display_time
        self.display_name = display_name


class _StreamFormatter(__TimeFormatter):
    COLORS = {
        'DEBUG': '\033[94m',
        'INFO': '\033[92m',
        'WARNING': '\033[93m',
        'ERROR': '\033[95m',
        'CRITICAL': '\033[31m',
        'RESET': '\033[0m',
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.message = record.getMessage()
        record.asctime = self.formatTime(record, self.datefmt)
        formatted_record = f'{color}'
        if self.display_time:
            formatted_record += f'[{record.asctime}] '
        if self.display_name:
            formatted_record += f'<{record.name}> '
        formatted_record += f'{record.message}{self.COLORS["RESET"]}'
        return formatted_record


class _FileFormatter(__TimeFormatter):

    def format(self, record):
        record.message = record.getMessage()
        record.asctime = self.formatTime(record, self.datefmt)
        formatted_record = ''
        if self.display_time:
            formatted_record += f'[{record.asctime}] '
        if self.display_name:
            formatted_record += f'<{record.name}> '
        formatted_record += f'|{record.levelname}| {record.message}'
        return formatted_record


class Logger(logging.Logger):

    def __init__(
        self, 
        name: str = None, 
        level: int = logging.DEBUG, 
        stream: bool = True, 
        file: str = None,
        display_time: bool = True,
        display_name: bool = False,
    ):
        name = name or 'QuoolLogger'
        super().__init__(name, level)

        if stream:
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(_StreamFormatter(
                display_time=display_time, display_name=display_name
            ))
            self.addHandler(stream_handler)

        if file:
            file_handler = logging.FileHandler(file)
            file_handler.setFormatter(_FileFormatter(
                display_time=display_time, display_name=display_name
            ))
            self.addHandler(file_handler)


def parse_commastr(
    commastr: 'str | list',
) -> pd.Index:
    if isinstance(commastr, str):
        commastr = commastr.split(',')
        return list(map(lambda x: x.strip(), commastr))
    elif commastr is None:
        return None
    else:
        return commastr

def reduce_mem_usage(df: pd.DataFrame):
    logger = Logger("QuoolReduceMemUsage")
    start_mem = df.memory_usage().sum()
    logger.info('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum()
    logger.info('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    logger.info('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

def evaluate(
        value: pd.Series, 
        cash: pd.Series = None,
        turnover: pd.Series = None,
        benchmark: pd.Series = None,
        image: str = None,
        result: str = None,
    ):
        cash = cash.squeeze() if isinstance(cash, (pd.Series, pd.DataFrame)) else \
            pd.Series(np.zeros(value.shape[0]), index=value.index)
        turnover = turnover.squeeze() if isinstance(turnover, (pd.Series, pd.DataFrame)) else \
            pd.Series(np.zeros(value.shape[0]), index=value.index)
        benchmark = benchmark if isinstance(benchmark, (pd.Series, pd.DataFrame)) else \
            pd.Series(np.zeros(value.shape[0]), index=value.index)
        benchmark = benchmark.loc[value.index]
        net_value = value / value.iloc[0]
        net_cash = cash / cash.iloc[0]
        returns = value.pct_change(fill_method=None).fillna(0)
        benchmark_returns = benchmark.pct_change(fill_method=None).fillna(0)
        benchmark_returns = benchmark_returns if not benchmark_returns.isna().all() else pd.Series(np.zeros(benchmark_returns.shape[0]), index=benchmark.index)
        drawdown = net_value / net_value.cummax() - 1

        # evaluation indicators
        evaluation = pd.Series(name='evaluation')
        evaluation['total_return(%)'] = (net_value.iloc[-1] / net_value.iloc[0] - 1) * 100
        evaluation['annual_return(%)'] = ((evaluation['total_return(%)'] / 100 + 1) ** (
            365 / (value.index.max() - value.index.min()).days) - 1) * 100
        evaluation['annual_volatility(%)'] = (returns.std() * np.sqrt(252)) * 100
        down_volatility = (returns[returns < 0].std() * np.sqrt(252)) * 100
        enddate = drawdown.idxmin()
        startdate = drawdown.loc[:enddate][drawdown.loc[:enddate] == 0].index[-1]
        evaluation['max_drawdown(%)'] = (-drawdown.min()) * 100
        evaluation['max_drawdown_period(days)'] = enddate - startdate
        evaluation['max_drawdown_start'] = startdate
        evaluation['max_drawdown_stop'] = enddate
        evaluation['daily_turnover(%)'] = turnover.mean() * 100
        evaluation['sharpe_ratio'] = evaluation['annual_return(%)'] / evaluation['annual_volatility(%)'] \
            if evaluation['annual_volatility(%)'] != 0 else np.nan
        evaluation['sortino_ratio'] = evaluation['annual_return(%)'] / down_volatility \
            if down_volatility != 0 else np.nan
        evaluation['calmar_ratio'] = evaluation['annual_return(%)'] / evaluation['max_drawdown(%)'] \
            if evaluation['max_drawdown(%)'] != 0 else np.nan

        if not (benchmark==0).all():
            exreturns = returns - benchmark_returns.loc[returns.index]
            benchmark_volatility = (benchmark_returns.std() * np.sqrt(252)) * 100
            exvalue = (1 + exreturns).cumprod()
            cum_benchmark_return = (1 + benchmark_returns).cumprod()
            exdrawdown = exvalue / exvalue.cummax() - 1
            evaluation['total_exreturn(%)'] = (exvalue.iloc[-1] - exvalue.iloc[0]) * 100
            evaluation['annual_exreturn(%)'] = ((evaluation['total_exreturn(%)'] / 100 + 1
                ) ** (365 / (exvalue.index.max() - exvalue.index.min()).days) - 1) * 100
            evaluation['annual_exvolatility(%)'] = (exreturns.std() * np.sqrt(252)) * 100
            enddate = exdrawdown.idxmin()
            startdate = exdrawdown.loc[:enddate][exdrawdown.loc[:enddate] == 0].index[-1]
            evaluation['ext_max_drawdown(%)'] = (exdrawdown.min()) * 100
            evaluation['ext_max_drawdown_period(days)'] = enddate - startdate
            evaluation['ext_max_drawdown_start'] = startdate
            evaluation['ext_max_drawdown_stop'] = enddate
            evaluation['beta'] = returns.cov(benchmark_returns) / benchmark_returns.var()
            evaluation['alpha(%)'] = (returns.mean() - (evaluation['beta'] * (benchmark_returns.mean()))) * 100
            evaluation['treynor_ratio(%)'] = (evaluation['annual_exreturn(%)'] / evaluation['beta'])
            evaluation['information_ratio'] = evaluation['annual_exreturn(%)'] / benchmark_volatility \
                if benchmark_volatility != 0 else np.nan
        else:
            exvalue = net_value
            exdrawdown = drawdown
            cum_benchmark_return = pd.Series(np.ones(returns.shape[0]), index=returns.index)
            
        data = pd.concat([value, net_value, exvalue, net_cash, returns, cum_benchmark_return, drawdown, exdrawdown, turnover], 
                axis=1, keys=['value', 'net_value', 'exvalue', 'net_cash', 'returns', 'benchmark', 'drawdown', 'exdrawdown', 'turnover'])
        
        if result is not None:
            data.to_excel(result, sheet_name="performances")
        
        if image is not None:
            fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
            plt.subplots_adjust(wspace=0.3, hspace=0.5)

            ax00 = data["net_value"].plot(ax=ax[0,0], title="Fund Return", color=['#1C1C1C'], legend=True)
            ax00.legend(loc='lower left')
            ax00.set_ylabel("Cumulative Return")
            ax00_twi = ax[0,0].twinx()
            ax00_twi.fill_between(data.index, 0, data['drawdown'], color='#009100', alpha=0.3)
            ax00_twi.set_ylabel("Drawdown")

            if not (benchmark==0).all():
                year = (data[['net_value', 'exvalue', 'benchmark']].resample('YE').last() - data[['net_value', 'exvalue', 'benchmark']].resample('YE').first())
            else:
                year = (data['net_value'].resample('YE').last() - data['net_value'].resample('YE').first())
            month = (data['net_value'].resample('ME').last() - data['net_value'].resample('ME').first())
            year.index = year.index.year
            year.plot(ax=ax[0,1], kind='bar', title="Yearly Return", rot=45, colormap='Paired')
            ax[0, 2].bar(month.index, month.values, width=20)
            ax[0, 2].set_title("Monthly Return")

            ax10 = data['exvalue'].plot(ax=ax[1,0], title='Extra Return', legend=True)
            ax10.legend(loc='lower left')
            ax10.set_ylabel("Cumulative Return")
            ax10_twi = ax[1,0].twinx()
            ax10_twi.fill_between(data.index, 0, data['exdrawdown'], color='#009100', alpha=0.3)
            ax10_twi.set_ylabel("Drawdown")

            data[['net_value', 'benchmark']].plot(ax=ax[1,1], title="Fund Return")

            ax12 = data['net_cash'].plot(ax=ax[1,2], title="Turnover")
            ax12.set_ylabel('net_cash')
            ax12_twi = ax[1,2].twinx()
            ax12_twi.set_ylabel('turnover')
            ax12_twi.plot(data.index, data['turnover'], color='red')

            fig.tight_layout()
            if isinstance(image, (str, Path)):
                fig.savefig(image)
            else:
                fig.show()

        return evaluation

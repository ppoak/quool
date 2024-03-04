import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from .table import PanelTable
from .util import parse_commastr
from joblib import Parallel, delayed


class Factor(PanelTable):

    def get_trading_days(
        self,
        start: str | pd.Timestamp = None,
        stop: str | pd.Timestamp = None,
    ):
        frag = self._read_fragment(self.fragments[0])
        field = frag.columns[0]
        start = start or frag.index.get_level_values(self._date_level).min()
        code = frag.index.get_level_values(self._code_level).min()
        dates = super().read(field, code=code, start=start, stop=stop
            ).droplevel(self._code_level).index
        return dates
    
    def get_trading_days_rollback(
        self, 
        date: str | pd.Timestamp, 
        rollback: int = 1
    ):
        if rollback > 0:
            trading_days = self.get_trading_days(start=None, stop=date)
            rollback = trading_days[trading_days <= date][-rollback - 1]
        else:
            trading_days = self.get_trading_days(start=date, stop=None)
            rollback = trading_days[min(len(trading_days), -rollback)]

    def read(
        self, 
        field: str | list = None, 
        code: str | list = None, 
        start: str | list = None, 
        stop: str = None, 
        filters: list[list[tuple]] = None
    ) -> pd.Series | pd.DataFrame:
        df = super().read(field, code, start, stop, filters)
        if df.columns.size == 1:
            df = df.unstack(level=self._code_level)
        return df

    def save(
        self,
        df: pd.DataFrame | pd.Series, 
        name: str = None, 
    ):
        if isinstance(df, pd.DataFrame) and df.index.nlevels == 1:
            code_level = self.get_levelname(self._code_level)
            date_level = self.get_levelname(self._date_level)
            code_level = 'code' if isinstance(code_level, int) else code_level
            date_level = 'date' if isinstance(date_level, int) else date_level
            df = df.stack(dropna=True).swaplevel()
            df.index.names = [code_level, date_level]
        
        if isinstance(df, pd.Series):
            if name is None:
                raise ValueError('name cannot be None')
            df = df.to_frame(name)
        
        update_data = df[df.columns[df.columns.isin(self.columns)]]
        add_data = df[df.columns[~df.columns.isin(self.columns)]]
        if not update_data.empty:
            self.update(df)
        if not add_data.empty:
            self.add(df)
    
    def perform_crosssection(
        self, name: str,
        future: pd.DataFrame,
        image: str | bool = True,
        result: str = None,
    ):
        factor = self.read(field=name, start=future.index)
        
        if image is not None:
            fig, axes = plt.subplots(factor.index.size, 1, 
                figsize=(20, 10 * factor.index.size))
            for i, date in enumerate(factor.index):
                data = pd.concat([factor.loc[date], future.loc[date]], 
                    axis=1, keys=[name, date])
                data[name].plot.kde(ax=axes[i], title=date)
                data.plot.scatter(ax=axes[i], secondary_y=True, x=name, y=date)
            
            fig.tight_layout()
            if isinstance(image, (str, Path)):
                fig.savefig(image)
            else:
                fig.show()
                
        if result is not None:
            pd.concat([factor, future], axis=1, 
                keys=[name, 'future']).to_excel(result)

    def perform_inforcoef(
        self, name: str,
        future: pd.DataFrame,
        method: str = 'pearson',
        image: str | bool = True,
        result: str = None,
    ):
        factor = self.read(field=name)
        inforcoef = factor.corrwith(future, axis=1, method=method).dropna()
        inforcoef.name = f"infocoef"

        if image is not None:
            fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            inforcoef.plot(ax=ax)
            inforcoef.rolling(5).mean().plot(linestyle='--', ax=ax)
            inforcoef.cumsum().plot(linestyle='-.', secondary_y=True)
            fig.tight_layout()
            if not isinstance(image, bool):
                fig.savefig(image)
            else:
                fig.show()
        
        if result is not None:
            inforcoef.to_excel(result)
        return inforcoef
    
    def perform_grouping(
        self, name: str,
        future: pd.DataFrame,
        topk: int = 100,
        ngroup: int = 5,
        commission: float = 0.002,
        image: str | bool = True,
        result: str = None,
    ):
        factor = self.read(field=name, start=future.index)
        # ngroup test
        try:
            groups = factor.apply(lambda x: pd.qcut(x, q=ngroup, labels=False), axis=1) + 1
        except:
            for date in factor.index:
                try:
                    pd.qcut(factor.loc[date], q=ngroup, labels=False)
                except:
                    raise ValueError(f"on date {date}, grouping failed")
        
        def _grouping(x):
            group = groups.where(groups == x)
            weight = (group / group).fillna(0)
            weight = weight.div(weight.sum(axis=1), axis=0)
            delta = weight.diff().fillna(0)
            turnover = delta.abs().sum(axis=1) / 2
            ret = (future * weight).sum(axis=1).shift(1).fillna(0)
            ret -= commission * turnover
            return ret
            
        ngroup_result = Parallel(n_jobs=-1, backend='loky')(
            delayed(weight_strategy)(
                (groups.where(groups == i) / groups.where(groups == i)).div(
                    groups.where(groups == i).count(axis=1), axis=0).fillna(0), 
                price, delay, 'both', commission, benchmark, False, None
        ) for i in range(1, ngroup + 1))
        ngroup_evaluation = pd.concat([res['evaluation'] for res in ngroup_result], 
            axis=1, keys=range(1, ngroup + 1)).add_prefix('group')
        ngroup_value = pd.concat([res['value'] for res in ngroup_result], 
            axis=1, keys=range(1, ngroup + 1)).add_prefix('group')
        ngroup_turnover = pd.concat([res['turnover'] for res in ngroup_result], 
            axis=1, keys=range(1, ngroup + 1)).add_prefix('group')

        # topk test
        topks = factor.rank(ascending=False, axis=1) < topk
        topks = factor.where(topks)
        topks = (topks / topks).div(topks.count(axis=1), axis=0).fillna(0)
        topk_result = quool.weight_strategy(topks, price, delay, 'both', 
            commission, benchmark, None, None)
        topk_evaluation = topk_result['evaluation']
        topk_value = topk_result['value']
        topk_turnover = topk_result['turnover']

        # compute returns
        ngroup_returns = ngroup_value.pct_change().fillna(0)
        topk_returns = topk_value.pct_change().fillna(0)
        
        # longshort test
        longshort_returns = ngroup_returns[f"group{ngroup}"] - ngroup_returns["group1"]
        longshort_value = (longshort_returns + 1).cumprod()
        longshort_value.name = "long-short"
        
        # naming
        ngroup_evaluation.name = "ngroup evaluation"
        ngroup_value.name = "ngroup value"
        topk_value.name = "topk value"
        longshort_value.name = "longshort value"
        ngroup_turnover.name = "ngroup turnover"
        topk_turnover.name = "topk turnover"
        if benchmark is not None:
            ngroup_exvalue.name = "ngroup exvalue"
            topk_exvalue.name = "topk exvalue"

        if image is not None:
            fignum = 5 + 2 * (benchmark is not None)
            fig, axes = plt.subplots(nrows=fignum, ncols=1, figsize=(20, 10 * fignum))
            ngroup_value.plot(ax=axes[0], title=ngroup_value.name)
            ngroup_turnover.plot(ax=axes[1], title=ngroup_turnover.name)
            topk_value.plot(ax=axes[2], title=topk_value.name)
            topk_turnover.plot(ax=axes[3], title=topk_turnover.name)
            longshort_value.plot(ax=axes[4], title=longshort_value.name)
            if benchmark is not None:
                ngroup_exvalue.plot(ax=axes[5], title=ngroup_exvalue.name)
                topk_exvalue.plot(ax=axes[6], title=topk_exvalue.name)
            fig.tight_layout()
            if not isinstance(image, bool):
                fig.savefig(image)
            else:
                fig.show()

        if result is not None:
            with pd.ExcelWriter(result) as writer:
                ngroup_evaluation.to_excel(writer, sheet_name=ngroup_evaluation.name)
                topk_evaluation.to_excel(writer, sheet_name=topk_evaluation.name)
                ngroup_value.to_excel(writer, sheet_name=ngroup_returns.name)
                ngroup_turnover.to_excel(writer, sheet_name=ngroup_turnover.name)
                topk_value.to_excel(writer, sheet_name=topk_returns.name)
                topk_turnover.to_excel(writer, sheet_name=topk_turnover.name)
                longshort_value.to_excel(writer, sheet_name=longshort_returns.name)
                
                if benchmark is not None:
                    ngroup_exvalue.to_excel(writer, sheet_name=ngroup_exreturns.name)
                    topk_exvalue.to_excel(writer, sheet_name=topk_exreturns.name)

        return {
            'ngroup_evaluation': ngroup_evaluation, 
            'ngroup_value': ngroup_value, 
            'ngroup_turnover': ngroup_turnover,
            'topk_evaluation': topk_evaluation, 
            'topk_value': topk_value, 
            'topk_turnover': topk_turnover,
            'longshort_value': longshort_value,
        }

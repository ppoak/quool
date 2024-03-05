import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from .table import PanelTable
from .trade import TradeRecorder
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
            df = df.unstack(level=self._code_level).squeeze()
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
        future: pd.Series,
        image: str | bool = True,
        result: str = None,
    ):
        factor = self.read(field=name, start=future.name, stop=future.name)
        data = pd.concat([factor.squeeze(), future], axis=1, keys=[name, future.name])
        if image is not None:
            pd.plotting.scatter_matrix(data, figsize=(20, 20), hist_kwds={'bins': 100})
            
            plt.tight_layout()
            if isinstance(image, (str, Path)):
                plt.savefig(image)
            else:
                plt.show()
                
        if result is not None:
            data.to_excel(result)

    def perform_inforcoef(
        self, name: str,
        future: pd.DataFrame,
        rolling: int = 20,
        method: str = 'pearson',
        image: str | bool = True,
        result: str = None,
    ):
        factor = self.read(field=name)
        inforcoef = factor.corrwith(future, axis=1, method=method).dropna()
        inforcoef.name = f"infocoef"

        if image is not None:
            fig, ax = plt.subplots(1, 1, figsize=(20, 10))
            inforcoef.plot(ax=ax, label='infor-coef', alpha=0.7, title=f'{name} Information Coef')
            inforcoef.rolling(rolling).mean().plot(linestyle='--', ax=ax, label='trend')
            inforcoef.cumsum().plot(linestyle='-.', secondary_y=True, ax=ax, label='cumm-infor-coef')
            pd.Series(np.zeros(inforcoef.shape[0]), index=inforcoef.index).plot(color='grey', ax=ax, alpha=0.5)
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
            val = (ret + 1).cumprod()
            return {
                'evaluation': TradeRecorder.evaluate(val, turnover=turnover, image=False),
                'value': val, 'turnover': turnover,
            }
            
        ngroup_result = Parallel(n_jobs=-1, backend='loky')(
            delayed(_grouping)(i) for i in range(1, ngroup + 1))
        ngroup_evaluation = pd.concat([res['evaluation'] for res in ngroup_result], 
            axis=1, keys=range(1, ngroup + 1)).add_prefix('group')
        ngroup_value = pd.concat([res['value'] for res in ngroup_result], 
            axis=1, keys=range(1, ngroup + 1)).add_prefix('group')
        ngroup_turnover = pd.concat([res['turnover'] for res in ngroup_result], 
            axis=1, keys=range(1, ngroup + 1)).add_prefix('group')
        ngroup_returns = ngroup_value.pct_change().fillna(0)
        longshort_returns = ngroup_returns[f"group{ngroup}"] - ngroup_returns["group1"]
        longshort_value = (longshort_returns + 1).cumprod()
        longshort_evaluation = TradeRecorder.evaluate(longshort_value, image=False)
        
        # naming
        longshort_evaluation.name = "longshort"
        longshort_value.name = "longshort value"

        if image is not None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
            longshort_value.plot(ax=ax, linestyle='--')
            ngroup_value.plot(ax=ax, alpha=0.8)
            ngroup_turnover.plot(ax=ax, secondary_y=True, alpha=0.2)
            fig.tight_layout()
            if isinstance(image, (str, Path)):
                fig.savefig(image)
            else:
                fig.show()            
        
        if result is not None:
            with pd.ExcelWriter(result) as writer:
                ngroup_evaluation.to_excel(writer, sheet_name="ngroup_evaluation")
                longshort_evaluation.to_excel(writer, sheet_name="longshort_evaluation")
                ngroup_value.to_excel(writer, sheet_name="ngroup_value")
                ngroup_turnover.to_excel(writer, sheet_name="ngroup_turnover")
                longshort_value.to_excel(writer, sheet_name="longshort_value")
        
        return pd.concat([ngroup_evaluation, longshort_evaluation], axis=1)
                

    def perform_topk(
        self, name: str,
        future: pd.DataFrame,
        topk: int = 100,
        commission: float = 0.002,
        image: str | bool = True,
        result: str = None,
    ):
        factor = self.read(field=name, start=future.index)
        topks = factor.rank(ascending=False, axis=1) < topk
        topks = factor.where(topks)
        topks = (topks / topks).div(topks.count(axis=1), axis=0).fillna(0)
        turnover = topks.diff().fillna(0).abs().sum(axis=1) / 2
        ret = (topks * future).sum(axis=1).shift(1) - turnover * commission
        val = (1 + ret).cumprod()
        eva = TradeRecorder.evaluate(val, turnover=turnover, image=False)

        val.name = "value"
        turnover.name = "turnover"
        eva.name = "evaluation"

        if image is not None:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 10))
            val.plot(ax=ax, title="Top K")
            turnover.plot(ax=ax, secondary_y=True, alpha=0.5)
            fig.tight_layout()
            if not isinstance(image, bool):
                fig.savefig(image)
            else:
                fig.show()

        if result is not None:
            pd.concat([eva, val, turnover], axis=1).to_excel(result)

        return eva

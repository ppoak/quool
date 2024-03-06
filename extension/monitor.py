import squarify
import config as c
import pandas as pd
import database as d
import matplotlib.pyplot as plt
from IPython.core.magic import magics_class, line_magic, Magics


@magics_class
class MonitorMagics(Magics):

    @line_magic
    def treemap(self, line):
        opts, _ = self.parse_options(line, "", "name=", "n=")
        n = int(opts.get("n", 1))
        ret = d.get_spot_return(day=n)
        bins = [-float('inf'), -20, -15, -10, -8, -7, -6, -5, -4,
            -3, -2, -1, 1, 2, 3, 4, 5, 6, 7,
            8, 10, 15, 20, float('inf')]
        labels = ["#00FF00", "#00EE00", "#00D000", "#00C000", 
            "#00B000", "#00A000", "#009C00", "#007C00", 
            "#005C00", "#003C00", "#002C00", "#2C0000", 
            "#3C0000", "#5C0000", "#7C0000", "#9C0000", 
            "#A00000", "#B00000", "#C00000", "#D00000", 
            "#E00000", "#EE0000", "#FF0000"]
        ret["colors"] = pd.cut(ret["change_rate"], bins=bins, labels=labels)
        name = opts.get("name", "circulating_market_cap")
        if name != "circulating_market_cap":
            date = c.fqtd.get_trading_days_rollback(rollback=n)
            table, name = name.split(".")
            ret[name] = getattr(c, table).read(name, start=date, stop=date)
        ret = ret.dropna(subset=[name, "change_rate"], axis=0, how='any')
        ret = ret.sort_values(by=name, ascending=False)
        fig, ax = plt.subplots(1, 1, figsize=(20, 20))
        squarify.plot(sizes=ret[name], color=ret["colors"], ax=ax)
        ax.xaxis.set_visible(False); ax.yaxis.set_visible(False)
        fig.tight_layout()
        return fig


def load_ipython_extension(ipython):
    ipython.register_magics(MonitorMagics)

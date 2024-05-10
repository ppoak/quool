import quool as q
import pandas as pd

qtm = q.PanelTable("data/quotes-min")
qtd = q.PanelTable("data/quotes-day")

trading_mins = qtm.read("close", code="600570.XSHG", start="20210101").index.get_level_values(qtm._date_level)
trading_days = qtd.read("close", code="600570.XSHG", start="20210101").index.get_level_values(qtd._date_level)

grid = q.TradeRecorder("grid", principle=150000, start=trading_days[0])
logger = q.Logger("grid", display_time=False)  # 初始化日志记录器

for date in trading_days:  # 遍历每个交易日
    back_days = qtd.read("close", code="600570.XSHG", stop=date).index.get_level_values(qtd._date_level)
    rollback = back_days[back_days <= date][-10 - 1]
    df = qtd.read("close", code="600570.XSHG" ,start=rollback, stop=date - pd.Timedelta(days=1)).droplevel(1).iloc[:, 0]
    basePrice = df.ewm(10).mean().mean()

    # 设置网格参数
    volume = 200
    step = max(round(0.02 * basePrice, 2), 0.01)
    security = ['600570.XSHG']

    for min in trading_mins:  # 遍历每个分钟
        prices = qtm.read("close", code="600570.XSHG", start=min, stop=min).droplevel(1).iloc[:, 0]
        positions = grid.peek(date, price=prices)   # 获取当前持仓情况
        cash = positions.loc['cash', 'size']        # 获取现金数量
        value = positions["value"].sum()            # 计算投资组合总市值

        price_buy = basePrice - step
        price_sell = basePrice + step
        for sell in security:  # 遍历卖出股票列表
            if sell in positions.index and positions.loc[sell, 'size'] > 0:
                price = prices.loc[sell]
                if price >= price_sell:
                    comm = max(volume * price * 0.003, 5)  # 计算佣金，最低5元
                    grid.trade(date, sell, -volume, price, commission=comm)  # 执行卖出
                    basePrice = price_sell
                    cash += volume * price - comm  # 更新现金余额
                    # 记录卖出操作日志
                    logger.debug(f"[{date}] selling {sell} at {price_sell:.2f}, comm {comm:.2f}, cash updated {cash:.2f}")

        for buy in security:  # 遍历买入股票列表
            price = prices.loc[buy]
            if price <= price_buy:
                if cash >= volume * price + comm:
                    comm = max(volume * price * 0.003, 5)  # 计算佣金，最低5元
                    grid.trade(date, sell, -volume, price, commission=comm)  # 执行卖出
                    basePrice = price_buy
                    cash -= volume * price - comm  # 更新现金余额
                    # 记录买入操作日志
                    logger.debug(f"[{date}] buying {buy} at {price:.2f}, comm {comm:.2f}, cash updated {cash:.2f}")
                
                else:  # 如果现金不足，记录警告日志
                    logger.warning(f"[{date}] not enough cash to buy {buy} at {price:.2f}")
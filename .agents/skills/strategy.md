---
name: strategy
description: 编写 Quool 量化策略的完整工作流，覆盖从策略结构、生命周期钩子、下单接口到回测与实盘运行的全部环节。
triggers:
  - 编写策略
  - 编写量化策略
  - Quool 策略
  - backtest strategy
  - strategy tutorial
---

# Quool 策略编写

Quool 是一个事件驱动的量化回测与交易框架，策略（Strategy）是其核心层，负责在每个时间节点协调数据源（Source）和经纪商（Broker）完成信号生成与订单提交。

## 策略结构

一个完整的策略代码结构如下：

```python
import pandas as pd
from quool import DataFrameSource, Broker, Strategy
from quool import FixedRateCommission, FixedRateSlippage

# 1. 准备市场数据（MultiIndex: time × code）
source = DataFrameSource(market_data)

# 2. 初始化 Broker（可配置佣金和滑点模型）
broker = Broker(
    commission=FixedRateCommission(),
    slippage=FixedRateSlippage(),
)
broker.transfer(pd.Timestamp("2024-01-01"), 1_000_000)  # 注入初始资金

# 3. 定义策略
class MyStrategy(Strategy):
    def init(self):          # 初始化 — 回测前调用一次
        pass

    def preupdate(self):     # 预更新 — 每次迭代先于 update 执行
        pass

    def update(self):         # 核心逻辑 — 在此下单
        pass

    def stop(self):          # 结束 — 回测完成后调用一次
        pass

# 4. 运行回测
strategy = MyStrategy(source, broker)
results = strategy.backtest()
```

## 生命周期钩子

策略的执行流程遵循固定的事件驱动循环，四个钩子方法按顺序调用：

| 钩子 | 调用时机 | 用途 |
|------|----------|------|
| `init()` | 回测开始前调用一次 | 初始化持仓、订阅行情、设置参数 |
| `preupdate()` | 每个时间节点，Broker 成交匹配**后**、下单**前**调用 | 订单状态处理、持仓审计、预警逻辑 |
| `update()` | 每个时间节点的核心下单环节 | 根据信号生成订单 |
| `stop()` | 回测结束时调用一次 | 生成报告、清理资源、保存状态 |

```python
class MyStrategy(Strategy):
    def init(self):
        # 设置滑点窗口
        self.window = 20
        # 订阅标的列表
        self.codes = ["000001", "000002"]

    def preupdate(self):
        # 取消超时未成交订单
        for order in broker.get_pendings().itertuples():
            if (self.current_time - order.time) > pd.Timedelta(days=5):
                self.cancel(order.id)

    def update(self):
        # 在此实现下单逻辑
        pass

    def stop(self):
        # 平掉所有持仓
        for code in self.codes:
            self.close(code)
```

## 下单接口

Strategy 提供五类下单辅助方法，均返回 `Order` 对象或 `None`。

### 基础买卖

```python
# 市价买入
order = self.buy(code, quantity)

# 限价买入
order = self.buy(code, quantity, exectype=LIMIT, limit=10.5)

# 市价卖出
order = self.sell(code, quantity)

# 止损卖出（价格跌至 trigger 时触发）
order = self.sell(code, quantity, exectype=STOP, trigger=9.5)
```

### 目标仓位

适合定期再平衡场景，自动计算需要买入/卖出的数量：

```python
# 目标持仓市值（绝对金额）
self.order_target_value("000001", 100_000)

# 目标持仓占组合百分比（10%）
self.order_target_percent("000001", 0.10)
```

### 平仓

```python
# 市价平掉全部持仓
self.close("000001")
```

### 订单类型一览

| 类型 | 说明 | 触发条件 |
|------|------|----------|
| `MARKET` | 市价立即执行 | 下一个 tick 成交 |
| `LIMIT` | 限价挂单 | 买入：最低价 ≤ limit；卖出：最高价 ≥ limit |
| `STOP` | 止损/止盈触发 | 买入：最高价 ≥ trigger；卖出：最低价 ≤ trigger |
| `STOPLIMIT` | STOP 触发后按 LIMIT 执行 | trigger 触发后等待 limit 条件 |
| `TARGET` | 目标价触发 | 买入：最低价 ≤ trigger；卖出：最高价 ≥ trigger |
| `TARGETLIMIT` | TARGET 触发后按 LIMIT 执行 | trigger 触发后等待 limit 条件 |

### 订单取消

```python
self.cancel(order_or_id)  # 传入 Order 对象或订单 ID 字符串
```

## 组合状态查询

```python
# 查询当前持仓
positions = self.get_positions()  # pd.Series，index 为 code

# 查询总权益（现金 + 持仓市值）
total_value = self.get_value()

# 获取持仓成本（需从 broker 获取）
positions_df = broker.get_positions()
```

## 运行模式

### 回测（backtest）

```python
results = strategy.backtest(
    benchmark=benchmark_series,  # 对比基准净值序列
    history=True,               # 存储完整历史（values/positions/trades）
)

# results 包含：
# results["values"]       # 净值序列
# results["positions"]     # 持仓历史
# results["trades"]        # 交易记录
# results["evaluation"]     # 绩效指标
# results["orders"]        # 订单历史
# results["delivery"]       # 成交明细
```

### 实盘调度（run / arun）

```python
# 前台运行（阻塞）
strategy.run(
    store="state.json",              # 持久化路径
    trigger="cron",                  # APScheduler 触发类型
    trigger_kwargs={"minute": "*/5"}, # 每 5 分钟执行一次
)

# 后台运行（非阻塞）
scheduler = strategy.arun(
    trigger="interval",
    trigger_kwargs={"seconds": 30},
)
```

### 状态持久化

```python
# 序列化到字典
state = strategy.dump(history=True)

# 从字典恢复
new_strategy = Strategy.load(
    data=state,
    source=source,
    commission=FixedRateCommission(),
    slippage=FixedRateSlippage(),
)

# 直接存储到文件
strategy.store("run_state.json")
strategy.restore("run_state.json")
```

## 日志与通知

```python
# 使用内置 logger（默认 DEBUG 级别，类名作为 name）
self.log("信号触发，下单中")          # DEBUG 级别
self.log("订单已提交", level="INFO") # INFO 级别

# 订单状态变更通知（可覆盖）
def notify(self, order: Order):
    if order.status == "FILLED":
        self.log(f"成交：{order.code} × {order.quantity} @ {order.price}")
```

## 完整示例：双均线策略

```python
import pandas as pd
from quool import DataFrameSource, Broker, Strategy
from quool import FixedRateCommission, FixedRateSlippage

# 市场数据（MultiIndex: time × code）
source = DataFrameSource(ohlcv_data)

broker = Broker(
    commission=FixedRateCommission(rate=0.0003, min_commission=5),
    slippage=FixedRateSlippage(slip_rate=0.001),
)
broker.transfer(pd.Timestamp("2024-01-01"), 1_000_000)

class MAStrategy(Strategy):
    def init(self):
        self.short_window = 5
        self.long_window = 20
        self.codes = ["000001"]

    def update(self):
        for code in self.codes:
            # 计算均线（需要自己实现或用 pd 计算）
            prices = source.data.loc[source.time, code]
            ma5 = prices["close"].rolling(self.short_window).mean()
            ma20 = prices["close"].rolling(self.long_window).mean()

            # 均线金叉买入，死叉卖出
            if ma5 > ma20 and self.get_positions().get(code, 0) == 0:
                self.order_target_percent(code, 0.5)
            elif ma5 < ma20 and self.get_positions().get(code, 0) > 0:
                self.close(code)

results = MAStrategy(source, broker).backtest(history=True)
print(results["evaluation"][["annual_return", "sharpe_ratio", "max_drawdown"]])
```

## A 股专用 Broker

A 股市场需要遵守 100 股整手规则，使用 `AShareBroker`：

```python
from quool import AShareBroker

broker = AShareBroker(
    commission=FixedRateCommission(rate=0.0003, min_commission=5),
    slippage=FixedRateSlippage(slip_rate=0.001),
)
```

## 绩效评估指标

回测完成后通过 `results["evaluation"]` 访问关键指标：

| 类别 | 指标 | 说明 |
|------|------|------|
| 收益 | `total_return`, `annual_return` | 总收益、年化收益 |
| 风险调整收益 | `sharpe_ratio`, `sortino_ratio`, `calmar_ratio` | 夏普、索提诺、卡玛比率 |
| 回撤 | `max_drawdown`, `max_drawdown_period` | 最大回撤及持续周期 |
| 风险 | `VaR_5%`, `CVaR_5%` | 风险价值、条件风险价值 |
| 相对收益 | `beta`, `alpha`, `information_ratio` | Beta、Alpha、信息比率 |
| 交易统计 | `trade_win_rate`, `trade_return`, `position_duration` | 胜率、单笔收益、持仓周期 |
| 分布特征 | `skewness`, `kurtosis`, `monthly_win_rate` | 收益分布偏度、峰度、月胜率 |

# Quool

> Quool 是一个以事件驱动回测为核心的量化工具库（Quantitative Toolkit），支持回测与实盘交易。

## 环境

```bash
uv sync
```

- 运行脚本或模块请使用：`.venv/scripts/python`
- Windows 平台路径请使用 `/` 正斜杠

## Rules

请在代码编写、仓库修改、运行测试时严格遵守如下规则。

- 代码编写、注释编写请完全使用英文，回答用户用中文。
- 所有的新增、修改、删除编码完成后，都需要检查修改、新增或完善docs和对应代码的Google风格的Docstring，再通过所有测试后方可做出最后的解答
- 添加依赖：`uv add <package>`
- 运行脚本：`<project>/.venv/scripts/python <script>`
- 所有 quool 公共接口均通过 `from quool import ...` 导入

## 代码架构

```
quool/
├── order.py          # Order & Delivery 模型
├── broker.py         # 核心 Broker（模拟经纪商）
├── strategy.py        # 策略基类
├── source.py         # 数据源抽象基类
├── evaluator.py       # 绩效评估
├── friction.py        # 佣金 & 滑点模型
├── scheduler.py       # 多策略编排器
├── brokers/           # Broker 实现
│   ├── ashare.py     # A股 100股整手规则
│   ├── xueqiu.py     # 雪球模拟交易
│   └── xuntou.py     # XtQuant 实盘
└── sources/           # Source 实现
    ├── dataframe.py   # DataFrame 数据源
    ├── duck.py        # DuckDB/Parquet 数据源
    ├── realtime.py    # 东方财富实时行情
    └── xuntou.py      # XtQuant 历史数据预加载
```

## 文档索引

见 [docs/README.md](docs/README.md)。

## 核心概念

- **Source**：市场数据提供者，负责 `update()` 推进时间和返回 OHLCV 快照
- **Broker**：订单执行 & 投资组合会计，维护现金、持仓、挂单队列
- **Strategy**：策略基类，提供 `init/update/preupdate/stop` 生命周期钩子
- **Order/Delivery**：订单生命周期 + 成交记录
- **Evaluator**：绩效指标计算（Sharpe/Calmar/Sortino/VaR 等）
- **Friction**：佣金（FixedRateCommission）和滑点（FixedRateSlippage）模型

## 环境说明

- 虚拟环境入口：`.venv/scripts/python`
- Windows 路径：使用 `/` 正斜杠
- 配置文件：`pyproject.toml`

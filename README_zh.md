# Quool

Quool是一个专为量化投资研究而设计的Python框架，旨在提供一套灵活、高效的工具集，帮助研究人员和开发者快速实现数据处理、因子分析、交易记录以及策略评估等功能。通过Quool，用户可以专注于策略和因子的研究，而无需花费大量时间在数据管理和基础设施构建上。

## 特性

- **数据管理**：提供统一的接口用于管理和访问金融市场数据，支持日内数据和日度数据处理。
- **因子研究**：简化因子开发和测试流程，支持因子的定义、存储、分析以及性能评估。
- **交易记录**：提供灵活的记录器（Recorder）类，用于记录和管理交易数据和模型运行数据。
- **策略评估**：集成策略评估工具，支持多种性能指标计算和结果可视化。

## 安装

您可以通过以下方式从源代码安装：

```bash
git clone https://github.com/your-username/quool.git
cd quool
pip install .
```

## 快速入门

以下是使用Quool进行因子研究和策略评估的基本步骤：

### 定义因子

首先，继承`BaseFactor`类来定义您自己的因子。例如，定义一个计算成交量加权平均价格（VWAP）的因子：

```python
from quool import BaseFactor

class VWAPFactor(BaseFactor):
    def get_vwap(self, date: pd.Timestamp):
        # 实现VWAP的计算逻辑
        pass
```

### 计算因子值

实例化您的因子类，并调用`get`方法来计算特定日期范围内的因子值：

```python
vwap_factor = VWAPFactor(uri="./path/to/factor/data")
vwap_values = vwap_factor.get("vwap", start="2021-01-01", stop="2021-12-31")
```

### 策略评估

使用`TradeRecorder`或其他记录器类来记录您的交易活动，并使用`evaluate`方法进行策略评估：

```python
from quool import TradeRecorder

# 记录交易活动
trade_recorder = TradeRecorder(uri="./path/to/trade/data")
trade_recorder.trade(date="2021-01-01", ...)

# 评估策略性能
performance = trade_recorder.evaluate(...)
```

## 贡献

欢迎通过GitHub提交问题报告和拉取请求来贡献代码。

## 许可证

Quool遵循MIT许可证发布。


# Quool

Quool is a Python framework designed specifically for quantitative investment research. It aims to provide a flexible and efficient set of tools to help researchers and developers quickly implement data management, factor analysis, trading recording, and strategy evaluation functionalities. With Quool, users can focus on strategy and factor research without spending excessive time on data management and infrastructure setup.

## Features

- **Data Management**: Offers a unified interface for managing and accessing financial market data, supporting both intraday and daily data processing.
- **Factor Research**: Simplifies the process of factor development and testing, supporting factor definition, storage, analysis, and performance evaluation.
- **Trading Recording**: Provides flexible Recorder classes for recording and managing trade data and model execution data.
- **Strategy Evaluation**: Integrates strategy evaluation tools, supporting calculations of various performance metrics and result visualization.

## Installation

Currently, the Quool framework is not available on PyPI. You can install it from the source code as follows:

```bash
git clone https://github.com/your-username/quool.git
cd quool
pip install .
```

## Quick Start

Here are the basic steps to conduct factor research and strategy evaluation using Quool:

### Define a Factor

First, inherit the `BaseFactor` class to define your own factor. For example, define a factor that calculates the Volume Weighted Average Price (VWAP):

```python
from quool import BaseFactor

class VWAPFactor(BaseFactor):
    def get_vwap(self, date: pd.Timestamp):
        # Implement the calculation logic for VWAP
        pass
```

### Calculate Factor Values

Instantiate your factor class and use the `get` method to calculate factor values for a specific date range:

```python
vwap_factor = VWAPFactor(uri="./path/to/factor/data")
vwap_values = vwap_factor.get("vwap", start="2021-01-01", stop="2021-12-31")
```

### Evaluate Strategies

Use the `TradeRecorder` or other recorder classes to record your trading activities and use the `evaluate` method to assess strategy performance:

```python
from quool import TradeRecorder

# Record trading activities
trade_recorder = TradeRecorder(uri="./path/to/trade/data")
trade_recorder.record(date="2021-01-01", ...)

# Evaluate strategy performance
performance = trade_recorder.evaluate(...)
```

## Contributing

Contributions in the form of issue reports and pull requests are welcome on GitHub.

## License

Quool is released under the MIT license.

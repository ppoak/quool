# Quool

## 总体概况

### 核心组件

- `Table`: 基本表格类，可能用于存储和操作数据。
- `ItemTable`, `DatetimeTable`, `PanelTable`: 特定类型的表格类，用于处理不同类型的数据，如项目、日期时间和面板数据。

### 记录器

- `RunRecorder`, `TradeRecorder`, `ProxyRecorder`: 用于记录和跟踪策略运行、交易和代理的信息，可能用于回测或实时监控。

### 因子投资

- `Factor`: 用于定义和操作因子，这在因子投资策略中至关重要。

### 工具和实用程序

- `Logger`: 日志记录工具，用于跟踪程序运行时的信息和错误。
- `parse_commastr`, `reduce_mem_usage`: 功能性工具，分别用于解析逗号分隔的字符串和减少内存使用。

### 日志级别

- 定义了不同的日志级别（DEBUG, INFO, WARNING, CRITICAL），这有助于过滤和控制日志输出的详细程度。

### 版本信息

- 包的版本被定义为`5.0.7`。

总的来说，`quool`似乎提供了一套完整的工具和框架，用于量化投资领域的研究和实践。它涵盖了从数据处理和存储、策略记录和回测到因子投资等多个方面，旨在提升研究和实践的效率和便捷性。

`Table`类在`quool`框架中扮演着数据管理和操作的核心角色，专为处理、存储、和查询量化投资数据设计。这个类通过对Parquet文件格式的操作，提供了一种高效和灵活的方式来处理大规模的金融数据集。

## `Table`类

- **数据路径管理**：初始化时，`Table`类通过`uri`参数确定数据存储的位置，并支持路径的自动创建。
- **片段管理**：对存储在多个Parquet文件中的数据片段进行管理，提供了查找、读取、更新和删除片段的能力。
- **数据读取和写入**：支持从Parquet文件读取数据，以及将数据以Parquet格式写入存储路径。
- **数据结构管理**：能够查询和管理数据的列名、数据类型和维度形状。
- **高级数据操作**：提供了添加、更新、删除数据和列，以及重命名列的高级操作。

### `Table`类公开接口和用法

#### 构造函数

```python
Table(uri: str | Path, create: bool = False)
```
- `uri`：数据的存储路径。
- `create`：如果为`True`，当路径不存在时会创建该路径。

#### 属性
- `minfrag`：获取最小的数据片段。
- `fragments`：获取所有数据片段的列表。
- `columns`：获取表的所有列名。
- `dtypes`：获取表的数据类型。
- `dimshape`：获取表的维度形状。

#### 方法

- `read(columns: str | list[str] | None = None, filters: list[list[tuple]] = None) -> pd.DataFrame | pd.Series`：读取数据，可以指定列和过滤条件。
- `update(df: pd.DataFrame | pd.Series)`：更新表中的数据。
- `add(df: pd.Series | pd.DataFrame)`：向表中添加新的数据。
- `delete(index: pd.Index)`：删除指定索引的数据。
- `remove(fragment: str | list = None)`：删除指定的数据片段。
- `sub(column: str | list)`：删除指定的列。
- `rename(column: dict)`：重命名列。

### 用法示例

```python
from pathlib import Path
import pandas as pd

# 创建Table实例，指定数据存储路径
table = Table(uri=Path("/path/to/data"), create=True)

# 读取所有列的数据
data = table.read()

# 更新表中的数据
table.update(df=pd.DataFrame({"new_column": [1, 2, 3]}))

# 向表中添加新数据
table.add(df=pd.DataFrame({"another_column": [4, 5, 6]}))

# 删除指定索引的数据
table.delete(index=pd.Index([0, 1]))

# 删除指定的数据片段
table.remove(fragment="specific_fragment")

# 删除指定的列
table.sub(column=["unnecessary_column"])

# 重命名列
table.rename(column={"old_name": "new_name"})
```

通过使用`Table`类，用户能够以高效和灵活的方式管理和操作存储在Parquet文件中的数据，特别适用于处理量化投资中的大规模金融数据集。

## `Logger`类

`Logger`类通过定制化的日志格式和记录方式，增强了日志记录的可读性和灵活性。它支持在控制台（stream）和文件（file）中记录日志，并且可以分别为这两种输出定制格式。控制台日志支持颜色编码，而文件日志则以更传统的文本格式记录，便于文件存储和后期分析。

### 公开的接口

- `__init__(self, name: str = None, level: int = logging.DEBUG, stream: bool = True, file: str = None, display_time: bool = True, display_name: bool = False)`: 初始化一个`Logger`实例。
  - `name`: 日志记录器的名称，默认为`QuoolLogger`。
  - `level`: 日志级别，默认为`DEBUG`。
  - `stream`: 是否在控制台上显示日志，默认为`True`。
  - `file`: 日志文件的路径，如果指定，则会将日志同时输出到该文件，默认为`None`。
  - `display_time`: 是否在日志消息中显示时间，默认为`True`。
  - `display_name`: 是否在日志消息中显示记录器的名称，默认为`False`。

### 用法示例

```python
# 创建Logger实例，同时在控制台和文件中记录日志
logger = Logger(file="quool_log.txt")

# 只在控制台显示日志，显示时间但不显示日志记录器名称
logger_stream_only = Logger(display_name=True)

# 记录不同级别的日志
logger.debug("This is a debug message.")
logger.info("This is an info message.")
logger.warning("This is a warning.")
logger.error("This is an error message.")
logger.critical("This is critical!")
```

这两个方法是`quool`框架的工具模块（`util`）中提供的实用功能，旨在辅助进行数据处理和优化。以下是对这两个方法的概况介绍和使用说明。

`ItemTable`是`quool`框架中定义的一个`Table`子类，专门用于处理具有明确项目（Item）维度的数据表。这个类通过扩展了基类的`read`方法，提供了基于项目维度的数据查询功能，支持时间范围过滤和列筛选。

## `ItemTable`类

- 继承自`Table`基类，`ItemTable`利用`Table`提供的基础功能，专注于处理具有项目维度的数据。
- 通过重写`read`方法，增加了对时间范围（`start`和`stop`参数）和列筛选的支持。
- 支持使用过滤器来精确查询数据。

### 使用说明

#### 方法

- `read(column: str | list = None, start: str | list = None, stop: str = None, filters: list[list[tuple]] = None)`: 读取数据的方法，支持列选择、时间范围选择以及自定义过滤器。

#### 参数

- `column`: 指定要查询的列，可以是单个列名的字符串，也可以是多个列名组成的列表。
- `start`: 查询的开始时间点或者包含多个开始时间点的列表。
- `stop`: 查询的结束时间点。
- `filters`: 自定义的过滤条件，用于进一步限制查询结果。

### 示例用法

```python
from pathlib import Path

# 假设已有ItemTable的实例化对象item_table

# 读取特定列的数据
data = item_table.read(column="price")

# 读取指定时间范围内的数据
data = item_table.read(start="2022-01-01", stop="2022-12-31")

# 读取多个指定时间点的数据
data = item_table.read(start=["2022-01-01", "2022-06-01"])

# 使用自定义过滤条件读取数据
filters = [("price", ">", 100)]
data = item_table.read(filters=filters)
```

通过`ItemTable`，用户可以方便地查询和处理与特定项目相关的数据，尤其是在涉及时间序列数据处理时，该类提供了灵活的查询功能，以支持量化投资策略的数据分析需求。

`DatetimeTable`是`quool`框架中定义的另一个`Table`子类，专为处理具有时间维度的数据表设计。这个类通过定制化的初始化参数和方法，提供了基于时间范围的数据查询功能，支持对时间序列数据的高效管理和查询。

## `DatetimeTable`类

- 继承自`Table`基类，`DatetimeTable`扩展了针对时间序列数据的特定功能。
- 类的初始化方法除了`uri`和`create`参数外，还引入了`freq`和`format`参数，用于指定数据的时间频率和时间格式。
- 重写了`spliter`和`namer`属性，分别用于定义数据分割的逻辑和命名规则，这两个属性的定制是为了更好地管理时间序列数据。
- 通过重写`read`方法，实现了基于时间范围的数据读取，支持按时间点、时间范围进行高效的数据查询。

### 使用说明

#### 初始化参数

- `uri`: 数据存储路径。
- `freq`: 时间序列的频率，例如`'M'`代表月度数据。
- `format`: 时间格式字符串，用于格式化时间序列索引。
- `create`: 指定是否在路径不存在时创建路径。

#### 方法

- `read(field: str | list = None, start: str | list = None, stop: str = None, filters: list[list[tuple]] = None) -> pd.Series | pd.DataFrame`: 根据时间范围和字段筛选条件读取数据。

#### 参数

- `field`: 指定要查询的字段，可以是单个字段名的字符串，也可以是多个字段名组成的列表。
- `start`: 查询的开始时间点。
- `stop`: 查询的结束时间点。
- `filters`: 自定义的过滤条件，用于进一步限制查询结果。

### 示例用法

```python
from pathlib import Path

# 假设已有DatetimeTable的实例化对象datetime_table

# 读取特定字段的数据
data = datetime_table.read(field="volume")

# 读取指定时间范围内的数据
data = datetime_table.read(start="2022-01-01", stop="2022-12-31")

# 使用自定义过滤条件读取数据
filters = [("volume", ">", 1000)]
data = datetime_table.read(filters=filters)
```

`DatetimeTable`通过其专为时间序列数据设计的接口，为用户提供了一个强大且灵活的工具，用于处理和查询时间序列数据。这对于需要进行时间维度分析的量化投资策略特别有用。

`PanelTable`是`quool`框架中定义的又一个`Table`子类，专门设计用于处理具有面板数据结构的表格，即数据同时跨越了时间和另一个或多个维度（如证券代码）。这类数据常见于金融领域，尤其是在进行跨多个资产的时间序列分析时。

## `PanelTable`类

- 继承自`Table`基类，`PanelTable`专注于管理和查询面板数据。
- 类的初始化方法添加了`code_level`和`date_level`参数，用于指定数据中代码和日期的层级，以及`freq`和`format`参数，用于处理时间序列。
- 重写了`spliter`和`namer`属性，这两个属性通过定制，能够更好地管理面板数据中的时间维度。
- 通过重写`read`方法，实现了基于代码、时间范围和字段筛选的数据读取功能。

### 使用说明

#### 初始化参数

- `uri`: 数据存储路径。
- `code_level`: 代表代码的层级，可以是层级的名称或索引。
- `date_level`: 代表日期的层级，可以是层级的名称或索引。
- `freq`: 时间序列的频率，例如`'M'`代表月度数据。
- `format`: 时间格式字符串，用于格式化时间序列索引。
- `create`: 指定是否在路径不存在时创建路径。

#### 方法

- `read(field: str | list = None, code: str | list = None, start: str | list = None, stop: str = None, filters: list[list[tuple]] = None) -> pd.Series | pd.DataFrame`: 根据代码、时间范围和字段筛选条件读取数据。

#### 参数

- `field`: 要查询的字段，可以是单个字段名的字符串，也可以是多个字段名组成的列表。
- `code`: 证券代码，可以是单个代码的字符串，也可以是多个代码组成的列表。
- `start`: 查询的开始时间点。
- `stop`: 查询的结束时间点。
- `filters`: 自定义的过滤条件，用于进一步限制查询结果。

### 示例用法

```python
from pathlib import Path

# 假设已有PanelTable的实例化对象panel_table

# 读取特定字段和代码的数据
data = panel_table.read(field="close_price", code="000001")

# 读取指定代码和时间范围内的数据
data = panel_table.read(code=["000001", "000002"], start="2022-01-01", stop="2022-12-31")

# 使用自定义过滤条件读取数据
filters = [("volume", ">", 1000)]
data = panel_table.read(filters=filters)
```

通过`PanelTable`类，用户可以高效地处理和查询面板数据，使得跨时间和资产的复杂数据分析变得更加简单和直观。这对于需要进行复杂金融数据分析和投资策略研究的场景尤为重要。

`RunRecorder`类是`quool`框架中定义的一个专门用于记录量化模型运行数据的类，继承自`ItemTable`，通过添加特定的功能，使得对模型运行时的各种数据进行记录变得更加方便和直观。

## `RunRecorder`类

- **基础功能**：作为`ItemTable`的子类，`RunRecorder`继承了处理具有项目维度数据的所有基本功能。
- **特化功能**：通过定制`__init__`构造函数和新增`record`方法，`RunRecorder`专注于记录模型运行时的关键数据，如参数、性能指标等。
- **灵活的数据记录**：自动区分已存在的数据列和新列，对已存在的数据进行更新，对新的数据列进行添加，极大地提高了数据记录的灵活性和便捷性。

### 方法介绍

#### 初始化

```python
__init__(self, uri: str | Path, model_path: str | Path)
```
- `uri`：数据存储路径。
- `model_path`：模型相关文件的存储路径。

#### 属性

- `spliter`和`namer`：定制的属性，用于数据分割和命名，`RunRecorder`类将数据分割的频率设置为每天，并以日期命名数据片段。

#### 记录方法

```python
record(self, **kwargs)
```
- 动态接收任意数量的关键字参数，每个关键字代表要记录的数据字段，参数值为相应的数据。
- 将传入的数据记录到`DataFrame`中，并根据字段是否存在于当前表格中，自动选择更新或添加操作。

### 使用实例

假设你正在运行一个量化模型，并希望记录模型每次运行时的不同配置参数和性能指标：

```python
from pathlib import Path

# 实例化RunRecorder，指定数据存储路径和模型路径
run_recorder = RunRecorder(uri=Path("/path/to/data/runs"), model_path=Path("/path/to/model"))

# 记录模型的运行参数和性能指标
run_recorder.record(epoch=10, loss=0.05, accuracy=0.98)

# 在另一次运行时，记录不同的参数和指标
run_recorder.record(epoch=20, loss=0.03, accuracy=0.99, new_metric=0.95)
```

通过使用`RunRecorder`，可以非常方便地记录和追踪量化模型在不同运行时的表现和配置，这对于模型的调优和性能分析至关重要。

`TradeRecorder`类是`quool`框架中为记录交易信息而设计的`ItemTable`子类，专注于记录和管理交易活动的数据。通过扩展了基类的功能，`TradeRecorder`提供了一套便捷的接口，用于捕获交易的各种细节，如交易时间、代码、数量、价格、金额和佣金等。

## `TradeRecorder`类

- **基础功能**：继承自`ItemTable`，具备处理具有项目维度数据的所有功能。
- **特化功能**：通过特化的`__init__`构造函数和新增的`trade`、`peek`、`report`和`evaluate`方法，`TradeRecorder`为量化交易提供了记录交易、查看持仓、生成报告和评估性能的便利方法。
- **灵活的交易记录**：自动管理交易记录的添加和更新，支持现金流的记录，以及交易产生的各种成本和收益的计算。

### 方法介绍

#### 初始化
```python
__init__(self, uri: str | Path, principle: float = None, start: str | pd.Timestamp = None)
```
- `uri`：数据存储路径。
- `principle`：起始资金。
- `start`：开始日期。

#### 记录交易
```python
trade(self, date, code, size=None, price=None, amount=None, commission=0, **kwargs)
```
- 记录一次交易活动，自动处理现金流。

#### 查看持仓

```python
peek(self, date: str | pd.Timestamp = None, price: pd.Series = None) -> pd.Series
```
- 根据给定日期查看持仓和成本，可选地提供价格序列以计算市值和盈亏。

#### 生成报告

```python
report(self, price: pd.Series, code_level: int | str = 0, date_level: int | str = 1) -> pd.DataFrame
```
- 根据价格信息生成交易和持仓的综合报告。

#### 评估性能

```python
evaluate(value: pd.Series, cash: pd.Series = None, turnover: pd.Series = None, benchmark: pd.Series = None, image: str = None, result: str = None)
```
- 对交易性能进行评估，生成性能指标和可选的图表和结果文件。

`ProxyRecorder`类是`quool`框架中专为获取和管理代理服务器信息而设计的`ItemTable`子类。它通过定义多个数据源的抓取方法，允许用户从多个公开代理网站获取代理信息，并通过一系列检查确保代理的有效性。此外，`ProxyRecorder`还提供了一套便利的接口，用于随机选择代理、检查代理有效性，并管理代理信息的存储。

### 使用实例

假设你正在运行一个量化策略，并希望记录策略执行的交易信息及其性能评估：

```python
from pathlib import Path

# 实例化TradeRecorder，指定数据存储路径、起始资金和开始日期
trade_recorder = TradeRecorder(uri=Path("/path/to/data/trades"), principle=100000, start="2022-01-01")

# 记录一次交易活动
trade_recorder.trade(date="2022-01-10", code="000001", size=100, price=10.0)

# 查看持仓
position = trade_recorder.peek(date="2022-01-10")

# 使用最新价格生成报告
price_series = pd.Series({...})  # 假设这是最新的价格序列
report = trade_recorder.report(price=price_series)

# 评估交易性能
trade_recorder.evaluate(value=report['value'], cash=report['cash'], turnover=report['turnover'], image="performance.png", result="performance.xlsx")
```

通过使用`TradeRecorder`，量化交易者可以轻松地记录、管理和评估其交易策略的执行情况，从而优化策略表现和风险控制。

## `ProxyRecorder`类

- **基础功能**：继承自`ItemTable`，具备处理具有项目维度数据的所有功能。
- **代理抓取和管理**：通过`add_kxdaili`、`add_kuaidaili`、`add_ip3366`和`add_89ip`方法，`ProxyRecorder`可以从不同的公开代理网站抓取代理信息。
- **代理检查**：通过`check`方法，对获取到的代理进行有效性检查，确保代理可用。
- **代理选择**：`pick`方法允许用户基于已存储的代理信息随机选择一个有效代理。

### 方法介绍

#### 初始化

```python
__init__(self, uri: str | Path, create: bool = False)
```

- `uri`：数据存储路径。
- `create`：指定是否在路径不存在时创建路径。

#### 抓取代理

- 提供了四个方法`add_kxdaili`、`add_kuaidaili`、`add_ip3366`、`add_89ip`用于从不同的代理网站抓取代理信息。

#### 检查代理有效性

```python
check(self, proxy: dict, timeout: int = 2)
```
- 对给定的代理进行有效性检查。

#### 选择代理

```python
pick(self, field: str | list = None)
```

- 从存储的代理信息中随机选择一个有效代理。

#### 忘记已选代理

```python
forget(self)
```

- 重置代理选择记录，允许之前被选择过的代理再次被选择。

### 使用实例

假设你需要获取有效的代理服务器信息，并使用这些代理进行网络请求：

```python
from pathlib import Path

# 实例化ProxyRecorder，指定数据存储路径
proxy_recorder = ProxyRecorder(uri=Path("/path/to/data/proxies"))

# 从不同的代理网站抓取代理信息
proxy_recorder.add_kxdaili(pages=2)
proxy_recorder.add_kuaidaili(pages=2)
proxy_recorder.add_ip3366(pages=2)
proxy_recorder.add_89ip(pages=2)

# 随机选择一个有效的代理
proxy = proxy_recorder.pick()

# 使用选择的代理进行网络请求
response = requests.get("http://example.com", proxies=proxy)
```

通过使用`ProxyRecorder`，用户可以轻松地管理和使用代理服务器，这在需要绕过网络限制或隐藏真实IP地址时非常有用。此外，`ProxyRecorder`还能够保证所使用的代理的有效性，提高网络请求的成功率。
`Factor`类是`quool`框架中针对因子投资领域设计的`PanelTable`子类。这个类通过简化因子数据的存储、管理和测试流程，为用户提供了一个高效的因子研究平台。用户只需定义因子的存储路径和计算逻辑，就能轻松进行因子的分层测试、Top K测试、IC测试以及因子分布图的绘制。

### `Factor`类

- **基础功能**：继承自`PanelTable`，`Factor`类提供了面板数据的基础管理功能。
- **因子数据处理**：通过重写`read`方法，增加了对因子数据处理流程的支持，包括自定义处理器的应用。
- **未来收益率计算**：`get_future`方法用于计算未来一定周期内的收益率，支持不同的价格类型和时间周期。
- **交易日处理**：提供了获取交易日列表和回滚到特定交易日的方法。
- **因子数据存储**：`save`方法允许用户将计算得到的因子数据存入数据库。
- **因子测试功能**：包括分层测试（`perform_grouping`）、Top K测试（`perform_topk`）、信息系数（IC）测试（`perform_inforcoef`）和因子分布图绘制（`perform_crosssection`）。

### 方法介绍

#### 因子数据读取

```python
read(self, field, code=None, start=None, stop=None, processor=None)
```
- 读取因子数据，支持对数据进行预处理。

#### 保存因子数据

```python
save(self, df, name=None)
```
- 将因子数据存储到数据库中。

#### 分层测试

```python
perform_grouping(self, name, period=1, start=None, stop=None, ptype="volume_weighted_price", ngroup=5, commission=0.002, image=True, result=None)
```
- 将因子数据分成多个分组，并计算每个分组的未来收益率。

#### Top K测试

```python
perform_topk(self, name, period=1, start=None, stop=None, ptype="volume_weighted_price", topk=100, commission=0.002, image=True, result=None)
```
- 选择因子分数最高的Top K个标的，并计算这些标的的未来收益率。

#### IC测试

```python
perform_inforcoef(self, name, period=1, start=None, stop=None, ptype="volume_weighted_price", rolling=20, method='pearson', image=True, result=None)
```
- 计算因子值与未来收益率之间的相关系数（信息系数IC）。

#### 因子分布图绘制

```python
perform_crosssection(self, name, date, processor=None, period=1, ptype="volume_weighted_price", image=True, result=None)
```
- 绘制特定日期的因子截面分布图。

### 使用实例

假设你已经计算了一个名为`momentum`的因子，并想要进行IC测试和分层测试：

```python
from pathlib import Path

# 实例化Factor类，指定因子数据的存储路径
factor = Factor(uri=Path("/path/to/data/factors"))

# 进行IC测试
ic_result = factor.perform_inforcoef(name="momentum", start="2020-01-01", stop="2020-12-31", image="momentum_ic.png", result="momentum_ic.xlsx")

# 进行分层测试
group_result = factor.perform_grouping(name="momentum", start="2020-01-01", stop="2020-12-31", ngroup=5, image="momentum_group.png", result="momentum_group.xlsx")
```

通过`Factor`类，用户可以轻松完成因子的存储、管理和各种统计测试，极大地提升了因子研究的效率和便捷性。

## 其他方法

### `parse_commastr(commastr: 'str | list') -> list`

此方法用于解析以逗号分隔的字符串，将其转换为字符串列表。如果输入已经是列表，则直接返回该列表。

#### 参数

- `commastr`: 可以是一个逗号分隔的字符串或者是一个列表。

#### 返回值

- 返回一个列表，包含分割后的字符串。

#### 使用说明

```python
# 使用逗号分隔的字符串
comma_str = "a,b,c"
parsed_list = parse_commastr(comma_str)
# 输出: ['a', 'b', 'c']

# 直接使用列表
list_input = ["a", "b", "c"]
parsed_list = parse_commastr(list_input)
# 输出: ["a", "b", "c"]
```

### `reduce_mem_usage(df: pd.DataFrame)`

此方法用于减少`DataFrame`的内存占用，通过将数据列的数据类型调整为更高效的类型来实现。它对整数、浮点数、和对象类型的列进行优化，并利用日志记录优化前后的内存使用情况。

#### 参数

- `df`: 需要优化内存使用的`DataFrame`。

#### 返回值

- 返回优化内存使用后的`DataFrame`。

#### 使用说明

```python
import pandas as pd
import numpy as np

# 创建一个DataFrame
df = pd.DataFrame({
    'A': np.random.randint(0, 100, size=10000),
    'B': np.random.rand(10000) * 100,
    'C': ['string' + str(i) for i in range(10000)]
})

# 减少内存使用
optimized_df = reduce_mem_usage(df)

# 查看优化前后的内存使用情况
# 日志信息将显示优化详情
```


"""
Quool is dedicated to provide a variety of tools in quantative analysis.
The modules, notations and example usages are presented as below:

**calculator**
------------------

The 'calculator' module in this package offers a suite of classes for various financial data operation tasks. 
These classes are specialized in data standardization, missing value handling, outlier processing, and correlation computation, tailored for financial data analysis.

Included in this module are:

Classes:
- Preprocessor: Serves as the foundational class for data preprocessing tasks. It is designed to be extended for specific preprocessing requirements.
- Corr: A subclass of Preprocessor, this class is used for calculating correlations between two datasets. It supports various methods like Pearson correlation.
- FillNa: A subclass of Preprocessor, FillNa provides multiple strategies for handling missing values in data, including zero fill, mean, median, forward fill, and backward fill.
- Standarize: As a subclass of Preprocessor, Standarize offers functionality for data standardization using methods like Z-score normalization or Min-Max scaling.
- DeOutlier: Also a subclass of Preprocessor, DeOutlier is designed for outlier detection and handling in data using techniques such as Median Absolute Deviation (MAD), standard deviation, or replacing outliers with NaNs.

Usage:
Each class in the 'calculator' module is compatible with pandas DataFrame and Series objects. Instantiate these classes with data and use the 'proc' method to process the data as per the operation specific to each class.

Example:
    from my_package.calculator import Standarize

    # Instantiate the Standarize class
    standardizer = Standarize(data)
    # Apply Z-score standardization to the data
    standardized_data = standardizer(method='zscore')

**backtest**
------------------

The 'backtest' module provides a framework for backtesting trading strategies in financial markets. 
Built on top of the Backtrader library, it offers enhanced functionality with custom classes and utilities 
for strategy implementation, analysis, and performance tracking.

Included in this module are:

Classes:
- Strategy: A base class for defining trading strategies. It includes methods for order placement, 
  logging, and handling notifications for orders and trades. Custom strategies can be built by 
  extending this class.
- RebalanceStrategy: A subclass of Strategy, designed for strategies that involve periodic rebalancing 
  of asset holdings.
- Indicator: A base class for creating custom technical indicators.
- Analyzer: A base class for building custom analyzers to assess the performance of strategies.
- Observer: A base class for creating custom observers to monitor aspects of the trading environment.
- TradeOrderRecorder: An Analyzer subclass for recording detailed information about each order and trade executed.
- CashValueRecorder: An Analyzer subclass for tracking the cash and total value of the trading account over time.
- Cerebro: A wrapper class around the Backtrader's Cerebro engine, facilitating strategy execution, data 
  handling, and analysis aggregation.

This module simplifies the process of strategy development and backtesting by providing a structured 
approach to define and analyze trading strategies in a simulated environment.

Example:
    from backtest import Cerebro, Strategy
    cerebro = Cerebro(data=my_dataframe)
    cerebro.run(strategy=MyStrategy)

**database**
------------------

The 'database' module in this package provides a set of classes for managing and manipulating large datasets, 
especially in the context of financial data. It is designed to handle data in a fragmented, file-based structure 
using Parquet files, offering efficient storage and access patterns.

This module includes:

Classes:
- Table: A foundational class for managing datasets in a fragmented, file-based structure. 
  It allows for efficient reading, updating, and manipulation of large datasets stored as Parquet files.
- FrameTable: A subclass of Table, FrameTable is tailored for handling DataFrame objects with enhanced 
  index management capabilities. It is particularly useful for datasets where index management is critical.
- PanelTable: A specialized subclass of Table designed for managing panel data. It is ideal for datasets 
  with time and categorical indexing, such as financial time series data involving multiple stocks or assets.

The module is built to handle complex data storage scenarios, allowing for operations such as adding new data, 
updating existing data, deleting rows or columns, and more. It is optimized for scenarios where traditional 
database systems might not be efficient, particularly in handling large-scale financial datasets.

Example:
    from database import PanelTable
    panel_table = PanelTable(uri="path/to/panel/data")
    data = panel_table.read(field="price", code="AAPL", start="2020-01-01", stop="2020-12-31")

**request**
------------------

The 'request' module in this package offers an extensive range of classes and methods for handling web requests 
and interacting with APIs. It is designed to simplify tasks such as data scraping, API communication, 
and processing web content in various formats.

This module includes:

Classes:
- Request: A versatile class for making HTTP requests. It supports features like automatic retries, 
  response delay handling, and parallel request processing. It also provides properties for easy 
  response parsing and custom callback functions.
- KaiXin, KuaiDaili, Ip3366, Ip98: Specialized classes for scraping proxy data from specific websites. 
  These classes extend the Request class and provide custom callback methods to parse the scraped data.
- Checker: A class designed to check the validity of proxy addresses.
- WeiXin: A class providing an interface to interact with WeChat, including functionalities like 
  QR code-based login and sending notifications through WeChat Work.
- WeiboSearch: A class for scraping and parsing data from Weibo searches.
- AkShare: A class that provides methods to fetch financial data using the AkShare API.
- Em: A class to interact with East Money (东方财富) for financial data and analysis.
- StockUS: A class to interact with the US stock API, providing functionalities to get stock prices, 
  index prices, and research reports.

Each class is equipped with methods to handle specific types of requests and parse responses appropriately. 
The module aims to facilitate complex web scraping and data extraction tasks, particularly in the domain 
of financial data analysis.

Usage Example:
---------------
# Example of using the Request class for making parallel requests
req = Request(url=["https://example.com"], method="get")
response = req(para=True)

# Example of using the WeiXin class to send a notification
WeiXin.notify(key='your_webhook_key', message='Hello, WeChat!', message_type='text')

**tool**
------------------

The 'tool' module in this package offers a range of utility functions and classes to support common operations 
such as logging, data formatting, and memory management. These tools are designed to enhance efficiency and 
ease of use in data processing and logging tasks.

This module includes:

Classes:
- Logger: An enhanced logging class that extends Python's standard logging.Logger, providing additional 
  functionalities like console and file logging with customizable display options.
- __TimeFormatter, _StreamFormatter, _FileFormatter: Internal classes used by Logger for formatting log messages.

Functions:
- parse_date: Converts date strings to pandas.Timestamp objects, supporting a range of formats and error handling options.
- parse_commastr: Parses comma-separated strings into a list of strings or returns the input list as is.
- panelize: Ensures that a DataFrame or Series with a MultiIndex covers all combinations of index levels.
- reduce_mem_usage: Optimizes memory usage of a pandas DataFrame by downcasting data types.
- format_code: Formats a stock code according to a specified pattern, useful for standardizing financial data.
- strip_stock_code: Strips the market prefix from a stock code.

These utilities are particularly useful in data analysis and financial data processing where tasks like logging, 
date parsing, and memory optimization are common.

Usage Example:
---------------
# Using Logger for enhanced logging
logger = Logger(name="MyLogger", level=logging.INFO, file="log.txt")
logger.info("This is an info message")

# Using parse_date to convert date strings to pandas.Timestamp
date = parse_date("2021-01-01")

# Reducing memory usage of a DataFrame
optimized_df = reduce_mem_usage(original_df)
"""


__all__ = ["backtest", "calculator", "database", "model", "request", "tool"]
__version__ = "1.2.2"

import numpy as np
import pandas as pd
from .util import evaluate
from .base import StrategyBase, OrderBase


class Strategy(StrategyBase):
    """Concrete strategy implementation for algorithmic trading systems.

    Inherits from StrategyBase to provide complete trading framework. Users should
    subclass this to create custom strategies by implementing the update() method.

    Key Features:
        - Pre-built order creation methods (market/limit orders)
        - Position management utilities
        - Performance evaluation system
        - Built-in parallel backtesting

    Usage Workflow:
        1. Subclass Strategy and implement required methods
        2. Define trading logic in update()
        3. Use built-in order methods to execute trades
        4. Run backtest with historical data
        5. Analyze results using evaluate()

    Required Overrides:
        def update(self, **kwargs):
            '''Called at each timestep - implement trading logic here'''

    Example Strategy Template:

        class MyStrategy(Strategy):
            def init(self):
                # Initialization code (indicator setup, etc)
                self.add_indicator('SMA20', SimpleMovingAverage(window=20))
                
            def update(self):
                # Trading logic executed every timestep
                data = self.source.data
                
                # Example crossover strategy
                if crossover(data['close'], self.indicators['SMA20']):
                    self.buy(
                        code='AAPL',
                        quantity=100,
                        exectype=Order.MARKET
                    )
                    
                elif crossover(self.indicators['SMA20'], data['close']):
                    self.sell(
                        code='AAPL',
                        quantity=100,
                        exectype=Order.MARKET
                    )

    Order Methods:
        buy()       - Open long position
        sell()      - Close long position
        close()     - Close entire position
        order_target_value()  - Adjust position to target value
        order_target_percent() - Adjust position to target percentage

    Attributes:
        source (SourceBase): Data feed object
        broker (BrokerBase): Trading broker instance
        status (str): Current strategy state (INIT/RUNNING/STOPPED)
        logger (Logger): Optional logging instance

    Methods:
        log(): Record strategy messages
        get_value(): Get current portfolio value
        get_positions(): Get current holdings
        evaluate(): Generate performance report
        cancel(): Cancel pending order

    Example Usage:

        # 1. Strategy Implementation
        class MomentumStrategy(Strategy):
            def init(self):
                # Initialize 50-day momentum indicator
                self.momentum_window = 50
                self.threshold = 0.05
                
            def update(self):
                for code in self.source.data.index.get_level_values('code').unique():
                    close_prices = self.source.data.xs(code, level='code')['close']
                    
                    if len(close_prices) < self.momentum_window:
                        continue
                        
                    momentum = (close_prices[-1] / close_prices[-self.momentum_window] - 1)
                    
                    if momentum > self.threshold:
                        # Buy 5% of portfolio value
                        self.order_target_percent(
                            code=code,
                            target=0.05,
                            exectype=Order.LIMIT,
                            limit=close_prices[-1] * 0.995  # Limit at 0.5% below market
                        )
                        
                    elif momentum < -self.threshold:
                        # Close existing position
                        self.close(
                            code=code,
                            exectype=Order.MARKET
                        )

        # 2. Running Backtest
        data_source = DataFrameSource(historical_data)
        broker = BacktestBroker()
        
        strategy = MomentumStrategy(
            id="Momentum_v1",
            source=data_source,
            broker=broker
        )
        
        # Run single backtest
        strategy(params={})
        
        # Parallel parameter optimization
        params_grid = {
            'params1': {
                'momentum_window': 20,
                'param2': 0.03,
            },
            'params2': {
                'momentum_window': 50,
                'param2': 0.05,
            },
            'params3': {
                'momentum_window': 100,
                'param2': 0.07,
            },
        }
        strategy(params=params_grid, n_jobs=4)
        
        # 3. Analyzing Results
        results = strategy.evaluate(benchmark=sp500_index)
        print(results['evaluation'])
        '''
        {
            'Sharpe Ratio': 1.32,
            'Max Drawdown': -15.2,
            'Annual Return': 12.5,
            'Win Rate': 55.6,
            'Profit Factor': 1.8
        }
        '''

    Order Execution Patterns:

        # Market Order - Immediate execution
        self.buy(code='TSLA', quantity=100, exectype=Order.MARKET)
        
        # Limit Order - Price constrained
        self.sell(
            code='AAPL',
            quantity=200,
            exectype=Order.LIMIT,
            limit=150.00  # Won't execute above this price
        )
        
        # Target Position Sizing
        # Achieve $10,000 position in MSFT
        self.order_target_value(
            code='MSFT',
            value=10000,
            exectype=Order.MARKET
        )
        
        # Close all positions in NVDA
        self.close(code='NVDA', exectype=Order.MARKET)

    Important Notes:
        1. All order quantities must be positive integers
        2. Market orders execute at next available price
        3. Limit orders may not fill immediately
        4. Positions are tracked per security
        5. Timestamps are managed automatically by data source
    """

    def log(self, message: str, level: str = "DEBUG"):
        if self.logger is not None:
            self.logger.log(level, f"[{self.manager.time}]: {message}")
        else:
            print(f"[{self.manager.time}]: {message}")

    def get_value(self):
        return self.broker.get_value(self.source.data)
    
    def get_positions(self):
        return self.broker.get_positions()

    def buy(
        self,
        code: str,
        quantity: int,
        exectype: str,
        limit: float = None,
        trigger: float = None,
        id: str = None,
        valid: str = None,
    ) -> OrderBase:
        return self.broker.create(
            side=self.broker.order_type.BUY,
            code=code,
            quantity=quantity,
            exectype=exectype,
            limit=limit,
            trigger=trigger,
            id=id,
            valid=valid,
        )

    def sell(
        self,
        code: str,
        quantity: int,
        exectype: str,
        limit: float = None,
        trigger: float = None,
        id: str = None,
        valid: str = None,
    ) -> OrderBase:
        return self.broker.create(
            side=self.broker.order_type.SELL,
            code=code,
            quantity=quantity,
            exectype=exectype,
            limit=limit,
            trigger=trigger,
            id=id,
            valid=valid,
        )
    
    def order_target_value(
        self,
        code: str,
        value: float,
        exectype: str,
        limit: float = None,
        trigger: float = None,
        id: str = None,
        valid: str = None,
    ) -> OrderBase:
        quantity = (value / self.source.data.loc[code, "close"] // 100) * 100
        if quantity > 0:
            side = self.broker.order_type.BUY
        else:
            side = self.broker.order_type.SELL
        return self.broker.create(
            side=side,
            code=code,
            quantity=abs(quantity),
            exectype=exectype,
            limit=limit,
            trigger=trigger,
            id=id,
            valid=valid,
        )

    def order_target_percent(
        self,
        code: str,
        value: float,
        exectype: str,
        limit: float = None,
        trigger: float = None,
        id: str = None,
        valid: str = None,
    ) -> OrderBase:
        quantity = (value / self.source.data.loc[code, "close"] // 100) * 100
        if quantity > 0:
            side = self.broker.order_type.BUY
        else:
            side = self.broker.order_type.SELL
        return self.broker.create(
            side=side,
            code=code,
            quantity=abs(quantity),
            exectype=exectype,
            limit=limit,
            trigger=trigger,
            id=id,
            valid=valid,
        )

    def close(
        self,
        code: str,
        exectype: str,
        limit: float = None,
        trigger: float = None,
        id: str = None,
        valid: str = None,
    ) -> OrderBase:
        return self.broker.create(
            side=self.broker.order_type.SELL,
            code=code,
            quantity=self.broker.positions.get(code),
            exectype=exectype,
            limit=limit,
            trigger=trigger,
            id=id,
            valid=valid,
        )

    def evaluate(self, benchmark: pd.Series = None):
        ledger = self.broker.ledger.set_index(["time", "code"]).sort_index()
        prices = self.source.datas["close"].unstack("code")

        # cash, position, trades, total_value, market_value calculation 
        cash = ledger.groupby("time")[["amount", "commission"]].sum()
        cash = (cash["amount"] - cash["commission"]).cumsum()
        positions = ledger.drop(index="CASH", level=1).groupby(["time", "code"])["unit"].sum().unstack().fillna(0).cumsum()
        timepoints = prices.index.union(cash.index).union(positions.index)
        cash = cash.reindex(timepoints).ffill()
        positions = positions.reindex(timepoints).ffill().fillna(0)
        market = (positions * prices).sum(axis=1)
        total = cash + market
        delta = positions.diff()
        delta.iloc[0] = positions.iloc[0]
        turnover = (delta * prices).abs().sum(axis=1) / total.shift(1).fillna(cash.iloc[0])
        
        ledger = ledger.drop(index="CASH", level=1)
        ledger["stock_cumsum"] = ledger.groupby("code")["unit"].cumsum()
        ledger["trade_mark"] = ledger["stock_cumsum"] == 0
        ledger["trade_num"] = ledger.groupby("code")["trade_mark"].shift(1).astype("bool").groupby("code").cumsum()
        trades = ledger.groupby(["code", "trade_num"]).apply(
            lambda x: pd.Series({
                "open_amount": -x[x["unit"] > 0]["amount"].sum(),
                "open_at": x[x["unit"] > 0].index.get_level_values("time")[0],
                "close_amount": x[x["unit"] < 0]["amount"].sum() if x["unit"].sum() == 0 else np.nan,
                "close_at": x[x["unit"] < 0].index.get_level_values("time")[-1] if x["unit"].sum() == 0 else np.nan,
            })
        )
        if not trades.empty:
            trades["duration"] = pd.to_datetime(trades["close_at"]) - pd.to_datetime(trades["open_at"])
            trades["return"] = (trades["close_amount"] - trades["open_amount"]) / trades["open_amount"]
        else:
            trades = pd.DataFrame(columns=["open_amount", "open_at", "close_amount", "close_at", "duration", "return"])
        return {
            "evaluation": evaluate(total, benchmark=benchmark, turnover=turnover, trades=trades),
            "values": pd.concat(
                [total, market, cash, turnover], 
                axis=1, keys=["total", "market", "cash", "turnover"]
            ),
            "positions": positions,
            "trades": trades,
        }
        
    def cancel(self, order_or_id: str | OrderBase) -> OrderBase:
        return self.broker.cancel(order_or_id)

    def update(self, **kwargs):
        raise NotImplementedError("`update` method must be implemented")

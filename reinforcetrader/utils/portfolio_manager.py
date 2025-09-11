import pandas as pd
import matplotlib.pyplot as plt
from pprint import pformat

class PortfolioBackTester:
    def __init__(self, signals: pd.DataFrame, prices: pd.DataFrame):
        self._signals = signals 
        self._prices = prices
        
        # Basic sanity checks
        if not self._signals.index.equals(self._prices.index):
            raise ValueError("signals and prices must share identical index (dates).")
        if list(self._signals.columns) != list(self._prices.columns):
            raise ValueError("signals and prices must share identical columns (tickers).")

        self._returns = None
        self._positions = None
        self._strategy_returns = None
    
    def _compute_strat_returns(self):
        # First, drop any rows wihout signal dict (NaN)
        self._signals = self._signals.dropna()
        self._prices = self._prices.loc[self._signals.index]
        
        # Compute the daily returns and align the index
        self._returns = self._prices.pct_change().shift(-1).dropna()
        self._signals = self._signals.loc[self._returns.index]
        
        # Compute daily in trade signals
        self._in_trade = pd.DataFrame(index=self._signals.index, columns=self._signals.columns, dtype=bool)
        
        for i in range(len(self._signals)):
            # Get the signal dict for the day for each ticker
            signal_dicts = self._signals.iloc[i].to_list()
            
            # Find weather we are in trade or not
            if i > 0:
                daily_in_trades = self._in_trade.iloc[i - 1].tolist()
            else:
                daily_in_trades = [False] * self._signals.shape[1]
                
            row_pos = []
            for prev, sig in zip(daily_in_trades, signal_dicts):
                if isinstance(sig, dict) and sig.get('buy', 0) == 1:
                    row_pos.append(True)
                elif isinstance(sig, dict) and sig.get('sell', 0) == 1:
                    row_pos.append(False)
                else:  # hold-in or hold-out (or malformed cell)
                    row_pos.append(prev)

            self._in_trade.iloc[i] = pd.Series(row_pos, index=self._in_trade.columns)

        
        # Compute the cumulative trade returns
        daily_strategy_returns = (self._returns * self._in_trade.astype(float)).mean(axis=1)
        self._strategy_returns = (1 + daily_strategy_returns).cumprod()
        
        # Compute baseline returns (buy and hold)
        self._baseline_returns = (1 + self._returns.mean(axis=1)).cumprod()
        
    def run_backtest(self):
        # Compute comulative strategy returns
        self._compute_strat_returns()
        
        # Plot the returns and compare to baseline (buy and hold)
        self.plot_returns()
        
        # Compute performance metrics
        print(self.get_performance_metrics())
        
    def plot_returns(self):
        if self._strategy_returns is None or self._baseline_returns is None:
            raise ValueError("You must run the backtest before plotting returns.")
        
        plt.figure(figsize=(12, 6))
        plt.plot(self._strategy_returns, label='Strategy Returns')
        plt.plot(self._baseline_returns, label='Baseline Returns')
        plt.title('Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid()
        plt.show()
    
    def get_performance_metrics(self) -> str:
        if self._strategy_returns is None:
            raise ValueError("You must run the backtest before computing performance metrics.")
        
        total_return = self._strategy_returns.iloc[-1] - 1
        baseline_return = self._baseline_returns.iloc[-1] - 1
        
        # Compute annualized return assuming 252 trading days
        num_days = len(self._strategy_returns)
        annualized_return = (self._strategy_returns.iloc[-1]) ** (252 / num_days) - 1
        baseline_annualized_return = (self._baseline_returns.iloc[-1]) ** (252 / num_days) - 1
        
        # Compute maximum drawdown
        rolling_max = self._strategy_returns.cummax()
        drawdown = (self._strategy_returns - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        baseline_rolling_max = self._baseline_returns.cummax()
        baseline_drawdown = (self._baseline_returns - baseline_rolling_max) / baseline_rolling_max
        baseline_max_drawdown = baseline_drawdown.min()
        
        metrics = {
            "Total Return": total_return,
            "Annualized Return": annualized_return,
            "Max Drawdown": max_drawdown,
            "Baseline Total Return": baseline_return,
            "Baseline Annualized Return": baseline_annualized_return,
            "Baseline Max Drawdown": baseline_max_drawdown
        }
        
        return pformat(metrics)
import math
import pandas as pd
import matplotlib.pyplot as plt

from pprint import pformat
from typing import Optional
from pandas.tseries.offsets import DateOffset

from ..data_pipeline import RawDataLoader

class BackTester:
    # Define constants for in-trade and out-of-trade positions
    OUT_TRADE = 0
    IN_TRADE = 1
    
    # Define constants that represent agent behaviour
    A_BUY = 0
    A_HOLD = 1
    A_SELL = 2
    def __init__(self, signals: pd.DataFrame, prices: pd.DataFrame, benchmark_index: str='DJI') -> None:
        self._signals = signals 
        self._prices = prices
        print(BackTester.OUT_TRADE)
        
        # Basic sanity checks
        if not self._signals.index.equals(self._prices.index):
            raise ValueError("signals and prices must share identical index (dates).")
        if list(self._signals.columns) != list(self._prices.columns):
            raise ValueError("signals and prices must share identical columns (tickers).")

        # Clean the signals and prices data (e.g. remove NaN rows)
        self._clean_signals_data()
        
        # Download the benchmark index data and compute daily returns
        start_date = self._signals.index[0]
        end_date = self._signals.index[-1]
        self._index_returns = self._get_bench_returns(benchmark_index, start_date, end_date)
        
        # Initalize variables storing various cumulative returns
        self._cum_strategy_returns: Optional[pd.Series] = None
        self._cum_EWP_returns: Optional[pd.Series] = None
        self._cum_index_returns: Optional[pd.Series] = None
    
    def _clean_signals_data(self) -> None:
        # First, drop any rows without signal dict (NaN)
        self._signals = self._signals.dropna()
        self._prices = self._prices.loc[self._signals.index]
        
        # Compute the daily returns and align the index
        self._returns = self._prices.pct_change().shift(-1).dropna()
        self._signals = self._signals.loc[self._returns.index]
    
    def _get_bench_returns(self, benchmark_index, start_date, end_date: str) -> pd.Series:
        # Download the benchmark index data and compute daily returns
        benchmark_ticker = '^DJI' if benchmark_index == 'DJI' else '^SPX'
        bench_dl = RawDataLoader(start_date=start_date - DateOffset(days=3),
                                 end_date=end_date,
                                 tickers=[benchmark_ticker],
                                 verbose=False)
        bench_prices, _ = bench_dl.get_hist_prices()
        
        # Just keep the closing prices
        bench_prices = bench_prices['Close']
        
        return bench_prices.pct_change().loc[start_date: end_date]  
    
    def _compute_strat_returns(self):
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
                if isinstance(sig, dict) and sig.get('action', '') == 'buy':
                    row_pos.append(True)
                elif isinstance(sig, dict) and sig.get('action', '') == 'sell':
                    row_pos.append(False)
                else:  # hold-in or hold-out (or malformed cell)
                    row_pos.append(prev)

            self._in_trade.iloc[i] = pd.Series(row_pos, index=self._in_trade.columns)

        # Daily strategy returns (equal-weighted across tickers)
        self._daily_strategy_returns = (self._returns * self._in_trade.astype(float)).mean(axis=1)

        # Compute the daily baseline returns
        self._daily_ewp_returns = self._returns.mean(axis=1)
        self._daily_index_returns = self._index_returns.reindex(self._daily_strategy_returns.index).fillna(0.0)

        # Compute the cumulative returns
        self._cum_strategy_returns = (1 + self._daily_strategy_returns).cumprod()
        self._cum_EWP_returns = (1 + self._daily_ewp_returns).cumprod()
        self._cum_index_returns = (1 + self._daily_index_returns).cumprod()
        
    def run_backtest(self):
        # Compute comulative strategy returns
        self._compute_strat_returns()
        
        # Plot the returns and compare to baseline (buy and hold)
        self._plot_returns()
        
        # Compute performance metrics
        print(self.get_performance_metrics())
    
    def _plot_returns(self):
        assert self._cum_strategy_returns is not None
        assert self._cum_EWP_returns is not None
        assert self._cum_index_returns is not None
        
        plt.figure(figsize=(12, 6))
        plt.plot(self._cum_strategy_returns, label='Strategy Returns')
        plt.plot(self._cum_EWP_returns, label='EWP Returns')
        plt.plot(self._cum_index_returns, label='Index Returns (Buy/Hold)')
        
        plt.title('Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid()
        plt.show()
    
    
    def get_performance_metrics(self) -> str:
        assert self._cum_strategy_returns is not None
        assert self._cum_EWP_returns is not None
        assert self._cum_index_returns is not None
          
        final_strat_return = self._cum_strategy_returns.iloc[-1] - 1
        final_EWP_return = self._cum_EWP_returns.iloc[-1] - 1
        final_index_return = self._cum_index_returns.iloc[-1] - 1
        
        # Compute annualized return assuming 252 trading days
        num_days = len(self._cum_strategy_returns)
        strat_ann_return = (final_strat_return) ** (252 / num_days) - 1
        EWP_ann_return = (final_EWP_return) ** (252 / num_days) - 1
        index_ann_return = (final_index_return) ** (252 / num_days) - 1
        
        
        # Compute maximum drawdown
        strat_drawdown = (self._cum_strategy_returns / self._cum_strategy_returns.cummax()) - 1
        EWP_drawdown = (self._cum_EWP_returns / self._cum_EWP_returns.cummax()) - 1
        index_drawdown = (self._cum_index_returns / self._cum_index_returns.cummax()) - 1
        
        strat_max_drawdown = strat_drawdown.min()
        EWP_max_drawdown = EWP_drawdown.min()
        index_max_drawdown = index_drawdown.min()
        
        # Compute sharpe ratio (annualized)
        rf_daily = 0.0
        strat_excess_daily = self._daily_strategy_returns - rf_daily
        sharpe = float('nan')
        std_ret = self._daily_strategy_returns.std(ddof=1)
        if not math.isclose(std_ret, 0.0):
            sharpe = (strat_excess_daily.mean() / std_ret) * (252 ** 0.5)

        # Compute sortino ratio (annualized)
        sortino = float('nan')
        std_down = self._daily_strategy_returns[self._daily_strategy_returns < 0].std(ddof=1)
        if not math.isclose(std_down, 0.0):
            sortino = (strat_excess_daily.mean() / std_down) * (252 ** 0.5)
        
        # Information Ratio (vs. index, annualized): mean(active)/std(active) * sqrt(252)
        active = self._daily_strategy_returns - self._daily_index_returns
        info_ratio = float('nan')
        if active.std(ddof=1) != 0:
            info_ratio = (active.mean() / active.std(ddof=1)) * (252 ** 0.5)

        metrics = {
            "Total Strategy Returns": final_strat_return,
            "Annualized Strategy Returns": strat_ann_return,
            "Max Strategy Drawdown": strat_max_drawdown,
            "EWP Total Return": final_EWP_return,
            "EWP Annualized Return": EWP_ann_return,
            "EWP Max Drawdown": EWP_max_drawdown,
            "Index Total Return": final_index_return,
            "Index Annualized Return": index_ann_return,
            "Index Max Drawdown": index_max_drawdown,
            "Sharpe Ratio (ann, rf=0)": sharpe,
            "Sortino Ratio (ann, rf=0)": sortino,
            "Information Ratio (vs Index, ann)": info_ratio,
        }
        
        casted_metrics = {k: int(v) if v.is_integer() else float(v) for k, v in metrics.items()}
        
        return pformat(casted_metrics, sort_dicts=False)
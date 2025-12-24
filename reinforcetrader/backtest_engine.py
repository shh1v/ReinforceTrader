import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset
from collections import defaultdict

# Local imports for class dependencies
from .dqn_agent import DRLAgent
from .state import EpisodeStateLoader
from .data_pipeline import RawDataLoader


class EDBacktester:
    # Define ticker action constants (for readability)
    TA_REQUEST_BUY = 1 # Request to buy a stock
    TA_EXECUTED_BUY = 2 # buy stock successful
    TA_REJECTED_BUY = 3 # buy stock unsuccessful
    TA_EXECUTE_SELL = 4 # Execute a sell order
    TA_HOLD_IN = 5 # Hold position (in trade)
    TA_HOLD_OUT = 6 # Hold position (out of trade)
    
    def __init__(self, agent: DRLAgent, state_loader: EpisodeStateLoader, index: str, cash, tc: float=0.25, max_pos:int=5) -> None:
        self._agent = agent
        self._state_loader = state_loader

        if index == 'SP500':
            self._index_ticker = '^SPX'
        elif index == 'DJI':
            self._index_ticker = '^DJI'
        else:
            raise ValueError('Benchmark must be either "SP500" or "DJI".')
        
        # Set portfolio constraints
        self._cash_balance = cash
        self._trans_cost = tc # NOTE: Transaction cost in percentage (e.g., 0.25 for 0.25%)
        self._max_pos = max_pos
        self._pos_size_limit = self._cash_balance / self._max_pos
        
        # Per ticker state tracking
        self._tickers = self._state_loader.get_all_tickers()
        self._current_positions = {ticker: self._agent.OUT_TRADE for ticker in self._tickers}
        self._entry_prices = {ticker: 0.0 for ticker in self._tickers}
        self._shares_held = {ticker: 0 for ticker in self._tickers}
        
        # Container for reward params used in state represention
        self._agent_reward_states = {}
        
        # Strategy portfolio logging
        self._portfolio_history = []
        self._trade_logs = []
        
        # Benchmark portfolio tracking
        self._universe_prices = None
        self._benchmark_prices = None
        
        # Track weather backtest has been run
        self._backtest_ran = False

    def _init_agent_memory(self, t0) -> None:
        # NOTE: t=t0 is exclusive of the data used to init reward params
        for ticker in self._tickers:
            # Get historical returns from unused data of window for hot start
            Rts = [self._state_loader.get_reward_computes('test', 0, ticker, i)['1DFRet'] 
                   for i in range(t0)]
            # Init reward params and store for ticker
            # Bloated way of doing this, but avoids changing agent interface
            self._agent._init_reward_params(Rts)
            self._agent_reward_states[ticker] = self._agent._get_reward_computes()
    
    def run_backtest(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Get test window parameters
        L = self._state_loader.get_episode_len('test', 0)
        t0 = self._agent._window_size - 1
        dates = self._state_loader.get_test_dates(0)
        
        # Initialize agent reward parameters
        self._init_agent_memory(t0)
        
        # Collect price data for benchmarking later
        price_data = defaultdict(list[float])
        
        print(f'Starting Event-Driven Backtest. Initial Cash Balance: {self._cash_balance:.2f}; Max Positions: {self._max_pos}')
        
        for t in tqdm(range(t0, L-1), desc='Trading Days'):
            curr_date = dates[t]
            
            # For each timestep, iterate through all tickers
            buy_requests = []
            ticker_action = {}
            
            for ticker in self._tickers:
                # Capture current prices for benchmark calculation
                curr_price = self._state_loader.get_state_OHLCV('test', 0, ticker, t)['Close']
                price_data[ticker].append(curr_price)
                
                # Get current state and position
                curr_state = self._state_loader.get_state_matrix('test', 0, ticker, t, self._agent._window_size)
                prev_pos = self._current_positions[ticker]
                curr_ef = list(self._agent_reward_states[ticker].values())
                
                action, q_value = self._agent._act(curr_state, prev_pos, curr_ef, training=False)
                
                if prev_pos == self._agent.OUT_TRADE and action == self._agent.A_BUY:
                    buy_requests.append({'ticker': ticker, 'price': curr_price, 'q_value': q_value})
                    ticker_action[ticker] = self.TA_REQUEST_BUY
                elif prev_pos == self._agent.IN_TRADE and action == self._agent.A_SELL:
                    self._execute_sell(ticker, curr_price, curr_date)
                    ticker_action[ticker] = self.TA_EXECUTE_SELL
                else:
                    ticker_action[ticker] = self.TA_HOLD_IN if prev_pos == self._agent.IN_TRADE else self.TA_HOLD_OUT
            
            # Process buy requests based on available cash and max positions
            # First, sort requests based on confidence (q value)
            buy_requests.sort(key=lambda x: x['q_value'], reverse=True)
            for req in buy_requests:
                # NOTE: Relies on assumption that IN_TRADE/OUT_TRADE being 1/0
                active_pos_count = sum(self._current_positions.values())
                if min(self._cash_balance, self._pos_size_limit) >= req['price'] and active_pos_count < self._max_pos:
                    self._execute_buy(req['ticker'], req['price'], curr_date)
                    ticker_action[req['ticker']] = self.TA_EXECUTED_BUY
                else:
                    ticker_action[req['ticker']] = self.TA_REJECTED_BUY
            
            # Update agent state and it's reward state (params)
            for ticker in self._tickers:
                outcome = ticker_action[ticker]
                curr_pos = self._agent.IN_TRADE if outcome in {self.TA_EXECUTED_BUY, self.TA_HOLD_IN} else self._agent.OUT_TRADE
                if curr_pos == self._agent.IN_TRADE:
                    Rt = self._state_loader.get_reward_computes('test', 0, ticker, t)['1DFRet']
                else:
                    Rt = 0.0  # No return when out of trade
                
                # Load the reward params back to agent and update
                # WARNING: Could introduce floating precision issues over long runs
                # but, betting on the fact that they are negligible for practical purposes
                self._agent._set_reward_computes(self._agent_reward_states[ticker])
                self._agent_reward_states[ticker] = self._agent._update_reward_computes(Rt)
            
            # Compute portfolio value at end of day
            portfolio_value = self._cash_balance
            for p_ticker, num_shares in self._shares_held.items():
                if num_shares > 0:
                    asset_price = self._state_loader.get_state_OHLCV('test', 0, p_ticker, t)['Close']
                    portfolio_value += num_shares * asset_price
            
            # Log portfolio state into history
            self._portfolio_history.append({
                'date': curr_date,
                'portfolio_value': portfolio_value,
                'cash_balance': self._cash_balance,
                'num_positions': sum(self._current_positions.values())
            })
        
        # Prepare dataframes for portfolio history and trade logs
        self.portfolio_history_df = pd.DataFrame(self._portfolio_history).set_index('date')
        self.trade_logs_df = pd.DataFrame(self._trade_logs)
        
        # Build Universe Prices DF for EWP calculation
        # Note: Truncate dates to account for hot start timestep jump
        self.universe_prices = pd.DataFrame(price_data, index=self.portfolio_history_df.index)
        
        # Set backtest ran flag
        self._backtest_ran = True
        
        return self.portfolio_history_df, self.trade_logs_df
                    
    def _execute_buy(self, ticker, price, date):
        shares = min(self._pos_size_limit, self._cash_balance) // price
        if shares <= 0:
            raise ValueError(f'Insufficient cash or position limit to buy shares of {ticker} at price {price:.2f}')
        
        # Calculate transaction cost, new cash balance, and update holdings
        trans_cost_dollars = shares * price * self._trans_cost / 100
        trade_cost = (shares * price) + trans_cost_dollars
        self._cash_balance -= trade_cost
        self._shares_held[ticker] = shares
        self._entry_prices[ticker] = price
        self._current_positions[ticker] = self._agent.IN_TRADE
        
        # Append the transaction to trade logs
        self._trade_logs.append({
            'date': date,
            'ticker': ticker,
            'action': 'BUY',
            'price': price,
            'shares': shares,
            'trade_cost': trade_cost
        })
        
    def _execute_sell(self, ticker, price, date):
        shares = self._shares_held[ticker]
        entry_price = self._entry_prices[ticker]
        if shares <= 0:
            raise ValueError(f'No shares held to sell for ticker {ticker}.')
        trade_gross_val = shares * price
        trans_cost_dollars = trade_gross_val * self._trans_cost / 100
        buy_cost = shares * entry_price * (1 + self._trans_cost) / 100
        profit = trade_gross_val - trans_cost_dollars - buy_cost
        self._cash_balance += trade_gross_val - trans_cost_dollars
        self._shares_held[ticker] = 0
        self._current_positions[ticker] = self._agent.OUT_TRADE

        self._trade_logs.append({
            'date': date,
            'ticker': ticker,
            'action': 'SELL',
            'price': price,
            'shares': shares,
            'trade_proceeds': trade_gross_val - trans_cost_dollars,
            'profit': profit
        })
    
    def compute_performance_stats(self) -> pd.DataFrame:
        # Computes strategy perfomance and other benchmark (index, EWP)
        
        if not self._backtest_ran:
            raise RuntimeError('Backtest must be run before computing performance statistics.')
        
        # Compute the strategy returns
        strat_curve = self.portfolio_history_df['portfolio_value']
        strat_returns = strat_curve.pct_change().fillna(0.0)
        
        # Get index data for comparison
        start_date = strat_curve.index[0] - DateOffset(days=5)
        end_date = strat_curve.index[-1]
        index_data_loader = RawDataLoader(start_date=start_date, end_date=end_date, tickers=[self._index_ticker], verbose=False)
        index_df, _ = index_data_loader.get_hist_prices()
        index_curve = index_df['Close'].reindex(strat_curve.index)
        index_returns = index_curve.pct_change().fillna(0.0)
        
        # Compute Equal-Weighted Portfolio (EWP) returns
        ewp_returns = self.universe_prices.pct_change().mean(axis=1)
        ewp_curve = (1 + ewp_returns).cumprod() * self._cash_balance
        
        def get_curve_metrics(name, prices, returns):
            total_returns = (prices.iloc[-1] / prices.iloc[0]) - 1
            days = len(prices)
            ann_returns = (1 + total_returns) ** (252 / days) - 1
            
            # Compute the maximum drawdown
            rolling_max = prices.cummax()
            drawdown = (prices / rolling_max) - 1
            max_dd = drawdown.min()
            
            # Compute the Sharpe Ratio
            sigma = returns.std() * np.sqrt(252)
            sharpe = (returns.mean() * 252) / sigma
            
            # Compute the Sortino Ratio
            downside_returns = returns[returns < 0]
            down_sigma = downside_returns.std() * np.sqrt(252)
            sortino = (returns.mean() * 252) / down_sigma
            
            return {
                'Strategy': name,
                'Total Return': f"{total_returns*100:.2f}%",
                'Ann. Return': f"{ann_returns*100:.2f}%",
                'Max Drawdown': f"{max_dd*100:.2f}%",
                'Sharpe': round(sharpe, 3),
                'Sortino': round(sortino, 3)
            }
        
        # Compile stats for DRL strategy, and other benchmarks
        perf_stats = []
        perf_stats.append(get_curve_metrics('DRL Strategy', strat_curve, strat_returns))
        perf_stats.append(get_curve_metrics('Index (Buy and Hold)', index_curve, index_returns))
        perf_stats.append(get_curve_metrics('EWP (Daily; No Trans Cost)', ewp_curve, ewp_returns))
        
        self.perf_stats_df = pd.DataFrame(perf_stats).set_index('Strategy')
        
        # Store curves for plotting
        self.curves = pd.DataFrame({
            'DRL Strategy': strat_curve,
            'Index': index_curve,
            'EWP': ewp_curve
        })
        
        return self.perf_stats_df
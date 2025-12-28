import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset

# Local imports for class dependencies
from .dqn_agent import DRLAgent
from .state import EpisodeStateLoader
from .data_pipeline import RawDataLoader, FeatureBuilder


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
        self._initial_cash = cash
        self._cash_balance = cash
        self._trans_cost = tc # NOTE: Transaction cost in percentage (e.g., 0.25 for 0.25%)
        self._max_pos = max_pos
        self._pos_size_limit = self._cash_balance / self._max_pos
        
        # Per ticker state/reward tracking
        self._tickers = self._state_loader.get_all_tickers()
        self._entry_prices = {ticker: 0.0 for ticker in self._tickers}
        self._shares_held = {ticker: 0 for ticker in self._tickers}
        self._current_positions: dict[int, dict[str, int]] = {}
        self._agent_reward_states: dict[int, dict[str, dict[str, float]]] = {}
        
        # Strategy portfolio logging
        self._portfolio_history = []
        self._trade_logs = []
        
        # Benchmark portfolio tracking
        self._universe_prices = None
        self._benchmark_prices = None
        
        # Track weather backtest has been run
        self.ran_backtest = False

    def _init_agent_memory(self, t0) -> None:
        # Initialize agent trade position
        self._current_positions[t0] = {ticker: self._agent.OUT_TRADE for ticker in self._tickers}
        
        # NOTE: t=t0 is exclusive of the data used to init reward params
        self._agent_reward_states[t0] = {}
        for ticker in self._tickers:
            # Get historical returns from unused data of window for hot start
            Rts = [self._state_loader.get_reward_computes('test', 0, ticker, i)['1DFRet'] 
                   for i in range(t0)]
            # Init reward params and store for ticker
            # Bloated way of doing this, but avoids changing agent interface
            self._agent._init_reward_params(Rts)
            self._agent_reward_states[t0][ticker] = self._agent._get_reward_computes()
    
    def run_backtest(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Get test window parameters
        L = self._state_loader.get_episode_len('test', 0)
        t0 = self._agent._window_size - 1
        self._test_dates = self._state_loader.get_test_dates(0)
        
        # Initialize agent reward parameters
        self._init_agent_memory(t0)
        
        # Collect price data for benchmarking later
        price_data = {}
        
        print(f'Starting Event-Driven Backtest. Initial Cash Balance: {self._cash_balance:.2f}; Max Positions: {self._max_pos}')
        
        for t in tqdm(range(t0, L-1), desc='Trading Days'):
            curr_date = self._test_dates[t]
            
            # For each timestep, iterate through all tickers
            buy_requests = []
            ticker_action = {}
            
            # Create trade pos and reward state container for next timestep
            self._current_positions[t+1] = {}
            self._agent_reward_states[t+1] = {}
            
            for ticker in self._tickers:
                # Capture current prices for benchmark calculation
                curr_price = self._state_loader.get_state_OHLCV('test', 0, ticker, t)['Close']
                if ticker not in price_data:
                    price_data[ticker] = []
                price_data[ticker].append(curr_price)
                
                # Get current state and position
                curr_state = self._state_loader.get_state_matrix('test', 0, ticker, t, self._agent._window_size)
                prev_pos = self._current_positions[t][ticker]
                curr_ef = list(self._agent_reward_states[t][ticker].values())
                
                action, q_value = self._agent._act(curr_state, prev_pos, curr_ef, training=False)
                
                if prev_pos == self._agent.OUT_TRADE and action == self._agent.A_BUY:
                    buy_requests.append({'ticker': ticker, 'price': curr_price, 'q_value': q_value})
                    ticker_action[ticker] = self.TA_REQUEST_BUY
                elif prev_pos == self._agent.IN_TRADE and action == self._agent.A_SELL:
                    if not self._execute_sell(ticker, curr_price, t):
                        raise RuntimeError(f'Sell execution failed for ticker {ticker} on date {curr_date}.')
                    ticker_action[ticker] = self.TA_EXECUTE_SELL
                else:
                    ticker_action[ticker] = self.TA_HOLD_IN if prev_pos == self._agent.IN_TRADE else self.TA_HOLD_OUT
            
            # Process buy requests based on available cash and max positions
            # First, sort requests based on confidence (q value)
            buy_requests.sort(key=lambda x: x['q_value'], reverse=True)
            for req in buy_requests:
                # NOTE: Relies on assumption that IN_TRADE/OUT_TRADE being 1/0
                active_pos_count = sum(self._current_positions[t].values())
                
                success = False
                if active_pos_count < self._max_pos:
                    success = self._execute_buy(req['ticker'], req['price'], t) 
                    
                if success:
                    ticker_action[req['ticker']] = self.TA_EXECUTED_BUY
                else:
                    ticker_action[req['ticker']] = self.TA_REJECTED_BUY
            
            # Update agent state and it's reward state (params)
            for ticker in self._tickers:
                outcome = ticker_action[ticker]
                curr_pos = self._agent.IN_TRADE if outcome in {self.TA_EXECUTED_BUY, self.TA_HOLD_IN} else self._agent.OUT_TRADE
                if curr_pos == self._agent.IN_TRADE:
                    fRt = self._state_loader.get_reward_computes('test', 0, ticker, t)['1DFRet']
                else:
                    fRt = 0.0  # No return when out of trade
                
                # Load the reward params back to agent and update
                # WARNING: Could introduce floating precision issues over long runs
                # but, betting on the fact that they are negligible for practical purposes
                self._agent._set_reward_computes(self._agent_reward_states[t][ticker])
                self._agent_reward_states[t+1][ticker] = self._agent._update_reward_computes(fRt)
                
                # Update current position for next timestep
                self._current_positions[t+1][ticker] = curr_pos
            
            # Compute portfolio value at end of day
            portfolio_value = self._cash_balance
            for p_ticker, num_shares in self._shares_held.items():
                if num_shares > 0:
                    asset_price = self._state_loader.get_state_OHLCV('test', 0, p_ticker, t)['Close']
                    portfolio_value += num_shares * asset_price
            
            # Recacalculate position size limit based on updated cash balance
            self._pos_size_limit = portfolio_value / self._max_pos
            
            # Log portfolio state into history
            self._portfolio_history.append({
                'date': curr_date,
                'portfolio_value': portfolio_value,
                'cash_balance': self._cash_balance,
                'num_positions': sum(self._current_positions[t].values())
            })
        
        # Prepare dataframes for portfolio history and trade logs
        self.portfolio_history_df = pd.DataFrame(self._portfolio_history).set_index('date')
        self.trade_logs_df = pd.DataFrame(self._trade_logs)
        
        # Build Universe Prices DF for EWP calculation
        # Note: Truncate dates to account for hot start timestep jump
        self.universe_prices = pd.DataFrame(price_data, index=self.portfolio_history_df.index)
        
        # Set backtest ran flag
        self.ran_backtest = True
        
        return self.portfolio_history_df, self.trade_logs_df
                    
    def _execute_buy(self, ticker: str, price: float, t: int) -> bool:
        # Compute the effective price including transaction cost
        eff_price = price * (1 + (self._trans_cost / 100))
        shares = min(self._pos_size_limit, self._cash_balance) // eff_price
        if shares <= 0:
            return False  # Not enough cash to buy any shares
        
        # Calculate transaction cost, new cash balance, and update holdings
        trade_cost = shares * eff_price
        self._cash_balance -= trade_cost
        self._shares_held[ticker] = shares
        self._entry_prices[ticker] = price
        
        # Append the transaction to trade logs
        self._trade_logs.append({
            'date': self._test_dates[t],
            'ticker': ticker,
            'action': 'BUY',
            'price': price,
            'shares': shares,
            'cash_flow': -trade_cost
        })
        
        return True
        
    def _execute_sell(self, ticker: str, price: float, t: int) -> bool:
        shares = self._shares_held[ticker]
        entry_price = self._entry_prices[ticker]
        if shares <= 0:
            return False # Should not happen, but in case of logic error
        
        trade_gross_val = shares * price
        trans_cost_dollars = trade_gross_val * self._trans_cost / 100
        buy_cost = shares * entry_price * (1 + (self._trans_cost/100))
        profit = trade_gross_val - trans_cost_dollars - buy_cost
        self._cash_balance += trade_gross_val - trans_cost_dollars
        self._shares_held[ticker] = 0

        self._trade_logs.append({
            'date': self._test_dates[t],
            'ticker': ticker,
            'action': 'SELL',
            'price': price,
            'shares': shares,
            'cash_flow': trade_gross_val - trans_cost_dollars,
            'net_profit': profit
        })
        
        return True
    
    def compute_performance_stats(self) -> pd.DataFrame:
        # Computes strategy perfomance and other benchmark (index, EWP)
        
        if not self.ran_backtest:
            raise RuntimeError('Backtest must be run before computing performance statistics.')
        
        # Compute the strategy returns
        strat_curve = self.portfolio_history_df['portfolio_value']
        strat_returns = strat_curve.pct_change().fillna(0.0)
        
        # Get index data for comparison
        start_date = strat_curve.index[0] - DateOffset(days=5)
        end_date = strat_curve.index[-1]
        index_data_loader = RawDataLoader(start_date=start_date, end_date=end_date, tickers=[self._index_ticker], verbose=False)
        index_df, _ = index_data_loader.get_hist_prices()
        index_close = index_df['Close'].reindex(strat_curve.index)
        index_returns = index_close.pct_change().fillna(0.0)
        index_curve = (1 + index_returns).cumprod() * self._initial_cash
        
        # Compute Equal-Weighted Portfolio (EWP) returns
        ewp_returns = self.universe_prices.pct_change().mean(axis=1).fillna(0.0)
        ewp_curve = (1 + ewp_returns).cumprod() * self._initial_cash
        
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
            'Index (Buy and Hold)': index_curve,
            'EWP (Daily; No Trans Cost)': ewp_curve
        })
        
        return self.perf_stats_df
    
    def get_trade_scenario(self, buy_date: pd.Timestamp, ticker: str) -> tuple[np.ndarray, np.ndarray]:
        # method to extract trade scenario for given trade
        # Returns the state matrix, reward computes, and plots buy sell action
        if not self.ran_backtest:
            raise RuntimeError('Backtest must be run before running trade scenrios.')
        
        trade_entry = self.trade_logs_df.loc[(self.trade_logs_df['date'] == buy_date) &
                                             (self.trade_logs_df['ticker'] == ticker) &
                                             (self.trade_logs_df['action'] == 'BUY')]
        
        trade_exits = self.trade_logs_df.loc[(self.trade_logs_df['date'] > buy_date) &
                                             (self.trade_logs_df['ticker'] == ticker) &
                                             (self.trade_logs_df['action'] == 'SELL')]
        
        # Some check for ensuring a long/close position was taken
        if trade_entry.empty:
            raise ValueError(f'BUY trade not found for ticker {ticker} at {buy_date}.')
        elif trade_exits.empty:
            print(f'No SELL trade found for ticker {ticker} after {buy_date}. Simulating without an exit.')
            exit = False
        else:
            trade_exit = trade_exits.iloc[0]
            exit = True
        
        # Calculate the timestep indices for buy date
        buy_t = self._test_dates.get_loc(buy_date)
        buy_t = buy_t.start if isinstance(buy_t, slice) else int(buy_t)
        # buy_state = self._state_loader.get_state_matrix('test', 0, ticker, buy_t, self._agent._window_size)
        # buy_reward_computes = self._agent_reward_states[buy_t][ticker]
        
        # If the agent closes the position, get sell state as well
        if exit:
            sell_date = pd.Timestamp(trade_exit['date'])
            sell_t = self._test_dates.get_loc(sell_date)
            sell_t = sell_t.start if isinstance(sell_t, slice) else int(sell_t)     
            # sell_state = self._state_loader.get_state_matrix('test', 0, ticker, sell_t, self._agent._window_size)   
            # sell_reward_computes = self._agent_reward_states[sell_t][ticker]

        # Compute the trade duration and batch sizes
        batch_size = sell_t - buy_t + 1 if exit else 1
        num_reward_pars = len(self._agent_reward_states[buy_t][ticker])
        
        # Prepare the ndarrays for batch states and rewards
        # states will be (days, window, features), rewards will be (days, reward values)
        states_batch = np.empty((batch_size, self._agent._window_size, len(FeatureBuilder.STATE_FEATURES)))
        reward_batch = np.empty((batch_size, num_reward_pars + 1)) # +1 for trade position
        
        for i in range(batch_size):
            curr_t = buy_t + i
            trade_post = self._current_positions[curr_t][ticker]
            rewards_params = list(self._agent_reward_states[curr_t][ticker].values())
            states_batch[i] = self._state_loader.get_state_matrix('test', 0, ticker, curr_t, self._agent._window_size)
            reward_batch[i] = np.concatenate(([trade_post], rewards_params))
        
        # Plot the price action with buy/sell markers
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Get price series for the ticker (add buffer days for context)
        plot_start_date = buy_date - pd.Timedelta(days=self._agent._window_size - 1)
        plot_end_date = sell_date + pd.Timedelta(days=10) if exit else buy_date + pd.Timedelta(days=30)
        price_series = self.universe_prices[ticker].loc[plot_start_date : plot_end_date]
        
        
        # Plot the price series and buy/sell markers
        plt.plot(price_series, label=f'{ticker} Price', color='black')
        buy_point = price_series.loc[price_series.index == buy_date]
        plt.plot(buy_point, marker='^', color='green', markersize=12, label='Agent Buy', linestyle='None')
        if exit:
            sell_point = price_series.loc[price_series.index == sell_date]
            plt.plot(sell_point, marker='v', color='red', markersize=12, label='Agent Sell', linestyle='None')
            
            # Also plot other trade metrics (like duration, profit, etc.)
            trade_profit = trade_exit['net_profit']
            trade_duration = (sell_date - buy_date).days # type: ignore
            stats_text = (f'Trade Duration: {trade_duration} days\n'
                          f'Net Profit: ${trade_profit:,.2f}')
            
            # Add Text Box onto the plot
            ax.text(0.02, 0.95, stats_text, 
                    transform=ax.transAxes, 
                    fontsize=8, 
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
            
        plt.title(f'Trade Scenario for {ticker}: Buy on {buy_date.date()}' + (f', Sell on {sell_date.date()}' if exit else ''))
        plt.xlabel('Date')
        ax.tick_params(axis='x', labelrotation=45)
        plt.ylabel('Price ($)')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        
        return states_batch, reward_batch
    
    def get_random_test_samples(self, num_states: int=100) -> tuple[np.ndarray, np.ndarray]:
        # Get random states from the backtest period for analysis
        if not self.ran_backtest:
            raise RuntimeError('Backtest must be run before extracting random states.')
        
        # Prepare container for states
        t0 = self._agent._window_size - 1
        states_batch = np.empty((num_states, self._agent._window_size, len(FeatureBuilder.STATE_FEATURES)))
        rewards_batch = np.empty((num_states, 1 + len(self._agent_reward_states[t0][self._tickers[0]])))
        
        L = self._state_loader.get_episode_len('test', 0)
        t0 = self._agent._window_size - 1
        
        # Get num_states random indexes
        rand_t_indices = np.random.randint(t0, L-1, size=num_states)
        rand_tickers = np.random.choice(self._tickers, size=num_states)
        
        for i, t in enumerate(rand_t_indices):
            # Get the state and reward computes for the random ticker and timestep
            ticker = rand_tickers[i]
            
            # Get trade position
            trade_post = self._current_positions[t][ticker]
            reward_params = list(self._agent_reward_states[t][ticker].values())
            # Store in the batch containers
            states_batch[i] = self._state_loader.get_state_matrix('test', 0, ticker, t, self._agent._window_size)
            rewards_batch[i] = np.concatenate(([trade_post], reward_params))
        
        return states_batch, rewards_batch
    
    def plot_curves(self):
        if not hasattr(self, 'curves'):
            self.compute_performance_stats()
            
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # First plot: Equity Curves
        self.curves.plot(ax=ax1, linewidth=2)
        ax1.set_title(f'Equity Curve Comparison (Init. Capital: ${self._initial_cash:,.0f})')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.set_xlabel('Date')
        ax1.grid(True, alpha=0.3)
        
        # Second plot: Drawdowns
        strategy_dd = None
        global_max_dd = 0.0
        for col in self.curves.columns:
            series = self.curves[col]
            dd = (series / series.cummax()) - 1
            if col == 'DRL Strategy':
                strategy_dd = dd
            global_max_dd = min(global_max_dd, dd.min())
            ax2.plot(dd, label=col, linewidth=1.5)
            
        ax2.set_title('Drawdowns')
        ax2.set_ylabel('Drawdown (%)')
        ax2.fill_between(self.curves.index, 0, strategy_dd, color='red', alpha=0.05)
        ax2.set_ylim(global_max_dd * 1.05, 0)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
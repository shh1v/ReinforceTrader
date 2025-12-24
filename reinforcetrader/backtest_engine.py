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
    
    def __init__(self, agent: DRLAgent, state_loader: EpisodeStateLoader, benchmark: str, cash_balance: float=100_000, max_pos:int=5):
        self._agent = agent
        self._state_loader = state_loader

        if benchmark == 'SP500':
            self._benchmark_ticker = '^SPX'
        elif benchmark == 'DJI':
            self._benchmark_ticker = '^DJI'
        else:
            raise ValueError('Benchmark must be either "SP500" or "DJI".')
        
        # Set portfolio constraints
        self._cash_balance = cash_balance
        self._max_pos = max_pos
        
        # Per ticker state tracking
        self.tickers = self._state_loader.get_all_tickers()
        self.current_positions = {ticker: self._agent.OUT_TRADE for ticker in self.tickers}
        self.entry_prices = {ticker: 0.0 for ticker in self.tickers}
        self.shares_held = {ticker: 0 for ticker in self.tickers}
        
        # Container for reward params used in state represention
        self._agent_reward_states = {}
        
        # Strategy portfolio logging
        self._portfolio_history = []
        self._trade_logs = []
        
        # Benchmark portfolio tracking
        self._universe_prices = None
        self._benchmark_prices = None

    def _init_agent_memory(self, t0):
        # NOTE: t=t0 is exclusive of the data used to init reward params
        for ticker in self.tickers:
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
            
            for ticker in self.tickers:
                # Capture current prices for benchmark calculation
                curr_price = self._state_loader.get_state_OHLCV('test', 0, ticker, t)['Close']
                price_data[ticker].append(curr_price)
                
                # Get current state and position
                curr_state = self._state_loader.get_state_matrix('test', 0, ticker, t, self._agent._window_size)
                prev_pos = self.current_positions[ticker]
                curr_ef = list(self._agent_reward_states[ticker].values())
                
                action, q_value = self._agent._act(curr_state, prev_pos, curr_ef, training=False)
                
                if prev_pos == self._agent.OUT_TRADE and action == self._agent.A_BUY:
                    buy_requests.append({'ticker': ticker, 'price': curr_price, 'q_value': q_value})
                    ticker_action[ticker] = self.TA_REQUEST_BUY
                elif prev_pos == self._agent.IN_TRADE and action == self._agent.A_SELL:
                    self.execute_sell(ticker, curr_price, t, curr_date)
                    ticker_action[ticker] = self.TA_EXECUTE_SELL
                else:
                    ticker_action[ticker] = self.TA_HOLD_IN if prev_pos == self._agent.IN_TRADE else self.TA_HOLD_OUT
            
            # Process buy requests based on available cash and max positions
            # First, sort requests based on confidence (q value)
            buy_requests.sort(key=lambda x: x['q_value'], reverse=True)
            for req in buy_requests:
                # NOTE: Relies on assumption that IN_TRADE/OUT_TRADE being 1/0
                active_pos_count = sum(self.current_positions.values())
                if self._cash_balance >= req['price'] and active_pos_count < self._max_pos:
                    self.execute_buy(req['ticker'], req['price'], t, curr_date)
                    ticker_action[req['ticker']] = self.TA_EXECUTED_BUY
                else:
                    ticker_action[req['ticker']] = self.TA_REJECTED_BUY
            
            # Update agent state and it's reward state (params)
            for ticker in self.tickers:
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
            for p_ticker, num_shares in self.shares_held.items():
                if num_shares > 0:
                    asset_price = self._state_loader.get_state_OHLCV('test', 0, p_ticker, t)['Close']
                    portfolio_value += num_shares * asset_price
            
            # Log portfolio state into history
            self._portfolio_history.append({
                'date': curr_date,
                'portfolio_value': portfolio_value,
                'cash_balance': self._cash_balance,
                'num_positions': sum(self.current_positions.values())
            })
        
        # Prepare dataframes for portfolio history and trade logs
        self.portfolio_history_df = pd.DataFrame(self._portfolio_history).set_index('date')
        self.trade_logs_df = pd.DataFrame(self._trade_logs)
        
        # Build Universe Prices DF for EWP calculation
        # Note: Truncate dates to account for hot start timestep jump
        self.universe_prices = pd.DataFrame(price_data, index=self.portfolio_history_df.index)
        
        return self.portfolio_history_df, self.trade_logs_df
                    
    def execute_buy(self, ticker, curr_price, t, current_date):
        pass
    def execute_sell(self, ticker, curr_price, t, current_date):
        pass
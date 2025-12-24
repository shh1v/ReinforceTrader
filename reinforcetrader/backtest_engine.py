import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset

# Local imports for class dependencies
from .dqn_agent import DRLAgent
from .state import EpisodeStateLoader
from .data_pipeline import RawDataLoader


class EDBacktester:
    # Define constants for in-trade and out-of-trade positions
    OUT_TRADE = 0
    IN_TRADE = 1
    
    # Define constants that represent agent behaviour
    A_BUY = 0
    A_HOLD = 1
    A_SELL = 2
    
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
        self.current_positions = {ticker: EDBacktester.OUT_TRADE for ticker in self.tickers}
        self.entry_prices = {ticker: 0.0 for ticker in self.tickers}
        self.shares_held = {ticker: 0 for ticker in self.tickers}
        
        # Container for reward params used in state represention
        self._agent_reward_params = {}
        
        # Strategy portfolio logging
        self._portfolio_history = []
        self._trade_logs = []
        
        # Benchmark portfolio tracking
        self._universe_prices = None
        self._benchmark_prices = None

    def _init_agent_memory(self, t0):
        for ticker in self.tickers:
            # Get historical returns from unused data of window for hot start
            Rts = [self._state_loader.get_reward_computes('test', 0, ticker, i)['1DFRet'] 
                   for i in range(t0)]
            # Init reward params and store for ticker
            # Bloated way of doing this, but avoids changing agent interface
            self._agent._init_reward_params(Rts)
            self._agent_reward_params[ticker] = self._agent._get_reward_computes()
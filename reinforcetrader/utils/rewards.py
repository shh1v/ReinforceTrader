import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Callable
from ..data_pipeline import RawDataLoader

def evaluate_reward_function(ticker: str, start_date: str, end_date: str,
                             reward_fn: Callable[[int, int, float, float], tuple[float, int]]) -> pd.DataFrame:
    # Note: The reward_fn function takes four parameters:
    # prev_pos (int): 1 if in trade, otherwise 0
    # action (int): 0 is hold, 1 is buy, 2 is sell
    # curr_price (float): current price of the asset
    # next_price (float): next day price of the asset
    # Note: A buy signal when prev_pos is 1 is equivlent to hold
    # A sell signal when prev_pos is 0 is equivlent to hold
    
    # Load the data from yahoo finance
    data_loader = RawDataLoader(start_date=start_date, end_date=end_date, tickers=[ticker])
    
    # Drop multilevel column as only one ticker is considered
    data = data_loader.get_hist_prices().droplevel(0, axis=1)[['Close']]
    
    # Compute the reward function values for each day
    # positions store the conditions as key and (prev_pos, action) as value
    conditions = {'InTrade[Buy/Hold]': (1, 0),'InTradeSell': (1, 2), 'NotInTrade[Sell/Hold]': (0, 0)}
    reward_values = {action: [] for action in conditions.keys()}
    
    for condition, value in conditions.items():
        prev_pos, action = value
        for i in range(0, len(data.index) - 1):
            curr_price = data.iloc[i]['Close']
            next_price = data.iloc[i+1]['Close']
            reward, _ = reward_fn(prev_pos, action, curr_price, next_price)
            reward_values[condition].append(reward)
    
    # Drop the last day as it doesn't have a next day price
    data.drop(data.index[-1], inplace=True)        
    
    # Plot the closing data for the ticker
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot the close price of the ticker
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Price')
    ax1.plot(data.index, data, linewidth=1.5, color='black', alpha=0.7, label=f'{ticker} Close')
    for x in data.index:
        ax1.axvline(x=x, color='gray', alpha=0.2)
    
    ax1.set_xticklabels(data.index, rotation=45)
    ax1.set_title(f'Reward Function Evaluation for {ticker}')
    
    # Plot the reward function values on a twin axis
    ax2 = ax1.twinx()
    
    # Pick a colormap (e.g., tab10, Set1, viridis, etc.)
    colors = plt.cm.tab20.colors

    for i, (condition, values) in enumerate(reward_values.items()):
        ax2.plot(data.index, values, color=colors[i % len(colors)], label=condition, alpha=0.7)
    
    ax2.set_ylabel('Reward Values')
        
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
        
    plt.show()
    
    # Create a DataFrame to store the reward function values
    reward_df = pd.DataFrame(reward_values)
    reward_df.index = data.index
    reward_df['Close'] = data['Close']
    reward_df['Next Close'] = data['Close'].shift(-1)
    
    return reward_df
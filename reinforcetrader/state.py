import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

class EpisodeStateLoader:
    def __init__(self, features_data: pd.DataFrame, feature_indices: dict[str, np.ndarray], WFV_config: dict[str, str | int]):
        # Store the features data and indices
        self._features_data = features_data
        self._feature_indices = feature_indices
        
        # Load the configuration for Walking Forward Validation (WFV)
        self._WFV_mode = str(WFV_config["mode"]).lower()
        if self._WFV_mode not in ['expanding', 'moving']:
            raise ValueError(f"Invalid WFV mode: {self._WFV_mode}. Must be 'expanding' or 'moving'.")
        self._train_start = pd.Timestamp(WFV_config["train_start"])
        self._train_end = pd.Timestamp(WFV_config["train_end"])
        self._test_start = pd.Timestamp(WFV_config["test_start"])
        self._test_end = pd.Timestamp(WFV_config["test_end"])
        self._train_window_size = int(WFV_config["train_window_size"])
        self._val_window_size = int(WFV_config["val_window_size"])
        if self._train_window_size <= 0 or self._val_window_size <= 0:
            raise ValueError("Train and validation window sizes must be positive integers.")

        # Store all tickers symbols for getters function
        self._ticker_symbols = self._features_data.columns.get_level_values('Ticker').unique()

        # Store the window indices for each episode in their respective dicts
        # keyed by (episode_id) -> (start_index, end_index)
        self._train_slices: dict[int, tuple[int, int]] = {}
        self._val_slices: dict[int, tuple[int, int] | None] = {}
        self._test_slices: dict[int, tuple[int, int]] = {}

        # Build the train, val, test episodes slices for the data
        self._build_episode_slices()

    def _date_to_int_idx(self, date: pd.Timestamp) -> int:
        idx = self._features_data.index.get_indexer([date])[0]
        if idx == -1:
            raise ValueError(f"Date {date} not found in features data index.")
        return int(idx)
    
    def _build_episode_slices(self):
        # First, build the train and validation indices slices for WFV (all indexes inclusive)
        # Note: len(train_episodes) != len(val_episodes) if there is not enough data at the end
 
        # Get the integer indices for the train and test periods
        train_start_idx = self._date_to_int_idx(self._train_start)
        train_end_idx = self._date_to_int_idx(self._train_end)
        test_start_idx = self._date_to_int_idx(self._test_start)
        test_end_idx = self._date_to_int_idx(self._test_end)
        
        if train_end_idx >= test_start_idx:
            raise ValueError("Training period overlaps with test period introducing data leakage.")
        
        # Define pointers for the train and validation windows
        ep = 0
        w_train_start = train_start_idx
        w_train_end = w_train_start + self._train_window_size - 1
        w_val_start, w_val_end = None, None
        
        # Build the train and validation indices slices
        while True:
            # Case A: Train window starts beyond available training data
            if w_train_start > train_end_idx:
                # No more training data can be formed
                break
            
            # Case B: Train window ends beyond available training data
            if w_train_end > train_end_idx:
                if ep == 0:
                    warnings.warn('Not enough data to form a single train + validation episode.')

                self._train_slices[ep] = (w_train_start, train_end_idx)
                self._val_slices[ep] = None
                break
            
            # Train window is valid, and validation window can be checked
            w_val_start = w_train_end + 1
            w_val_end = w_val_start + self._val_window_size - 1
            
            # Case C: Validation window starts beyond available training data
            if w_val_start > train_end_idx:
                self._train_slices[ep] = (w_train_start, w_train_end)
                self._val_slices[ep] = None
                break
            
            # Case D: Validation window ends beyond available training data
            if w_val_end > train_end_idx:
                self._train_slices[ep] = (w_train_start, w_train_end)
                self._val_slices[ep] = (w_val_start, train_end_idx)
                break
            
            # Case E: Both train and validation windows are valid
            self._train_slices[ep] = (w_train_start, w_train_end)
            self._val_slices[ep] = (w_val_start, w_val_end)
            
            # Move the windows forward based on WFV mode
            if self._WFV_mode == 'moving':
                w_train_start += self._val_window_size
            w_train_end += self._val_window_size
            
            # Increment the episode counter
            ep += 1
        
        # Build the test indices slices
        self._test_slices[0] = (test_start_idx, test_end_idx)

    def get_all_tickers(self) -> list:
        return list(self._ticker_symbols)
    
    def get_num_episodes(self, episode_type: str) -> int:
        match episode_type:
            case 'train':
                return len(self._train_slices)
            case 'validate':
                return len([v for v in self._val_slices.values() if v is not None])
            case 'test':
                return len(self._test_slices)
            case _:
                raise ValueError(f"Invalid episode type: {episode_type}")

    def get_episode_len(self, episode_type: str, episode_id: int) -> int:
        # Get the start and end indices for the specified episode
        dates_idx = self._get_episode_window(episode_type, episode_id)
        
        return dates_idx[1] - dates_idx[0] + 1
    
    def _get_episode_window(self, episode_type: str, episode_id: int) -> tuple[int, int]:
        # NOTE: the start and end indices are inclusive
        # Retrieve the start and end indices for the specified episode
        match episode_type:
            case 'train':
                dates_idx = self._train_slices.get(episode_id)
            case 'validate':
                dates_idx =  self._val_slices.get(episode_id)
            case 'test':
                dates_idx = self._test_slices.get(episode_id)
            case _:
                raise ValueError(f"Invalid episode type: {episode_type}")
            
        # Check if the episode exists
        if dates_idx is None:
            raise ValueError(f"Episode id {episode_id} for {episode_type} does not exist.")
        
        return dates_idx
            
    
    def get_state_matrix(self, episode_type: str, episode_id: int, ticker: str, end_index: int, window_size: int, pad_overflow: bool = True) -> np.ndarray:
        # Check if ticker is valid
        if ticker not in self._ticker_symbols:
            raise ValueError(f"Ticker {ticker} not found in features data.")
        
        # Get the ticker specific data
        ticker_data = self._features_data.xs(ticker, axis=1, level='Ticker')   
        
        # Get epsiode start and end indices
        episode_start, episode_end = self._get_episode_window(episode_type, episode_id)
        
        # Check if end_index is valid
        episode_length = episode_end - episode_start + 1
        if not (0 <= end_index < episode_length):
            raise ValueError(f"end_index {end_index} out of range for episode length {episode_length}")
        
        # Compute start and index of the block (both inclusive)
        if pad_overflow:
            block_start_lb = self._date_to_int_idx(self._train_start)
        else:
            block_start_lb = episode_start
            
        block_start = episode_start + (end_index - window_size + 1)
        block_end = episode_start + end_index

        # Select the state matrix rows and the state features columns
        if block_start >= block_start_lb:
            state_row_idx = list(range(block_start, block_end + 1))
            data_window = ticker_data.iloc[block_start : block_end + 1].to_numpy(copy=False)
            
        else:
            pad_idx = [block_start_lb] * (block_start_lb - block_start)
            state_row_idx = pad_idx + list(range(block_start_lb, block_end + 1))        
            data_window = ticker_data.iloc[pd.Index(state_row_idx)].to_numpy(copy=False)
            
        state_matrix = data_window[:, self._feature_indices['State']]
        
        return state_matrix
    
    def get_state_OHLCV(self, episode_type: str, episode_id: int, ticker: str, index: int) -> dict[str, float]:
        # Get the respective feature data for the episode type
        episode_start, episode_end = self._get_episode_window(episode_type, episode_id)
        
        # Check if end_index is valid
        episode_length = episode_end - episode_start + 1
        if not (0 <= index < episode_length):
            raise ValueError(f"end_index {index} out of range for episode length {episode_length}")

        # Get the ticker specific row for the given index in the episode
        abs_index = episode_start + index
        ticker_row = self._features_data.xs(ticker, axis=1, level='Ticker').iloc[abs_index].to_numpy(copy=False)
        
        # Slice the OHLCV features specific columns and create dict
        OHLCV_params_idx = self._feature_indices['OHLCV']
        OHLCV_params = self._features_data.columns.get_level_values('Feature')[OHLCV_params_idx]
        OHLCV_values = ticker_row[OHLCV_params_idx].tolist()
        
        return {param: value for param, value in zip(OHLCV_params, OHLCV_values)}
    
    def get_reward_computes(self, episode_type: str, episode_id: int, ticker: str, index: int) -> dict[str, float]:
        # Get the reward parameters indices and parameter names
        reward_params_idx = self._feature_indices['Rewards']
        reward_params = self._features_data.columns.get_level_values('Feature')[reward_params_idx]
        
        # Get the respective feature data for the episode type
        episode_start, episode_end = self._get_episode_window(episode_type, episode_id)
        
        # Check if end_index is valid
        episode_length = episode_end - episode_start + 1
        if not (0 <= index < episode_length):
            raise ValueError(f"end_index {index} out of range for episode length {episode_length}")

        # Get the ticker specific row for the given index in the episode
        abs_index = episode_start + index
        ticker_row = self._features_data.xs(ticker, axis=1, level='Ticker').iloc[abs_index].to_numpy(copy=False)
        param_values =  ticker_row[self._feature_indices['Rewards']].tolist()
        
        return {param: value for param, value in zip(reward_params, param_values)}
    
    def get_test_dates(self, episode_id: int) -> pd.DatetimeIndex:
        # This is a utility function to get aligned date index for test episodes
        # Used for aligning the signals and prices dataframes during backtesting
        start_date, end_date = self._get_episode_window('test', episode_id)
        indices =  self._features_data.loc[start_date : end_date].index
        
        return pd.DatetimeIndex(pd.to_datetime(indices))

    def get_WFV_plot(self):
        dates = pd.DatetimeIndex(self._features_data.index)
        idx_to_date = lambda slice: (dates[slice[0]], dates[slice[1]])

        fig, ax = plt.subplots(figsize=(10, 0.6 * (len(self._train_slices) + 2)))
        colors = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green"}

        # plot train/val episodes
        for ep in sorted(self._train_slices.keys()):
            t0, t1 = idx_to_date(self._train_slices[ep])
            ax.barh(y=ep, left=t0, width=(t1 - t0).days, color=colors["train"])
            v = self._val_slices.get(ep)
            if v is not None:
                v0, v1 = idx_to_date(v)
                ax.barh(y=ep, left=v0, width=(v1 - v0).days, color=colors["val"])

        # plot test as last episode row
        test_ep = len(self._train_slices)
        t0, t1 = idx_to_date(self._test_slices[0])
        ax.barh(y=test_ep, left=t0, width=(t1 - t0).days, color=colors["test"])

        ax.set_yticks(range(test_ep + 1))
        ax.set_yticklabels([str(i) for i in range(test_ep)] + ["Test"])
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
        ax.set_xlabel("Date")
        ax.set_ylabel("Episode ID")
        ax.set_title("Walk-Forward Validation Layout")
        
        plt.show()

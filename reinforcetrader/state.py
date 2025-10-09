import numpy as np
import pandas as pd
import warnings

from pathlib import Path

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

    def get_episode_len(self, episode_type: str, episode_id: int, ticker: str) -> int:
        match episode_type:
            case 'train':
                return len(self._train_features[(episode_id, ticker)])
            case 'validate':
                return len(self._val_features[(episode_id, ticker)])
            case 'test':
                return len(self._test_features[(episode_id, ticker)])
            case _:
                raise ValueError(f"Invalid episode type: {episode_type}")
    
    def _get_features_set(self, episode_type: str) -> dict[tuple[int, str], np.ndarray]:
        match episode_type:
            case 'train':
                return self._train_features
            case 'validate':
                return self._val_features
            case 'test':
                return self._test_features
            case _:
                raise ValueError(f"Invalid episode type: {episode_type}")
    
    def get_state_matrix(self, episode_type: str, episode_id: int, ticker: str, end_index: int, window_size: int):
        # Get the respective feature data for the episode type
        store = self._get_features_set(episode_type)
        episode_ticker_features = store[(episode_id, ticker)]

        # Compute start index of the block
        start_index = end_index - window_size + 1

        # Slice features (skip first 5 columns)
        feats = episode_ticker_features[:, self._feature_indices['State']]

        if start_index >= 0:
            state_matrix = feats[start_index:end_index + 1, :]
        else:
            pad_len = -start_index
            pad_block = np.repeat(feats[[0], :], repeats=pad_len, axis=0)
            window_block = feats[:end_index + 1, :]
            state_matrix = np.vstack([pad_block, window_block])

        return state_matrix
    
    def get_state_OHLCV(self, episode_type: str, episode_id: int, ticker: str, index: int) -> np.ndarray:
        # Get the respective feature data for the episode type
        store = self._get_features_set(episode_type)
        episode_ticker_features = store[(episode_id, ticker)]

        # Slice the OHLCV features specific columns
        return episode_ticker_features[index, self._feature_indices['OHLCV']]
    
    def get_reward_computes(self, episode_type: str, episode_id: int, ticker: str, index: int) -> dict[str, float]:
        # Get the reward parameters indices and parameter names
        reward_comp_indices = self._feature_indices['Rewards']
        reward_comp_names = self._features_data.columns.get_level_values('Feature')[reward_comp_indices]
        
        # Get the respective feature data for the episode type
        store = self._get_features_set(episode_type)
        episode_ticker_features = store[(episode_id, ticker)]

        # Slice the features relevant to reward computation
        reward_comp_values = episode_ticker_features[index, reward_comp_indices]
        
        return {rn: rv for rn, rv in zip(reward_comp_names, reward_comp_values)}
    
    def get_test_dates(self, episode_id: int, ticker: str) -> pd.DatetimeIndex:
        # find the test episode config
        ep_cfg = next(
            ep for ep in self._episode_config["episodes"]
            if int(ep["episode_id"]) == int(episode_id) and ep["type"] == "test"
        )
        start = pd.Timestamp(ep_cfg["start_date"])
        end   = pd.Timestamp(ep_cfg["end_date"])

        # full date slice in the features DataFrame
        full_idx = self._features_data.loc[start:end].index

        # trim to stored length (in case of dropna/padding during feature build)
        L = self._test_features[(episode_id, ticker)].shape[0]
        
        if len(full_idx) < L:
            # fall back to last L rows from available
            return pd.DatetimeIndex(pd.to_datetime(full_idx[-L:]))
        
        return pd.DatetimeIndex(pd.to_datetime(full_idx[:L]))

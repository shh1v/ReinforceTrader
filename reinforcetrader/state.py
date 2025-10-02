import numpy as np
import pandas as pd
import json

from pathlib import Path

class EpisodeStateLoader:
    def __init__(self, features_data: pd.DataFrame, feature_indices: dict[str, np.ndarray], episode_config_path: str):
        self._features_data = features_data
        self._feature_indices = feature_indices
        self._episode_config = self._load_config(episode_config_path)

        # Store all tickers symbols for getters function
        self._ticker_symbols = self._features_data.columns.get_level_values('Ticker').unique()

        # Store the features in their respoective dicts
        # keyed by (episode_id, ticker) -> np.ndarray [len(ep), F]
        self._train_features: dict[tuple[int, str], np.ndarray] = {}
        self._val_features: dict[tuple[int, str], np.ndarray] = {}
        self._test_features: dict[tuple[int, str], np.ndarray] = {}

        # Build the episodes states for each ticker
        self._build_episode_data()

    def _load_config(self, config_path: str) -> dict:
        try:
            path = Path(config_path)
            if not path.exists():
                raise FileNotFoundError(f'Config file not found: {config_path}')
            if not path.is_file():
                raise ValueError(f'Config path is not a file: {config_path}')

            with path.open('r', encoding='utf-8') as f:
                config = json.load(f)

            return config

        except json.JSONDecodeError:
            raise ValueError(f'Invalid JSON format in {config_path}')
        except Exception as e:
            raise RuntimeError(f'Error loading config: {e}')

    def _build_episode_data(self):
        for ep in self._episode_config["episodes"]:
            # Check for valid episode types
            episode_type = ep["type"]
            if episode_type not in {"train", "test"}:
                raise ValueError(f"Invalid episode type: {episode_type}")

            episode_id = int(ep["episode_id"])
            episode_start_date = pd.Timestamp(ep["start_date"])
            episode_end_date = pd.Timestamp(ep["end_date"])

            # Slice for the episode once; reuse per ticker
            # We'll select per-ticker below with self._features_data.loc[start:end, ticker]
            for ticker in self._ticker_symbols:
                # (len(ep),F) for this ticker and episode window
                ticker_data = self._features_data.loc[episode_start_date : episode_end_date, ticker]

                # Should never be the case, but just for safety
                if ticker_data.empty:
                    continue

                ticker_values = ticker_data.to_numpy(dtype=np.float32)

                if episode_type == "train":
                    split_idx = int(len(ticker_data) * 0.8)
                    self._train_features[(episode_id, ticker)] = ticker_values[:split_idx]
                    self._val_features[(episode_id, ticker)] = ticker_values[split_idx:]
                else:
                    # No need for splitting dataset
                    self._test_features[(episode_id, ticker)] = ticker_values

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
    
    def _get_set_features(self, episode_type: str) -> dict[tuple[int, str], np.ndarray]:
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
        store = self._get_set_features(episode_type)
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
        store = self._get_set_features(episode_type)
        episode_ticker_features = store[(episode_id, ticker)]

        # Slice the OHLCV features specific columns
        return episode_ticker_features[index, self._feature_indices['OHLCV']]
    
    def get_reward_computes(self, episode_type: str, episode_id: int, ticker: str, index: int) -> dict[str, float]:
        # Get the reward parameters indices and parameter names
        reward_comp_indices = self._feature_indices['Rewards']
        reward_comp_names = self._features_data.columns.get_level_values('Feature')[reward_comp_indices]
        
        # Get the respective feature data for the episode type
        store = self._get_set_features(episode_type)
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

import numpy as np
import pandas as pd
import json

from pathlib import Path
from tqdm import tqdm

class EpisodeStateLoader:
    def __init__(self, features_data: pd.DataFrame, episode_config_path: str):
        self._features_data = features_data
        self._episode_config = self._load_config(episode_config_path)

        # Store all tickers symbols and their respective indexes
        ordered_ticker_symbols = self._features_data.columns.get_level_values('Ticker').unique()
        self._ticker_idx = {ticker: idx for idx, ticker in enumerate(ordered_ticker_symbols)}

        # Store the states in dicts keyed by (episode_id, ticker) -> np.ndarray [len(ep), F]
        self._train_states: dict[tuple[int, str], np.ndarray] = {}
        self._val_states: dict[tuple[int, str], np.ndarray] = {}
        self._test_states: dict[tuple[int, str], np.ndarray] = {}

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
        for ep in tqdm(self._episode_config["episodes"], desc='Building states for each episode', ncols=80):
            # Check for valid episode types
            episode_type = ep["type"]
            if episode_type not in {"training", "testing"}:
                raise ValueError(f"Invalid episode type: {episode_type}")

            episode_id = int(ep["episode_id"])
            episode_start_date = pd.Timestamp(ep["start_date"])
            episode_end_date = pd.Timestamp(ep["end_date"])

            # Slice for the episode once; reuse per ticker
            # We'll select per-ticker below with self._features_data.loc[start:end, ticker]
            for ticker in self._ticker_idx.keys():
                # (len(ep),F) for this ticker and episode window
                ticker_data = self._features_data.loc[episode_start_date:episode_end_date, ticker]

                if ticker_data.empty:
                    continue

                ticker_values = ticker_data.to_numpy(dtype=np.float32)

                if episode_type == "training":
                    split_idx = int(len(ticker_data) * 0.8)
                    self._train_states[(episode_id, ticker)] = ticker_values[:split_idx]
                    self._val_states[(episode_id, ticker)] = ticker_values[split_idx:]
                else:
                    self._test_states[(episode_id, ticker)] = ticker_values
        
        print('Build train, valid, test states complete!')
import numpy as np
import pandas as pd

from pathlib import Path

class EpisodeStateLoader:
    def __init__(self, features_data: pd.DataFrame, episode_config_path: str):
        self._features_data = features_data
        self.episode_config = self._load_config(episode_config_path)
    
    def _load_config(self, config_path: str) -> dict:
        try:
            path = Path(config_path)
            if not path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
            if not path.is_file():
                raise ValueError(f"Config path is not a file: {config_path}")

            with path.open("r", encoding="utf-8") as f:
                config = json.load(f)

            return config

        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON format in {config_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading config: {e}")
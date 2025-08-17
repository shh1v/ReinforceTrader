import numpy as np
import pandas as pd
import json
from pathlib import Path
import yfinance as yf


class RawDataLoader:
    def __init__(self, episode_config_pth, tickers=None):
        # Load the config file
        self.episode_config = self._load_config(episode_config_pth)

        # Fetch all the tickers if not provided
        if not tickers:
            tickers = self._fetch_tickers()

    
    def _load_config(self, config_path) -> dict:
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
            
    def _fetch_tickers(self):
        # Fetch all the tickers data from wikipedia
        ticker_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        
        # Get ticker names and exclude class B shares
        tickers = [ticker for ticker in ticker_table['Symbol'] if '.B' not in ticker]
    
        return tickers
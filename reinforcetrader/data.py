import numpy as np
import pandas as pd
import json
from pathlib import Path
import yfinance as yf


class RawDataLoader:
    def __init__(self, episode_config_pth):
        # Load the config file
        self.episode_config = self._load_config(episode_config_pth)

        # Fetch all the tickers in S&P 500
        tickers = self._fetch_tickers()

        # Load all the ticker price and volume data
        self.hist_data = self._load_hist_prices(tickers)

    
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

    def _download_hist_prices(self, tickers, start_date, end_date, save, save_path=None):
        # Download data from yfinance
        data = yf.download(tickers=tickers, start=start_date, end=end_date, auto_adjust=True)

        # Keep close prices and volumes
        data = data[['Close', 'Volume']]

        # Reorder multi-column index to ['Ticker', 'Price']
        data = data.reorder_levels(['Ticker', 'Price'], axis=1)

        # Save the data locally to reduce API calls
        if save:
            data.to_csv(save_path)
            print(f"Data saved to {save_path}")

        return data

    def _load_hist_prices(self, tickers, cache_path='data/raw'):
        # Extract start and end dates that include all episodes
        start_date = self.episode_config['notes']['start_date']
        end_date = self.episode_config['notes']['end_date']

        # Build cache directory and file path
        cache_dir = Path(cache_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        file_path = cache_dir / f"tickers_data_{start_date}_{end_date}.csv"

        # If cached file exists, load and return
        if file_path.exists():
            print(f"Loading cached data from {file_path}")
            return pd.read_csv(file_path, header=[0, 1], index_col=0, parse_dates=True)

        return self._download_hist_prices(tickers, start_date, end_date, save=True, save_path=file_path)

    def get_hist_prices(self):
        return self.hist_data
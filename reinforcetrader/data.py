import numpy as np
import pandas as pd

from pathlib import Path
import yfinance as yf


class RawDataLoader:
    def __init__(self, start_date: str, end_date: str):
        # Store the start and end dates of data to be downloaded/load from cache
        self._start_date = start_date
        self._end_date = end_date

        # Fetch all the tickers in S&P 500
        tickers = self._fetch_tickers()

        # Load all the ticker price and volume data
        self._hist_data = self._load_hist_prices(tickers)
            
    def _fetch_tickers(self) -> list:
        # Fetch the S&P 500 tickers list from wikipedia
        ticker_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
        
        # Get ticker names and exclude class B shares
        tickers = [ticker for ticker in ticker_table['Symbol'] if '.B' not in ticker]
    
        return tickers

    def _clean_hist_prices(self, data: pd.DataFrame) -> pd.DataFrame:
        # Try forward forward fill to fill any missing values in between
        data = data.ffill()

        # Find all tickers that don't have data from start_date
        tickers_with_nan = data.T.groupby(level=0).apply(lambda x: x.T.isna().any().any())
        tickers_to_drop = tickers_with_nan[tickers_with_nan].index

        # Drop all tickers that have NaN values
        data = data.drop(columns=tickers_to_drop, level=0)
        columns_left = data.columns.get_level_values('Ticker').nunique()
        print(f'Dropped {len(tickers_to_drop)} tickers. {columns_left} tickers left.')

        return data 


    def _download_hist_prices(self, tickers: list, save: bool, save_path=None) -> pd.DataFrame:
        # Download data from yfinance
        data = yf.download(tickers=tickers, start=self._start_date, end=self._end_date, auto_adjust=True)

        # Reorder multi-column index to ['Ticker', 'Price']
        data = data.reorder_levels(['Ticker', 'Price'], axis=1)

        # Perform data-cleaning
        data = self._clean_hist_prices(data)

        # Save the data locally to reduce API calls
        if save:
            data.to_csv(save_path)
            print(f"Data saved to {save_path}")

        return data

    def _load_hist_prices(self, tickers: list, cache_path: str='data/raw') -> pd.DataFrame:

        # Build cache directory and file path
        cache_dir = Path(cache_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        file_path = cache_dir / f"tickers_data_{self._start_date}_{self._end_date}.csv"

        # If cached file exists, load and return
        if file_path.exists():
            print(f"Loading cached data from {file_path}")
            return pd.read_csv(file_path, header=[0, 1], index_col=0, parse_dates=True)

        print(f'Downloading from yfinance as cached data does not exist in {cache_path}')
        return self._download_hist_prices(tickers, save=True, save_path=file_path)

    def get_hist_prices(self, tickers: list=[]):
        # Only return selected columns
        if tickers:
            return self._hist_data[tickers]
        
        return self._hist_data
import numpy as np
import pandas as pd
import yfinance as yf

from tqdm import tqdm
from pathlib import Path
from ta.trend import EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator


class RawDataLoader:
    def __init__(self, start_date: str, end_date: str, index: str='DJI'):
        # Store the start and end dates of data to be downloaded/load from cache
        self._start_date = start_date
        self._end_date = end_date
        self._index = index

        # Fetch all the tickers in S&P 500
        tickers = self._fetch_tickers(index=self._index)

        # Load all the ticker price and volume data
        self._hist_data = self._load_hist_prices(tickers)
            
    def _fetch_tickers(self, index: str='DJI') -> list:
        if index == 'DJI':
            table_link = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
            table_idx = 2
        elif index == 'SP500':
            table_link = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
            table_idx = 0
        else:
            raise ValueError('Invalid index: {index}')
        
        # Fetch the S&P 500 tickers list from wikipedia
        ticker_table = pd.read_html(table_link)[table_idx]

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
        file_path = cache_dir / f"{self._index}_tickers_data_{self._start_date}_{self._end_date}.csv"

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

class FeatureBuilder:
    def __init__(self, hist_prices, index='DJI'):
        self._hist_prices = hist_prices.sort_index()
        self._features_data = None
        self._index = index

    def _save_features_data(self, save_dir='data/processed') -> bool:
        if self._features_data is None or self._features_data.empty:
            print('Features data not built or empty')
            return False
        
        # Build save directory and file path
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        dates = self._features_data.index
        start_date = dates[0].strftime('%Y-%m-%d')
        end_date = dates[-1].strftime('%Y-%m-%d')
        file_path = save_dir / f"{self._index}_tickers_features_{start_date}_{end_date}.csv"

        # Check if feature file already exists
        if file_path.exists():
            print(f"File already exists, skipping save: {file_path}")
            return False

        # Save the features data
        self._features_data.to_csv(file_path)
        print(f"Features data saved to {file_path}")

        return True

    def build_features(self, save=True):
        # Get tickers symbols
        tickers = self._hist_prices.columns.get_level_values('Ticker').unique()

        # Predefine feature names
        OHLCV_f = ['Open', 'High', 'Low', 'Close', 'Volume']
        price_f = ['Body/HL', 'UShadow/HL', 'LShadow/HL']
        technical_f = ['C/EMA5', 'EMA5/EMA13', 'EMA13/EMA26', 'B%B', 'BBW', 'RSI', 'ADX', 'V/Vol20']

        # Predefine the feature dataframe
        feature_columns = pd.MultiIndex.from_product([tickers, OHLCV_f + price_f + technical_f],
                                                     names=['Ticker', 'Feature'])
        self._features_data = pd.DataFrame(index=self._hist_prices.index, columns=feature_columns, dtype=float)

        for ticker in tqdm(tickers, ncols=100, desc='Building ticker features'):
            open = self._hist_prices[ticker]['Open']
            high = self._hist_prices[ticker]['High']
            low = self._hist_prices[ticker]['Low']
            close = self._hist_prices[ticker]['Close']
            volume = self._hist_prices[ticker]['Volume']

            # Keep OHLCV as is (may not be part of state representation)
            self._features_data.loc[:, (ticker, 'Open')] = open
            self._features_data.loc[:, (ticker, 'High')] = high
            self._features_data.loc[:, (ticker, 'Low')] = low
            self._features_data.loc[:, (ticker, 'Close')] = close
            self._features_data.loc[:, (ticker, 'Volume')] = volume

            # Compute the candle body features
            # Body relative to total range, clip for stability
            candle_range = (high - low).replace(0, 1e-12)
            self._features_data.loc[:, (ticker, 'Body/HL')] = (
                (close - open) / candle_range
            ).clip(-1, 1)

            # Upper shadow relative to range
            self._features_data.loc[:, (ticker, 'UShadow/HL')] = (
                    (high - np.maximum(open, close)) / candle_range
            ).clip(lower=0)

            # Lower shadow relative to range
            self._features_data.loc[:, (ticker, 'LShadow/HL')] = (
                (np.minimum(open, close) - low) / candle_range
            ).clip(lower=0)

            # Compute rolling Exponential Moving Averages: 5, 13, 26
            ema5 = EMAIndicator(close=close, window=5).ema_indicator()
            ema13 = EMAIndicator(close=close, window=13).ema_indicator()
            ema26 = EMAIndicator(close=close, window=26).ema_indicator()

            # Compute the EMA ratios
            self._features_data.loc[:, (ticker, 'C/EMA5')] = (close / ema5) - 1
            self._features_data.loc[:, (ticker, 'EMA5/EMA13')] = (ema5 / ema13) - 1
            self._features_data.loc[:, (ticker, 'EMA13/EMA26')] = (ema13 / ema26) - 1

            # Compute Bollinger Bands (%B and Bandwidth)
            bb = BollingerBands(close, window=20, window_dev=2)
            bb_sma20 = bb.bollinger_mavg()
            bb_upper = bb.bollinger_hband()
            bb_lower = bb.bollinger_lband()
            bb_width = bb_upper - bb_lower
            self._features_data.loc[:, (ticker, 'B%B')] = ((close - bb_lower) / bb_width)
            self._features_data.loc[:, (ticker, 'BBW')] = bb_width / bb_sma20

            # Compute Relative Strength Index (RSI) scaled bw [-1, 1]
            rsi = RSIIndicator(close, window=14).rsi()
            self._features_data.loc[:, (ticker, 'RSI')] = (rsi - 50) / 50

            # Compute Average Directional Index scaled bw [0, 1]
            adx = ADXIndicator(high=high, low=low, close=close, window=14).adx()
            self._features_data.loc[:, (ticker, 'ADX')] = adx / 100.0

            # Compute ratio of current volume over 20 volume moving average
            vol20 = volume.rolling(20, min_periods=20).mean()
            self._features_data.loc[:, (ticker, 'V/Vol20')] = volume / vol20
        
        # Drop all rows with NaN values
        self._features_data = self._features_data.dropna()

        # Save the features data to use for later
        if save:
            self._save_features_data()

    def get_features(self) -> pd.DataFrame:
        if self._features_data is None:
            return pd.DataFrame()
        
        return self._features_data
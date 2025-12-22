import numpy as np
import pandas as pd
import yfinance as yf

from tqdm import tqdm
from pathlib import Path
from ta.trend import EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator


class RawDataLoader:
    def __init__(self, start_date: str, end_date: str, tickers=[], index: str='', verbose=True):
        # Store the start and end dates of data to be downloaded/load from cache
        self._start_date = start_date
        self._end_date = end_date
        self._index = index
        self._verbose = verbose

        # Make sure there is no conflict between tickers and index
        if tickers and index:
            raise ValueError('Tickers and index cannot be provided simultaneously.')
        elif not tickers and not index:
            raise ValueError('Either tickers or index must be provided.')
        
        # Fetch all the tickers in index if specific tickers are not provided
        if not tickers:
            # Prefer loading from cache to avoid API calls
            load_from_cache = True
            tickers = self._fetch_tickers(index=self._index)
            benchmark_ticker = '^DJI' if index == 'DJI' else '^SPX'
        else:
            # Only load from cache if tickers are fetched from index
            load_from_cache = False
            benchmark_ticker = None
            
            

        # Load all the ticker price and volume data
        self._ticker_data = self._load_hist_prices(tickers, load_from_cache=load_from_cache)
        if benchmark_ticker:
            self._benchmark_data = self._download_hist_prices([benchmark_ticker], save=False)
        else:
            self._benchmark_data = None
            
    def _fetch_tickers(self, index: str='DJI') -> list:
        if index == 'DJI':
            url = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
        elif index == 'SP500':
            url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        else:
            raise ValueError('Invalid index: {index}')
        
        # Fetch the Index tables from wikipedia (one of them should be list of tickers)
        headers = {"User-Agent": "Mozilla/5.0 (compatible; ReinforceTrader/1.0)"}
        tables = pd.read_html(url, storage_options=headers)
        
        # Find the table with a 'Symbol' column (which is usually the list of tickers)
        ticker_table = next((table for table in tables if 'Symbol' in table.columns), None)
        if ticker_table is None:
            raise ValueError('Couldn\'t find a table with a \'Symbol\' column on the page.')
        
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
        if self._verbose:
            print(f'Dropped {len(tickers_to_drop)} tickers. {columns_left} tickers left.')
            print(f"Tickers dropped: {', '.join(tickers_to_drop)}")
        
        # Drop the multi-column index if only one ticker is present
        if columns_left == 1:
            data.columns = data.columns.droplevel('Ticker')

        return data


    def _download_hist_prices(self, tickers: list, save: bool, save_path=None) -> pd.DataFrame:
        # Download data from yfinance
        data = yf.download(tickers=tickers, start=self._start_date, end=self._end_date, auto_adjust=True, progress=self._verbose, threads=True)

        # Reorder multi-column index to ['Ticker', 'Price']
        data = data.reorder_levels(['Ticker', 'Price'], axis=1)

        # Perform data-cleaning
        data = self._clean_hist_prices(data)

        # Save the data locally to reduce API calls
        if save:
            data.to_csv(save_path)
            if self._verbose:
                print(f"Data saved to {save_path}")

        return data

    def _load_hist_prices(self, tickers: list, load_from_cache, cache_path: str='../data/raw') -> pd.DataFrame:

        # Build cache directory and file path
        cache_dir = Path(cache_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        file_path = cache_dir / f"{self._index}_tickers_data_{self._start_date}_{self._end_date}.csv"

        # If cached file exists, load and return
        if load_from_cache and file_path.exists():
            if self._verbose:
                print(f"Loading cached data from {file_path}")
            return pd.read_csv(file_path, header=[0, 1], index_col=0, parse_dates=True)
        
        if self._verbose:
            print(f'Downloading from yfinance as cached data does not exist in {cache_path}')
        return self._download_hist_prices(tickers, save=load_from_cache, save_path=file_path)

    def get_hist_prices(self, tickers: list=[]):
        # Only return selected columns
        if tickers:
            return self._ticker_data[tickers], self._benchmark_data
        
        return self._ticker_data, self._benchmark_data

class FeatureBuilder:
    def __init__(self, ticker_data, benchmark_data: pd.DataFrame, f_prefix: str) -> None:
        if ticker_data.empty or benchmark_data.empty:
            raise ValueError('Ticker data and benchmark data cannot be empty.')
        
        self._ticker_data = ticker_data.sort_index()
        self._benchmark_data = benchmark_data.sort_index()
        
        self._f_prefix = f_prefix
        
        self._features_data = None
        self._feature_indices = None

    def _save_features_data(self, save_dir='../data/processed') -> bool:
        if self._features_data is None or self._features_data.empty:
            print('Features data not built or empty')
            return False
        
        # Build save directory and file path
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        dates = self._features_data.index
        start_date = dates[0].strftime('%Y-%m-%d')
        end_date = dates[-1].strftime('%Y-%m-%d')
        file_path = save_dir / f"{self._f_prefix}_tickers_features_{start_date}_{end_date}.csv"

        # Check if feature file already exists
        if file_path.exists():
            print(f"File already exists, skipping save: {file_path}")
            return False

        # Save the features data
        self._features_data.to_csv(file_path)
        print(f"Features data saved to {file_path}")

        return True

    
    def _norm(self, feature_data: pd.Series, window=126) -> pd.Series:
        # Compute rolling mean and std
        mean = feature_data.rolling(window=window, min_periods=window).mean()
        std = feature_data.rolling(window=window, min_periods=window).std()
        
        # Avoid division by zero by replacing with machine epsilon if zero
        std = std.mask(std < np.finfo(float).eps, np.finfo(float).eps)
        
        return (feature_data - mean) / std
    
    def build_features(self, save=True):
        # Get tickers symbols
        tickers = self._ticker_data.columns.get_level_values('Ticker').unique()

        # Predefine feature names
        # Include OHLCV portfolio managment and more
        OHLCV_f = ['Open', 'High', 'Low', 'Close', 'Volume']
        # Include the stock returns for the reward fn (e.g. DSR or DDDR)
        returns_f = [f'1DFRet']
        # Include all the DQN features used in state representations
        state_f = ['Body/HL', 'UWick/HL', 'LWick/HL', 'Gap', 'GapFill',
                        'EMA5/13', 'EMA13/26', 'EMA26/50', 'B%B', 'BBW',
                        'RSI', 'ADX', 'V/Vol20']

        # Predefine the feature dataframe
        feature_columns = pd.MultiIndex.from_product([tickers, OHLCV_f + returns_f + state_f],
                                                     names=['Ticker', 'Feature'])
        self._features_data = pd.DataFrame(index=self._ticker_data.index, columns=feature_columns, dtype=float)
        
        for ticker in tqdm(tickers, ncols=100, desc='Building ticker features'):
            o = self._ticker_data[ticker]['Open']
            h = self._ticker_data[ticker]['High']
            l = self._ticker_data[ticker]['Low']
            c = self._ticker_data[ticker]['Close']
            v = self._ticker_data[ticker]['Volume']

            # Keep OHLCV as is (may not be part of state representation)
            self._features_data.loc[:, (ticker, 'Open')] = o
            self._features_data.loc[:, (ticker, 'High')] = h
            self._features_data.loc[:, (ticker, 'Low')] = l
            self._features_data.loc[:, (ticker, 'Close')] = c
            self._features_data.loc[:, (ticker, 'Volume')] = v
            
            # Use eps to avoid division by zero
            eps = np.finfo(float).eps
            
            # Compute the 1-day forward returns
            self._features_data.loc[:, (ticker, '1DFRet')] = (c.shift(-1) / c) - 1
            
            # Compute the candle body features
            # Body relative to total range, clip for stability
            candle_height = (h - l)
            candle_height = candle_height.mask(candle_height < eps, eps)
            
            self._features_data.loc[:, (ticker, 'Body/HL')] = (
                (c - o) / candle_height
            ).clip(-1.0, 1.0)

            # Upper shadow relative to range
            self._features_data.loc[:, (ticker, 'UWick/HL')] = (
                    (h - np.maximum(o, c)) / candle_height
            ).clip(0.0, 1.0)

            # Lower shadow relative to range
            self._features_data.loc[:, (ticker, 'LWick/HL')] = (
                (np.minimum(o, c) - l) / candle_height
            ).clip(0.0, 1.0)
            
            # Gap and Gap fill relative to the previous close
            pc = c.shift(1)
            gap_raw = o - pc
            gap = np.log(o / pc)
            
            # Identify positions for gap up and gap down
            gap_up = gap_raw > 0
            gap_down = gap_raw < 0

            # Compute the gap fill %
            gap_fill = pd.Series(0.0, index=gap.index)
            gap_fill[gap_up] = (o - l).where(gap_up) / (gap_raw).where(gap_up)
            gap_fill[gap_down] = (h - o).where(gap_down) / (-gap_raw).where(gap_down)
            
            # Assign computed gap metrics in features data
            self._features_data.loc[:, (ticker, 'Gap')] = self._norm(gap)
            self._features_data.loc[:, (ticker, 'GapFill')] = gap_fill.clip(0.0, 1.0)

            # Compute rolling Exponential Moving Averages: 5, 13, 26
            ema5 = EMAIndicator(close=c, window=5).ema_indicator()
            ema13 = EMAIndicator(close=c, window=13).ema_indicator()
            ema26 = EMAIndicator(close=c, window=26).ema_indicator()
            ema50 = EMAIndicator(close=c, window=50).ema_indicator()

            # Compute the EMA ratios
            self._features_data.loc[:, (ticker, 'EMA5/13')] = self._norm((ema5 / ema13) - 1)
            self._features_data.loc[:, (ticker, 'EMA13/26')] = self._norm((ema13 / ema26) - 1)
            self._features_data.loc[:, (ticker, 'EMA26/50')] = self._norm((ema26 / ema50) - 1)


            # Compute Bollinger Bands (%B and Bandwidth)
            bb = BollingerBands(c, window=20, window_dev=2)
            bb_sma20 = bb.bollinger_mavg()
            bb_upper = bb.bollinger_hband()
            bb_lower = bb.bollinger_lband()
            bb_width = bb_upper - bb_lower
            # Avoid division by zero
            bb_width = bb_width.mask(bb_width.abs() < eps, eps)
            bb_sma20 = bb_sma20.mask(bb_sma20.abs() < eps, eps)
            self._features_data.loc[:, (ticker, 'B%B')] = (c - bb_lower) / bb_width
            self._features_data.loc[:, (ticker, 'BBW')] = self._norm(bb_width / bb_sma20)

            # Compute Relative Strength Index (RSI)
            rsi = RSIIndicator(c, window=14).rsi()
            self._features_data.loc[:, (ticker, 'RSI')] = self._norm(rsi)

            # Compute Average Directional Index scaled bw [0, 1]
            adx = ADXIndicator(high=h, low=l, close=c, window=14).adx()
            self._features_data.loc[:, (ticker, 'ADX')] = adx / 100.0

            # Compute ratio of current volume over 20 volume moving average
            vol20 = v.rolling(20, min_periods=20).mean()
            vol20 = vol20.mask(vol20 < eps, eps)
            self._features_data.loc[:, (ticker, 'V/Vol20')] = self._norm(v / vol20)
        
        # Drop all rows with NaN values
        self._features_data = self._features_data.dropna()

        # Save the features data to use for later
        if save:
            self._save_features_data()
        
        # build once at the end of build_features
        feat_level = self._features_data.columns.get_level_values('Feature').unique()
        
        # Rewards are seperated from other features to avoid data leakage causing bias    
        feature_indices = {
            'OHLCV': np.flatnonzero(feat_level.isin(OHLCV_f)),
            'Rewards': np.flatnonzero(feat_level.isin(returns_f)),
            'State': np.flatnonzero(feat_level.isin(state_f))
        }
        
        self._feature_indices = feature_indices

    def get_features(self) -> pd.DataFrame:
        if self._features_data is None:
            return pd.DataFrame()
        
        return self._features_data
    
    def get_feature_indices(self) -> dict:
        if self._feature_indices is None:
            return {}
        
        return self._feature_indices
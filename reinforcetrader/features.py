import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from ta.trend import EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator

class FeatureBuilder:
    def __init__(self, hist_prices):
        self._hist_prices = hist_prices.sort_index()
        self._features_data = None

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
        file_path = save_dir / f"tickers_features_{start_date}_{end_date}.csv"

        # Save the features data
        self._features_data.to_csv(file_path)
        print(f"Features data saved to {file_path}")

        return True

    def build_features(self, save=True):
        # Get tickers symbols
        tickers = self._hist_prices.columns.get_level_values('Ticker').unique()

        # Predefine feature names and dataframe
        price_features = ['Close', 'Body/HL', 'UShadow/HL', 'LShadow/HL']
        technical_features = ['C/EMA5', 'EMA5/EMA13', 'EMA13/EMA26', 'B%B', 'BBW', 'RSI', 'ADX', 'V/Vol20']
        feature_columns = pd.MultiIndex.from_product([tickers, price_features + technical_features],
                                                     names=['Ticker', 'Feature'])
        self._features_data = pd.DataFrame(index=self._hist_prices.index, columns=feature_columns, dtype=float)

        for ticker in tqdm(tickers, ncols=100, desc='Building ticker features'):
            open = self._hist_prices[ticker]['Open']
            high = self._hist_prices[ticker]['High']
            low = self._hist_prices[ticker]['Low']
            close = self._hist_prices[ticker]['Close']
            volume = self._hist_prices[ticker]['Volume']

            # Keep close price as is (may not be part of state representation)
            self._features_data.loc[:, (ticker, 'Close')] = close

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
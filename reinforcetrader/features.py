import numpy as np
import pandas as pd

from tqdm import tqdm
from ta.trend import EMAIndicator, ADXIndicator
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator

class FeatureBuilder:
    def __init__(self, hist_prices):
        """
        hist_prices: DataFrame with MultiIndex columns ['Ticker','Price']
                     where Price ∈ ['Open','High','Low','Close','Volume']
        """
        self._hist_prices = hist_prices.sort_index()
        self._features_data = None


    def build_features(self):
        # Get tickers symbols
        tickers = self._hist_prices.columns.get_level_values('Ticker').unique()

        # Predefine feature names and dataframe
        technical_features = ['Close', 'C/EMA5', 'EMA5/EMA13', 'EMA13/EMA26', 'B%B', 'BBW', 'RSI', 'ADX', 'V/Vol20']
        feature_columns = pd.MultiIndex.from_product([tickers, technical_features],
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

    def get_features(self):
        return self._features_data
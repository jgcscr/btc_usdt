"""
MarketRegimeDetector: Identifies market regimes (trending, ranging, volatile) for BTC-USDT using technical indicators.
"""

import pandas as pd
import numpy as np
from typing import Optional, Literal

class MarketRegimeDetector:
    """
    Detects market regimes using ATR, ADX, and Bollinger Band width.
    Methods:
        detect_volatility_regime: Classifies volatility using ATR.
        detect_trend_strength: Classifies trend using ADX.
        detect_range_bound: Detects range-bound using Bollinger Band width.
        classify_regime: Combines all signals for overall regime.
    """
    def __init__(self, atr_window: int = 14, adx_window: int = 14, bb_window: int = 20, bb_std: float = 2.0):
        self.atr_window = atr_window
        self.adx_window = adx_window
        self.bb_window = bb_window
        self.bb_std = bb_std

    def detect_volatility_regime(self, df: pd.DataFrame, threshold: Optional[float] = None) -> pd.Series:
        """
        Detects volatility regime using ATR.
        Args:
            df: DataFrame with columns ['high', 'low', 'close']
            threshold: Optional fixed ATR threshold. If None, uses rolling median * 1.5.
        Returns:
            pd.Series: 'volatile' or 'normal'
        """
        high, low, close = df['high'], df['low'], df['close']
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift()), abs(low - close.shift())))
        atr = tr.rolling(self.atr_window).mean()
        if threshold is None:
            threshold = atr.rolling(self.atr_window * 2).median() * 1.5
        regime = np.where(atr > threshold, 'volatile', 'normal')
        return pd.Series(regime, index=df.index, name='volatility_regime')

    def detect_trend_strength(self, df: pd.DataFrame, strong_threshold: float = 25.0, weak_threshold: float = 20.0) -> pd.Series:
        """
        Detects trend strength using ADX.
        Args:
            df: DataFrame with 'high', 'low', 'close', and 'ADX_14' or 'adx_14' column.
            strong_threshold: ADX value above which trend is strong.
            weak_threshold: ADX value below which trend is weak.
        Returns:
            pd.Series: 'trending' or 'not_trending'
        """
        adx_col = 'adx_14' if 'adx_14' in df.columns else 'ADX_14'
        adx = df[adx_col]
        regime = np.where(adx >= strong_threshold, 'trending',
                 np.where(adx <= weak_threshold, 'not_trending', 'neutral'))
        return pd.Series(regime, index=df.index, name='trend_regime')

    def detect_range_bound(self, df: pd.DataFrame, width_threshold: Optional[float] = None) -> pd.Series:
        """
        Detects range-bound regime using Bollinger Band width.
        Args:
            df: DataFrame with 'close', 'bb_upper', 'bb_lower' columns.
            width_threshold: Optional fixed width threshold. If None, uses rolling median * 0.7.
        Returns:
            pd.Series: 'range' or 'not_range'
        """
        if 'bb_upper' not in df.columns or 'bb_lower' not in df.columns:
            # Compute Bollinger Bands if not present
            ma = df['close'].rolling(self.bb_window).mean()
            std = df['close'].rolling(self.bb_window).std()
            bb_upper = ma + self.bb_std * std
            bb_lower = ma - self.bb_std * std
        else:
            bb_upper = df['bb_upper']
            bb_lower = df['bb_lower']
        width = (bb_upper - bb_lower) / df['close']
        if width_threshold is None:
            width_threshold = width.rolling(self.bb_window * 2).median() * 0.7
        regime = np.where(width < width_threshold, 'range', 'not_range')
        return pd.Series(regime, index=df.index, name='range_regime')

    def classify_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Combines volatility, trend, and range signals into a single regime label.
        Args:
            df: DataFrame with required columns for indicators.
        Returns:
            pd.Series: Regime label ('trending', 'ranging', 'volatile', 'normal')
        """
        vol = self.detect_volatility_regime(df)
        trend = self.detect_trend_strength(df)
        rng = self.detect_range_bound(df)
        regime = []
        for v, t, r in zip(vol, trend, rng):
            if v == 'volatile':
                regime.append('volatile')
            elif t == 'trending':
                regime.append('trending')
            elif r == 'range':
                regime.append('ranging')
            else:
                regime.append('normal')
        return pd.Series(regime, index=df.index, name='market_regime')

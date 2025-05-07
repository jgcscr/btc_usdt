import numpy as np
import pandas as pd
from typing import List, Callable, Optional, Dict, Any, Union
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.inspection import permutation_importance
import pickle
import matplotlib.pyplot as plt

# --- Technical Indicator Transformers ---
class RSICalculator(BaseEstimator, TransformerMixin):
    def __init__(self, period: int = 14, col: str = 'close', out_col: str = 'rsi_14'):
        self.period = period
        self.col = col
        self.out_col = out_col
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df = X.copy()
        delta = df[self.col].diff()
        up = delta.clip(lower=0)
        down = -delta.clip(upper=0)
        ma_up = up.rolling(self.period, min_periods=1).mean()
        ma_down = down.rolling(self.period, min_periods=1).mean()
        rs = ma_up / (ma_down + 1e-9)
        df[self.out_col] = 100 - (100 / (1 + rs))
        return df

class MACDCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9, col: str = 'close'):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.col = col
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df = X.copy()
        ema_fast = df[self.col].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df[self.col].ewm(span=self.slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=self.signal, adjust=False).mean()
        df['macd'] = macd
        df['macd_signal'] = signal_line
        df['macd_diff'] = macd - signal_line
        return df

class BollingerBandsCalculator(BaseEstimator, TransformerMixin):
    def __init__(self, period: int = 20, std: float = 2.0, col: str = 'close'):
        self.period = period
        self.std = std
        self.col = col
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df = X.copy()
        ma = df[self.col].rolling(self.period, min_periods=1).mean()
        std = df[self.col].rolling(self.period, min_periods=1).std()
        df['bb_upper'] = ma + self.std * std
        df['bb_middle'] = ma
        df['bb_lower'] = ma - self.std * std
        return df

# --- Feature Scaling Transformers ---
class DataFrameScaler(BaseEstimator, TransformerMixin):
    def __init__(self, scaler=StandardScaler(), cols: Optional[List[str]] = None):
        self.scaler = scaler
        self.cols = cols
        self.fitted = False
    def fit(self, X, y=None):
        cols = self.cols or X.select_dtypes(include=[np.number]).columns.tolist()
        self.scaler.fit(X[cols].fillna(0))
        self.cols_ = cols
        self.fitted = True
        return self
    def transform(self, X):
        df = X.copy()
        if not self.fitted:
            raise RuntimeError("Scaler not fitted. Call fit or fit_transform first.")
        arr = self.scaler.transform(df[self.cols_].fillna(0))
        df[self.cols_] = arr
        return df

# --- Feature Selection Utilities ---
def variance_threshold_selector(df: pd.DataFrame, threshold: float = 0.0) -> List[str]:
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df.fillna(0))
    return df.columns[selector.get_support()].tolist()

def correlation_selector(df: pd.DataFrame, threshold: float = 0.95) -> List[str]:
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return [col for col in df.columns if col not in to_drop]

# --- FeaturePipeline Class ---
class FeaturePipeline:
    def __init__(self) -> None:
        self.transformers: List[Any] = []
        self.fitted: bool = False
        self.columns_: Optional[pd.Index] = None
        self.index_: Optional[pd.Index] = None
    def add_transformer(self, transformer: Any) -> None:
        self.transformers.append(transformer)
    def fit(self, df: pd.DataFrame) -> 'FeaturePipeline':
        for t in self.transformers:
            if hasattr(t, 'fit'):
                t.fit(df)
        self.fitted = True
        self.columns_ = df.columns
        self.index_ = df.index
        return self
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        for t in self.transformers:
            if hasattr(t, 'transform'):
                df = t.transform(df)
        if self.columns_ is not None:
            df = df.reindex(columns=self.columns_, fill_value=np.nan)
        if self.index_ is not None and len(df) == len(self.index_):
            df.index = self.index_
        return df
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
    def save_pipeline(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    @staticmethod
    def load_pipeline(path: str) -> 'FeaturePipeline':
        with open(path, 'rb') as f:
            return pickle.load(f)

# --- Feature Importance Analysis ---
def calculate_feature_importance(model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5, random_state: int = 42) -> Dict[str, float]:
    result = permutation_importance(model, X.fillna(0), y, n_repeats=n_repeats, random_state=random_state)
    return dict(zip(X.columns, result.importances_mean))

def plot_feature_importance(importances: Dict[str, float], top_n: int = 20):
    sorted_items = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    labels, values = zip(*sorted_items)
    plt.figure(figsize=(10, 6))
    plt.barh(labels, values)
    plt.xlabel('Importance')
    plt.title('Feature Importance (Permutation)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

def select_top_features(X: pd.DataFrame, importances: Dict[str, float], top_n: int) -> pd.DataFrame:
    top_features = [k for k, v in sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]]
    return X[top_features]

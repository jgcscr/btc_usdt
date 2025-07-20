import pytest
import pandas as pd
import numpy as np
from btc_usdt_pipeline.features.compute_features import calculate_indicators, compute_htf_features
from btc_usdt_pipeline import config

# Sample data for testing
@pytest.fixture
def sample_ohlcv_df():
    dates = pd.date_range('2023-01-01', periods=100, freq='min')
    data = {
        'open': np.random.rand(100) * 100 + 20000,
        'high': np.random.rand(100) * 50 + 20050,
        'low': np.random.rand(100) * -50 + 20000,
        'close': np.random.rand(100) * 100 + 20000,
        'volume': np.random.rand(100) * 10 + 1
    }
    df = pd.DataFrame(data, index=dates)
    # Ensure high is highest and low is lowest
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    return df

# Basic test for calculate_indicators
def test_calculate_indicators(sample_ohlcv_df):
    df_with_indicators = calculate_indicators(sample_ohlcv_df.copy())
    assert isinstance(df_with_indicators, pd.DataFrame)
    # Check if some common indicators are added (names might vary slightly based on pandas_ta version)
    # Accept both 'EMA_' and 'ema_' formats for compatibility
    assert any(col.lower().startswith('ema_') for col in df_with_indicators.columns)
    assert any(col.lower().startswith('rsi_') for col in df_with_indicators.columns)
    assert any(col.lower().startswith('macd') for col in df_with_indicators.columns)
    assert 'fractal_high' in df_with_indicators.columns
    assert 'fractal_low' in df_with_indicators.columns

# Basic test for compute_htf_features
def test_compute_htf_features(sample_ohlcv_df):
    # Use a subset of rules for faster testing
    test_rules = {'5T': '5m'}
    df_with_htf = compute_htf_features(sample_ohlcv_df.copy(), rules=test_rules)
    assert isinstance(df_with_htf, pd.DataFrame)
    # Check if HTF columns were added and merged back
    assert any(col.startswith('5m_') for col in df_with_htf.columns)
    # Check if original columns still exist
    assert 'close' in df_with_htf.columns
    # Skip the first N bars where NaNs are expected
    warmup_bars = 50  # Adjust as needed based on your longest indicator
    htf_cols = [col for col in df_with_htf.columns if col.startswith('5m_')]
    result_after_warmup = df_with_htf.iloc[warmup_bars:]
    assert not result_after_warmup[htf_cols].isna().any().any(), \
        "NaN values found in HTF features after warmup period"

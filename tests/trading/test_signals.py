import pytest
import pandas as pd
import numpy as np
from btc_usdt_pipeline.trading.signals import generate_signals, multi_timeframe_signal_logic
from btc_usdt_pipeline import config

# Sample data for signal generation
@pytest.fixture
def sample_signal_data():
    dates = pd.date_range('2023-01-01', periods=5, freq='min')
    data = {
        'close': [100, 101, 102, 103, 104],
        # Add dummy HTF features required by multi_timeframe_signal_logic
        '4h_ema_20': [95]*5, '4h_rsi_14': [60]*5,
        '1h_ema_20': [98]*5, '1h_rsi_14': [55]*5,
    }
    df = pd.DataFrame(data, index=dates)
    # Probabilities corresponding to the dataframe rows
    probabilities = np.array([0.4, 0.6, 0.7, 0.3, 0.8])
    return df, probabilities

# Basic test for generate_signals
def test_generate_signals(sample_signal_data):
    df, probabilities = sample_signal_data
    threshold = 0.55 # Use a specific threshold for testing
    signals = generate_signals(df, probabilities, threshold=threshold)

    assert isinstance(signals, np.ndarray)
    assert len(signals) == len(df)
    expected_signals = np.array(["Flat", "Long", "Long", "Flat", "Long"])
    np.testing.assert_array_equal(signals, expected_signals)

# Test generate_signals with different threshold
def test_generate_signals_high_threshold(sample_signal_data):
    df, probabilities = sample_signal_data
    threshold = 0.75
    signals = generate_signals(df, probabilities, threshold=threshold)
    expected_signals = np.array(["Flat", "Flat", "Flat", "Flat", "Long"])
    np.testing.assert_array_equal(signals, expected_signals)

# Test generate_signals with length mismatch
def test_generate_signals_length_mismatch(sample_signal_data):
    df, probabilities = sample_signal_data
    short_probs = probabilities[:-1]
    with pytest.raises(ValueError):
        generate_signals(df, short_probs)

# Basic tests for multi_timeframe_signal_logic
def test_multi_timeframe_signal_logic_long(sample_signal_data):
    df, _ = sample_signal_data
    row = df.iloc[1] # Example row where HTF is bullish
    prob = 0.8 # High probability
    signal = multi_timeframe_signal_logic(row, prob, prob_threshold=0.7)
    assert signal == "Long"

def test_multi_timeframe_signal_logic_flat_low_prob(sample_signal_data):
    df, _ = sample_signal_data
    row = df.iloc[1] # Example row where HTF is bullish
    prob = 0.5 # Low probability
    signal = multi_timeframe_signal_logic(row, prob, prob_threshold=0.7)
    assert signal == "Flat"

def test_multi_timeframe_signal_logic_flat_mixed_htf(sample_signal_data):
    df, _ = sample_signal_data
    row = df.iloc[1].copy()
    row['1h_rsi_14'] = 40 # Make 1h bearish
    prob = 0.8 # High probability
    signal = multi_timeframe_signal_logic(row, prob, prob_threshold=0.7)
    assert signal == "Flat"

def test_multi_timeframe_signal_logic_short(sample_signal_data):
    df, _ = sample_signal_data
    row = df.iloc[1].copy()
    # Make HTF bearish
    row['close'] = 90
    row['4h_ema_20'] = 95
    row['4h_rsi_14'] = 40
    row['1h_ema_20'] = 92
    row['1h_rsi_14'] = 35
    prob = 0.8 # High probability
    signal = multi_timeframe_signal_logic(row, prob, prob_threshold=0.7)
    assert signal == "Short"

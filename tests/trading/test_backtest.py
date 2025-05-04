import pytest
import pandas as pd
import numpy as np
from btc_usdt_pipeline.trading.backtest import run_backtest
from btc_usdt_pipeline import config

# Sample data for backtesting
@pytest.fixture
def sample_backtest_data():
    dates = pd.date_range('2023-01-01', periods=10, freq='min')
    data = {
        'open': [100, 101, 102, 101, 103, 104, 105, 104, 106, 107],
        'high': [101, 102, 103, 102, 104, 105, 106, 105, 107, 108],
        'low':  [99, 100, 101, 100, 102, 103, 104, 103, 105, 106],
        'close':[101, 102, 101, 103, 104, 105, 104, 106, 107, 106],
        config.BACKTEST_ATR_COLUMN: [1.0] * 10 # Simple constant ATR for testing
    }
    df = pd.DataFrame(data, index=dates)
    signals = np.array(["Flat", "Long", "Flat", "Flat", "Long", "Flat", "Short", "Flat", "Flat", "Flat"])
    return df, signals

# Basic test for run_backtest structure and output types
def test_run_backtest_output_types(sample_backtest_data):
    df, signals = sample_backtest_data
    equity_curve, trade_log = run_backtest(df, signals, initial_equity=10000)

    assert isinstance(equity_curve, list)
    assert isinstance(trade_log, list)
    assert len(equity_curve) == len(df) + 1 # Initial equity + equity after each bar
    assert all(isinstance(e, float) for e in equity_curve)
    if trade_log: # Only check if trades occurred
        assert all(isinstance(t, dict) for t in trade_log)

# Test backtest with a simple winning long trade
def test_run_backtest_simple_long_win(sample_backtest_data):
    df, _ = sample_backtest_data
    # Force a long entry and TP hit
    signals = np.array(["Flat", "Long", "Flat", "Flat", "Flat", "Flat", "Flat", "Flat", "Flat", "Flat"])
    # Adjust data slightly to ensure TP hit
    df.loc[df.index[2], 'high'] = 106 # Ensure TP (102 + 1*2 = 104) is hit on 3rd bar

    equity_curve, trade_log = run_backtest(
        df, signals,
        initial_equity=10000,
        sl_multiplier=1.0, # SL = 1 point
        tp_multiplier=2.0, # TP = 2 points
        commission_rate=0, # No commission for simplicity
        slippage_points=0 # No slippage for simplicity
    )

    assert len(trade_log) == 1
    trade = trade_log[0]
    assert trade['Type'] == 'Long'
    assert trade['Entry'] == 102 # Close of bar 1
    assert trade['Exit Reason'] == 'TP'
    assert trade['Exit'] == 104 # TP target
    assert trade['PnL'] > 0 # Should be profitable
    assert equity_curve[-1] > 10000 # Final equity should be higher

# Test backtest with length mismatch
def test_run_backtest_length_mismatch(sample_backtest_data):
    df, signals = sample_backtest_data
    short_signals = signals[:-1] # Make signals shorter
    equity_curve, trade_log = run_backtest(df, short_signals)
    assert equity_curve == [config.INITIAL_EQUITY] # Should return initial equity only
    assert trade_log == [] # No trades

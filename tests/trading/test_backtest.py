import pytest
import pandas as pd
import numpy as np
import math
from btc_usdt_pipeline.trading.backtest import run_backtest, validate_inputs
from btc_usdt_pipeline import config
from btc_usdt_pipeline.utils.helpers import ParameterValidationError, DataAlignmentError
from btc_usdt_pipeline.utils.data_processing import align_and_validate_data

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

def make_df(entry_price, atr=10, n=3):
    # Create a simple DataFrame for 3 bars
    return pd.DataFrame({
        'open': [entry_price]*n,
        'high': [entry_price+5, entry_price+15, entry_price+5],
        'low': [entry_price-5, entry_price-5, entry_price-15],
        'close': [entry_price]*n,
        'atr_14': [atr]*n
    })

def test_long_slippage():
    df = make_df(100)
    signals = np.array(["Long", "Flat", "Flat"])
    # Set SL and TP so that TP is hit on bar 1
    equity_curve, trade_log = run_backtest(df, signals, initial_equity=10000, atr_col='atr_14', sl_multiplier=0.5, tp_multiplier=1.0, commission_rate=0, slippage_points=2, risk_fraction=1)
    trade = trade_log[0]
    # Entry: 100+2=102, TP: 102+10=112, Exit: 112-2=110
    assert trade['Entry'] == 102
    assert trade['Exit'] == 110
    assert trade['Type'] == 'Long'
    # SL test
    signals = np.array(["Long", "Flat", "Flat"])
    df2 = make_df(100)
    df2.iloc[1, df2.columns.get_loc('low')] = 80 # Force SL hit
    equity_curve, trade_log = run_backtest(df2, signals, initial_equity=10000, atr_col='atr_14', sl_multiplier=1.0, tp_multiplier=2.0, commission_rate=0, slippage_points=3, risk_fraction=1)
    trade = trade_log[0]
    # Entry: 100+3=103, SL: 103-10=93, Exit: 93-3=90
    assert trade['Entry'] == 103
    assert trade['Exit'] == 90
    assert trade['Type'] == 'Long'

def test_short_slippage():
    df = make_df(200)
    signals = np.array(["Short", "Flat", "Flat"])
    # Set SL and TP so that TP is hit on bar 1
    equity_curve, trade_log = run_backtest(df, signals, initial_equity=10000, atr_col='atr_14', sl_multiplier=0.5, tp_multiplier=1.0, commission_rate=0, slippage_points=4, risk_fraction=1)
    trade = trade_log[0]
    # Entry: 200-4=196, TP: 196-10=186, Exit: 186+4=190
    assert trade['Entry'] == 196
    assert trade['Exit'] == 190
    assert trade['Type'] == 'Short'
    # SL test
    signals = np.array(["Short", "Flat", "Flat"])
    df2 = make_df(200)
    df2.iloc[1, df2.columns.get_loc('high')] = 220 # Force SL hit
    equity_curve, trade_log = run_backtest(df2, signals, initial_equity=10000, atr_col='atr_14', sl_multiplier=1.0, tp_multiplier=2.0, commission_rate=0, slippage_points=5, risk_fraction=1)
    trade = trade_log[0]
    # Entry: 200-5=195, SL: 195+10=205, Exit: 205+5=210
    assert trade['Entry'] == 195
    assert trade['Exit'] == 210
    assert trade['Type'] == 'Short'

def test_long_slippage_applied():
    df = make_df(100)
    signals = np.array(["Long", "Flat", "Flat"])
    equity_curve, trade_log = run_backtest(df, signals, initial_equity=10000, atr_col='atr_14', sl_multiplier=1.0, tp_multiplier=1.0, commission_rate=0, slippage_points=5, risk_fraction=1)
    trade = trade_log[0]
    # Entry: 100+5=105, TP: 105+10=115, Exit: 115-5=110
    assert trade['Entry'] == 105
    assert trade['Exit'] == 110
    assert trade['Type'] == 'Long'
    assert trade['Exit Reason'] == 'TP'

def test_short_slippage_applied():
    df = make_df(200)
    signals = np.array(["Short", "Flat", "Flat"])
    equity_curve, trade_log = run_backtest(df, signals, initial_equity=10000, atr_col='atr_14', sl_multiplier=1.0, tp_multiplier=1.0, commission_rate=0, slippage_points=7, risk_fraction=1)
    trade = trade_log[0]
    # Entry: 200-7=193, TP: 193-10=183, Exit: 183+7=190
    assert trade['Entry'] == 193
    assert trade['Exit'] == 190
    assert trade['Type'] == 'Short'
    assert trade['Exit Reason'] == 'TP'

def test_high_slippage_warning_and_impact(caplog):
    df = make_df(100)
    signals = np.array(["Long", "Flat", "Flat"])
    # Use very high slippage
    with caplog.at_level('WARNING'):
        equity_curve, trade_log = run_backtest(df, signals, initial_equity=10000, atr_col='atr_14', sl_multiplier=1.0, tp_multiplier=1.0, commission_rate=0, slippage_points=50, risk_fraction=1)
        # Should warn about excessive slippage
        assert any("Slippage (50) is more than 10% of average price" in m for m in caplog.messages)
    # Check that slippage cost is as expected
    from btc_usdt_pipeline.trading.backtest import estimate_total_slippage_cost
    total_slip = estimate_total_slippage_cost(trade_log)
    # Entry: 100+50=150, TP: 150+10=160, Exit: 160-50=110, so entry slip=50, exit slip=110-160=-50
    assert total_slip == 100

def generate_market_data(n=10, volatility=1.0, gap_indices=None, trend=0.0, base_price=100, volume=1000):
    """
    Generate synthetic OHLCV data with configurable volatility, gaps, and trend.
    gap_indices: list of indices where price gaps occur
    trend: per-bar drift
    """
    np.random.seed(42)
    prices = [base_price]
    for i in range(1, n):
        change = np.random.randn() * volatility + trend
        if gap_indices and i in gap_indices:
            change += np.random.choice([-10, 10])  # Large gap
        prices.append(prices[-1] + change)
    prices = np.array(prices)
    high = prices + np.abs(np.random.randn(n))
    low = prices - np.abs(np.random.randn(n))
    open_ = prices + np.random.randn(n) * 0.5
    close = prices + np.random.randn(n) * 0.5
    atr = np.abs(np.random.randn(n)) + 1
    vol = np.abs(np.random.randn(n)) * volume
    df = pd.DataFrame({
        'open': open_, 'high': high, 'low': low, 'close': close, 'atr_14': atr, 'volume': vol
    }, index=pd.date_range('2025-01-01', periods=n, freq='min'))
    return df

def test_extreme_market_conditions():
    # Simulate limit up/down and gaps
    df = generate_market_data(n=6, volatility=2, gap_indices=[2, 4], trend=0.5)
    signals = np.array(["Long", "Flat", "Short", "Flat", "Long", "Flat"])
    equity_curve, trade_log = run_backtest(df, signals, initial_equity=10000, slippage_points=1)
    assert isinstance(equity_curve, list)
    assert isinstance(trade_log, list)
    # Check that trades are logged and equity is always >= 0
    assert all(e >= 0 for e in equity_curve)

def test_zero_equity_bankruptcy():
    df = generate_market_data(n=5, volatility=0.1)
    signals = np.array(["Long"] * 5)
    # Set initial equity to zero
    equity_curve, trade_log = run_backtest(df, signals, initial_equity=0)
    assert all(e == 0 for e in equity_curve)
    # Set risk_fraction to 1, but commission so high that equity goes to zero
    equity_curve, trade_log = run_backtest(df, signals, initial_equity=100, commission_rate=1.0, risk_fraction=1)
    assert any(e == 0 for e in equity_curve)

def test_trade_spanning_end_of_data():
    df = generate_market_data(n=4, volatility=1)
    signals = np.array(["Flat", "Long", "Flat", "Flat"])
    # Should open a trade and close it at end of data
    equity_curve, trade_log = run_backtest(df, signals, initial_equity=1000)
    assert trade_log[-1]['Exit Reason'] == 'EndOfData'

def test_invalid_inputs_raise():
    df = generate_market_data(n=3)
    signals = np.array(["Long", "Short"])  # Mismatched length
    with pytest.raises(DataAlignmentError):
        run_backtest(df, signals)
    # Missing required column
    df2 = df.drop(columns=['open'])
    with pytest.raises(ParameterValidationError):
        run_backtest(df2, np.array(["Long"]*3))
    # NaN in required column
    df3 = df.copy(); df3['close'][1] = np.nan
    with pytest.raises(ParameterValidationError):
        run_backtest(df3, np.array(["Long"]*3))
    # Out-of-range parameters
    with pytest.raises(ParameterValidationError):
        run_backtest(df, np.array(["Long"]*3), sl_multiplier=-1)
    with pytest.raises(ParameterValidationError):
        run_backtest(df, np.array(["Long"]*3), tp_multiplier=0)
    with pytest.raises(ParameterValidationError):
        run_backtest(df, np.array(["Long"]*3), risk_fraction=2)
    with pytest.raises(ParameterValidationError):
        run_backtest(df, np.array(["Long"]*3), commission_rate=-0.1)

def test_alignment_validation():
    df = generate_market_data(n=5)
    # Index mismatch: signals as Series with shifted index
    signals = pd.Series(["Long"]*4, index=df.index[1:])
    aligned_df, aligned_signals = align_and_validate_data(df, signals, arr_name="signals")
    assert len(aligned_df) == len(aligned_signals) == 4
    # No overlap
    signals2 = pd.Series(["Long"]*4, index=pd.date_range('2030-01-01', periods=4, freq='min'))
    with pytest.raises(DataAlignmentError):
        align_and_validate_data(df, signals2, arr_name="signals")

def test_negative_slippage():
    df = generate_market_data(n=3)
    signals = np.array(["Long", "Flat", "Flat"])
    # Negative slippage: should still run, but log warning if logic is violated
    equity_curve, trade_log = run_backtest(df, signals, slippage_points=-2)
    assert isinstance(equity_curve, list)
    # Extremely high slippage
    equity_curve, trade_log = run_backtest(df, signals, slippage_points=1e6)
    assert all(e >= 0 for e in equity_curve)
    # Slippage exceeds ATR
    equity_curve, trade_log = run_backtest(df, signals, slippage_points=100)
    for trade in trade_log:
        assert abs(trade['Entry'] - df['close'].iloc[0]) >= 0

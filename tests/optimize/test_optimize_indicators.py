import pytest
import pandas as pd
import numpy as np
from btc_usdt_pipeline.optimize.optimize_indicators import load_data_for_indicator_opt, objective_indicators
from btc_usdt_pipeline import config

# Basic test for load_data_for_indicator_opt (mocking file reads)
def test_load_data_for_indicator_opt(mocker):
    # Dummy data for enriched parquet
    dummy_enriched_df = pd.DataFrame({
        'open_time': pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:01:00']),
        'open': [100, 101], 'high': [102, 102], 'low': [99, 100], 'close': [101, 101.5], 'volume': [10, 12],
        config.BACKTEST_ATR_COLUMN: [1.5, 1.6] # Include ATR column
    }).set_index('open_time')
    # Dummy data for predictions json
    dummy_preds_dict = {
        'index': ['2023-01-01 00:00:00', '2023-01-01 00:01:00'],
        'ensemble_prob': [0.6, 0.7]
    }
    mocker.patch('pandas.read_parquet', return_value=dummy_enriched_df)
    mocker.patch('btc_usdt_pipeline.utils.helpers.load_json', return_value=dummy_preds_dict)

    df_opt, preds_opt = load_data_for_indicator_opt()

    assert df_opt is not None
    assert preds_opt is not None
    assert isinstance(df_opt, pd.DataFrame)
    assert isinstance(preds_opt, np.ndarray)
    assert not df_opt.empty
    assert len(df_opt) == len(preds_opt)
    assert config.BACKTEST_ATR_COLUMN in df_opt.columns
    assert 'ensemble_prob' in df_opt.columns

# Minimal test for objective_indicators structure (mocking load_data and backtest)
def test_objective_indicators_structure(mocker):
    # Mock load_data to return minimal data
    dummy_df = pd.DataFrame({
        'open': [100]*5, 'high': [101]*5, 'low': [99]*5, 'close': [100]*5, 'volume': [10]*5,
        config.BACKTEST_ATR_COLUMN: [1]*5,
        'ensemble_prob': [0.6, 0.7, 0.4, 0.8, 0.5]
    }, index=pd.date_range('2023-01-01', periods=5, freq='min'))
    dummy_probs = dummy_df['ensemble_prob'].values
    mocker.patch('btc_usdt_pipeline.optimize.optimize_indicators.load_data_for_indicator_opt', return_value=(dummy_df, dummy_probs))

    # Mock generate_signals and run_backtest
    mocker.patch('btc_usdt_pipeline.trading.signals.generate_signals', return_value=np.array(["Long", "Long", "Flat", "Long", "Flat"]))
    # Mock backtest to return some basic results
    mock_equity_curve = [10000, 10010, 10005, 10025, 10020]
    mock_trade_log = [{'Type': 'Long', 'PnL': 10}, {'Type': 'Long', 'PnL': 20}]
    mocker.patch('btc_usdt_pipeline.optimize.optimize_indicators.run_backtest', return_value=(mock_equity_curve, mock_trade_log))
    # Mock calculate_metrics
    mocker.patch('btc_usdt_pipeline.utils.helpers.calculate_metrics', return_value={'Profit Factor': 2.5, 'Net Profit': 30, 'Total Trades': 2})

    # Mock optuna trial
    mock_trial = mocker.Mock()
    mock_trial.suggest_float.side_effect = [0.6, 1.5, 2.0] # prob_threshold, sl_multiplier, tp_multiplier
    mock_trial.number = 0
    mock_trial.params = {'prob_threshold': 0.6}

    result = objective_indicators(mock_trial)
    assert isinstance(result, float)
    assert result != -float('inf') # Should not fail
    assert result > 0 # Expecting positive profit factor in this mock

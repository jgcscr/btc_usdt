import pytest
import pandas as pd
import numpy as np
from btc_usdt_pipeline.optimize.auto_optimize import load_data_once, objective
from btc_usdt_pipeline import config

# Basic test for load_data_once (mocking pd.read_parquet)
def test_load_data_once(mocker):
    # Create a dummy DataFrame to be returned by read_parquet
    dummy_df = pd.DataFrame({
        'open_time': pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:01:00']),
        'close': [100, 101],
        'feature1': [1, 2],
        config.TARGET_COLUMN_NAME: [0, 1] # Include target column
    }).set_index('open_time')
    # Mock read_parquet
    mocker.patch('pandas.read_parquet', return_value=dummy_df)
    # Mock config features to match dummy data
    mocker.patch.object(config, 'ALL_TREE_FEATURES', ['feature1'])

    loaded_df = load_data_once()
    assert loaded_df is not None
    assert isinstance(loaded_df, pd.DataFrame)
    assert not loaded_df.empty
    assert 'feature1' in loaded_df.columns
    assert config.TARGET_COLUMN_NAME in loaded_df.columns

    # Test that it returns the cached version
    loaded_df_again = load_data_once()
    assert loaded_df_again is loaded_df # Should be the same object

# Minimal test for objective function structure (mocking load_data_once)
def test_objective_structure(mocker):
    # Mock load_data_once to return a minimal DataFrame
    dummy_df = pd.DataFrame({
        'feature1': np.random.rand(20),
        config.TARGET_COLUMN_NAME: np.random.randint(0, 2, 20)
    }, index=pd.date_range('2023-01-01', periods=20, freq='min'))
    mocker.patch('btc_usdt_pipeline.optimize.auto_optimize.load_data_once', return_value=dummy_df)
    # Mock config features
    mocker.patch.object(config, 'ALL_TREE_FEATURES', ['feature1'])

    # Mock optuna trial
    mock_trial = mocker.Mock()
    mock_trial.suggest_categorical.return_value = "RandomForest" # Force RF
    mock_trial.suggest_int.side_effect = [10, 5, 2, 1] # n_estimators, max_depth, min_split, min_leaf
    mock_trial.number = 0
    mock_trial.params = {'model_type': 'RandomForest', 'rf_n_estimators': 10} # Example params

    # Mock the model fitting and prediction
    mocker.patch('sklearn.ensemble.RandomForestClassifier.fit')
    mocker.patch('sklearn.ensemble.RandomForestClassifier.predict_proba', return_value=np.array([[0.6, 0.4]] * 6)) # Dummy probabilities for validation set (20 * 0.3 = 6)
    mocker.patch('sklearn.ensemble.RandomForestClassifier.predict', return_value=np.array([0] * 6))

    result = objective(mock_trial)
    assert isinstance(result, float)
    assert result != float('inf') # Should not fail with basic setup
    mock_trial.set_user_attr.assert_called() # Check if accuracy was logged

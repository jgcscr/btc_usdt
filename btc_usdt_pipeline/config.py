# config.py
"""Central configuration file for the BTC/USDT trading pipeline."""

import os
import logging
from typing import List
from pathlib import Path # Import Path
from btc_usdt_pipeline.utils.config_manager import ConfigManager
from btc_usdt_pipeline.utils.colab_utils import is_colab

# --- Paths ---
if is_colab():
    DRIVE_BASE_PATH_STR = os.getenv('DRIVE_BASE_PATH', '/content/drive/My Drive/btc_usdt_trading_pipeline')
else:
    DRIVE_BASE_PATH_STR = os.getenv('DRIVE_BASE_PATH', '/workspaces/btc_usdt')
BASE_DIR = Path(DRIVE_BASE_PATH_STR)

DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
LOGS_DIR = BASE_DIR / 'logs'

# NOTE: Directory creation (e.g., LOGS_DIR.mkdir(parents=True, exist_ok=True))
# should now happen where the directories are first needed,
# for example, within the setup_logger function for LOGS_DIR,
# or before saving data/models in respective functions.
# Removing automatic creation from here:
# for path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
#     os.makedirs(path, exist_ok=True) # Removed

# Specific file paths using pathlib
RAW_DATA_PATH = DATA_DIR / '1m_btcusdt_raw.parquet'
ENRICHED_DATA_PATH = DATA_DIR / '1m_btcusdt_enriched.parquet'
AUTO_OPTIMIZE_RESULTS_PATH = RESULTS_DIR / 'auto_optimize_results.json'
INDICATOR_OPTIMIZE_RESULTS_PATH = RESULTS_DIR / 'optimize_indicators_results.json'
MODEL_PREDICTIONS_PATH = RESULTS_DIR / 'model_predictions.json'

# Model file names (can remain strings, paths constructed when saving/loading)
RF_MODEL_NAME = 'rf_model.joblib'
XGB_MODEL_NAME = 'xgb_model.joblib'
LSTM_MODEL_NAME = 'lstm_model.h5'
GRU_MODEL_NAME = 'gru_model.h5'
LGBM_MODEL_NAME = 'lgbm_model.joblib'

# Log file paths (Removed specific paths - handled by logger setup)
# Example: The setup_logger in helpers.py will now construct the full path,
# e.g., LOGS_DIR / 'fetch_data.log'
# FETCH_DATA_LOG = os.path.join(LOGS_DIR, 'fetch_data.log') # Removed
# COMPUTE_FEATURES_LOG = os.path.join(LOGS_DIR, 'compute_features.log') # Removed
# ... and so on for other log files ...
UTILS_LOG_NAME = 'utils.log' # Keep base names if needed by setup_logger

# --- Data Fetching ---
SYMBOL = os.getenv('SYMBOL', 'BTCUSDT')
INTERVAL = os.getenv('INTERVAL', '1m')
BINANCE_API_URL = os.getenv('API_URL', 'https://api.binance.us/api/v3/klines') # Use .com or appropriate endpoint
FETCH_LIMIT = 1000  # Binance max per request
FETCH_DAYS = int(os.getenv('DAYS', 365)) # Days of data to fetch initially if no file exists
FETCH_DELAY_SECONDS = 0.25 # Delay between API calls

# --- Feature Engineering ---
# Largest lookback window needed for any indicator calculation
FEATURE_LOOKBACK = 200
# Higher Timeframes (HTF)
HTF_RULES = {'5T': '5m', '30T': '30m', '1H': '1h', '4H': '4h'}
HTF_MIN_ROWS = 20 # Min rows needed to compute HTF features

# List of features used by models (ensure consistency)
# Base 1m features
FEATURES_1M: List[str] = [
    'ema_9', 'ema_20', 'ema_50', 'ema_100', 'ema_200', 'sma_10', 'sma_20', 'sma_50',
    'rsi_14', 'rsi_7', 'macd', 'macd_signal', 'macd_diff', 'stoch_k', 'stoch_d',
    'cci_20', 'willr_14', 'roc_1', 'roc_5', 'roc_15', 'obv', 'bb_upper', 'bb_middle',
    'bb_lower', 'atr_14', 'adx_14', 'psar', 'fractal_high', 'fractal_low'
]
# HTF features (dynamically generated names in compute_features)
FEATURES_HTF_INDICATORS: List[str] = ['ema_20', 'rsi_14', 'macd', 'bb_upper', 'bb_lower', 'atr_14', 'adx_14']
FEATURES_HTF: List[str] = [
    f'{prefix}_{ind}'
    for prefix in HTF_RULES.values()
    for ind in FEATURES_HTF_INDICATORS
]
# All features used for tree models
ALL_TREE_FEATURES: List[str] = FEATURES_1M + FEATURES_HTF
# Subset of features for sequence models (example: first 15 of 1m)
SEQUENCE_FEATURES: List[str] = FEATURES_1M[:15]
SEQUENCE_TIMESTEPS: int = 60

# --- Model Training & Target Definition ---
TARGET_COLUMN_NAME = 'target_move' # Name for the generated target column
TARGET_FUTURE_WINDOW = 60 # Bars into the future to predict move
TARGET_THRESHOLD_USD = 500 # Minimum absolute price move to be considered positive target

# Training parameters (examples)
RF_N_ESTIMATORS = 100
XGB_N_ESTIMATORS = 100
LSTM_UNITS = 50 # Adjusted from 32
GRU_UNITS = 50 # Adjusted from 32
LGBM_N_ESTIMATORS = 100
TRAIN_EPOCHS = 10 # Default epochs if sequence-specific not set
TRAIN_BATCH_SIZE = 64 # Default batch size if sequence-specific not set
SEQUENCE_EPOCHS = 20 # Epochs specifically for LSTM/GRU
SEQUENCE_BATCH_SIZE = 128 # Batch size specifically for LSTM/GRU
EARLY_STOPPING_PATIENCE = 3 # Increased patience slightly

# Data splitting fractions
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
# TEST_FRAC is implicitly 1 - TRAIN_FRAC - VAL_FRAC
RANDOM_STATE = 42 # Added for reproducibility if not already present

# --- Backtesting & Signals ---
INITIAL_EQUITY = 10000
SIGNAL_THRESHOLD = 0.5 # Threshold for multi_timeframe_signal logic (Adjust based on optimization/analysis)
PROBABILITY_THRESHOLD = 0.55 # Min prediction probability to consider a signal (Adjust based on optimization/analysis)
ATR_STOP_LOSS_MULTIPLIER = 1.5 # Example: Tighter SL
ATR_TAKE_PROFIT_MULTIPLIER = 3.0 # Example: Wider TP
BACKTEST_ATR_COLUMN = 'atr_14' # Which ATR to use for SL/TP (Ensure this is calculated in features)
COMMISSION_RATE = 0.00075 # Example commission rate (e.g., 0.075% per trade)
SLIPPAGE_POINTS = 2 # Example slippage in price points (e.g., $2 for BTC)
RISK_FRACTION = 0.01 # Example: Risk 1% of equity per trade

# --- Optimization ---
OPTUNA_N_TRIALS_AUTO = 50 # Trials for auto_optimize
OPTUNA_N_TRIALS_INDICATORS = 30 # Trials for optimize_indicators
# Adjust based on Colab/environment resources. -1 uses all cores, but can cause issues on resource-constrained environments.
OPTUNA_N_JOBS = 1 # Defaulting to 1 for broader compatibility, adjust as needed.

# --- Execution ---
LIVE_TRADING = False # Set to True to enable actual execution calls (requires broker integration)

# --- Logging ---
LOG_LEVEL = "INFO" # e.g., "DEBUG", "INFO", "WARNING", "ERROR"
LOG_FORMAT = '%(asctime)s %(levelname)-8s [%(name)s] [%(filename)s:%(lineno)d] %(message)s' # Added logger name
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# At the end of the config file, update paths for Colab if needed
ConfigManager(locals()).update_paths_for_colab()

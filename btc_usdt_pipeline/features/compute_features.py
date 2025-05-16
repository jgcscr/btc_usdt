# features/compute_features.py
"""Computes technical indicators and features."""

import pandas as pd
import numpy as np
import os
from pathlib import Path

# Add compatibility for pandas-ta with NumPy 2.0+
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

import pandas_ta as ta # type: ignore
from typing import List, Dict

from btc_usdt_pipeline import config
from btc_usdt_pipeline.utils.logging_config import setup_logging
from btc_usdt_pipeline.utils.data_processing import optimize_dataframe_dtypes, preprocess_data
from btc_usdt_pipeline.io.data_manager import DataManager
from btc_usdt_pipeline.utils.data_quality import detect_missing_data
from btc_usdt_pipeline.utils.config_manager import config_manager
from btc_usdt_pipeline.utils.helpers import make_binary_target

logger = setup_logging(log_filename='compute_features.log')

def standardize_column_names(df):
    """Standardize indicator column names to match config.FEATURES_1M format"""
    rename_map = {
        # Moving Averages
        'EMA_9': 'ema_9',
        'EMA_20': 'ema_20', 
        'EMA_50': 'ema_50',
        'EMA_100': 'ema_100',
        'EMA_200': 'ema_200',
        'SMA_10': 'sma_10',
        'SMA_20': 'sma_20',
        'SMA_50': 'sma_50',
        # Oscillators
        'RSI_2': 'rsi_2',
        'RSI_7': 'rsi_7',
        'RSI_14': 'rsi_14',
        'MACD_12_26_9': 'macd',
        'MACDs_12_26_9': 'macd_signal',
        'MACDh_12_26_9': 'macd_diff',
        'STOCHk_14_3_3': 'stoch_k',
        'STOCHd_14_3_3': 'stoch_d',
        'CCI_20_0.015': 'cci_20',
        'WILLR_14': 'willr_14',
        # Momentum
        'ROC_1': 'roc_1',
        'ROC_5': 'roc_5',
        'ROC_15': 'roc_15',
        # Volume
        'OBV': 'obv',
        'volume_osc': 'volume_osc',
        # Volatility
        'BBU_20_2.0': 'bb_upper',
        'BBM_20_2.0': 'bb_middle',
        'BBL_20_2.0': 'bb_lower',
        'ATRr_14': 'atr_14',
        # Trend
        'ADX_14': 'adx_14',
        'PSARl_0.02_0.2': 'psar',
        # Others
        'fractal_high': 'fractal_high',
        'fractal_low': 'fractal_low'
    }
    # Only rename columns that exist in the dataframe
    cols_to_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    return df.rename(columns=cols_to_rename)

# Helper function to calculate features for a given timeframe df
def calculate_indicators(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    """Calculates a standard set of indicators using pandas_ta."""
    logger.debug(f"Calculating indicators with prefix: '{prefix}'")
    # Ensure columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        logger.error(f"Missing required columns for indicator calculation: {missing}")
        raise ValueError(f"Missing required columns: {missing}")

    # Use pandas_ta strategy for multiple indicators
    # Adjust strategy based on config.FEATURES_1M or FEATURES_HTF_INDICATORS
    # For simplicity, calculate a broad set and select later if needed
    custom_strategy = ta.Strategy(
        name="Custom Indicators",
        description="Common TA indicators",
        ta=[
            # Moving Averages
            {"kind": "ema", "length": 9}, {"kind": "ema", "length": 20}, {"kind": "ema", "length": 50}, {"kind": "ema", "length": 100}, {"kind": "ema", "length": 200},
            {"kind": "sma", "length": 10}, {"kind": "sma", "length": 20}, {"kind": "sma", "length": 50},
            # Oscillators
            {"kind": "rsi", "length": 2}, {"kind": "rsi", "length": 7}, {"kind": "rsi", "length": 14},
            {"kind": "macd"}, # Uses default lengths (12, 26, 9)
            {"kind": "stoch"}, # Uses default lengths (14, 3, 3)
            {"kind": "cci", "length": 20},
            {"kind": "willr", "length": 14},
            # Momentum
            {"kind": "roc", "length": 1}, {"kind": "roc", "length": 5}, {"kind": "roc", "length": 15},
            # Volume
            {"kind": "obv"},
            # Volatility
            {"kind": "bbands", "length": 20}, # Bollinger Bands
            {"kind": "atr", "length": 14}, # Average True Range
            # Trend
            {"kind": "adx", "length": 14}, # ADX
            {"kind": "psar"}, # Parabolic SAR
        ]
    )
    try:
        df.ta.strategy(custom_strategy)
        # Volume Oscillator: difference between SMA(volume, 14) and SMA(volume, 28)
        df['vol_sma_14'] = df['volume'].rolling(14).mean()
        df['vol_sma_28'] = df['volume'].rolling(28).mean()
        df['volume_osc'] = df['vol_sma_14'] - df['vol_sma_28']
    except Exception as e:
        logger.error(f"Error applying pandas_ta strategy: {e}")
        # Depending on the error, might return df or raise
        raise

    # Add Fractals (simple high/low over 5 periods)
    # Ensure index is sorted if not already
    df_sorted = df.sort_index()
    df['fractal_high'] = (df_sorted['high'].rolling(5, center=True).max() == df_sorted['high']).astype(int)
    df['fractal_low'] = (df_sorted['low'].rolling(5, center=True).min() == df_sorted['low']).astype(int)

    # Rename columns with prefix if provided
    if prefix:
        rename_map = {col: f"{prefix}_{col}" for col in df.columns if col not in required_cols and col not in ['fractal_high', 'fractal_low']}
        # Handle specific indicator names generated by pandas_ta (e.g., MACD_12_26_9)
        for col in list(rename_map.keys()): # Iterate over a copy of keys
            if col.startswith('MACD_') or col.startswith('STOCH_') or col.startswith('BBL_') or col.startswith('BBM_') or col.startswith('BBU_') or col.startswith('PSAR'):
                 new_name = f"{prefix}_{col.lower().replace('_', '-')}" # Example: 5m_macd-12-26-9
                 # Or simplify: just prefix the original name
                 new_name_simple = f"{prefix}_{col}"
                 rename_map[col] = new_name_simple # Use simpler naming for consistency with config
            elif col == 'fractal_high':
                 rename_map[col] = f"{prefix}_fractal_high"
            elif col == 'fractal_low':
                 rename_map[col] = f"{prefix}_fractal_low"

        df = df.rename(columns=rename_map)

    # Standardize column names to match config expectations
    df = standardize_column_names(df)
    return df

def compute_htf_features(df: pd.DataFrame, rules: Dict[str, str] = config.HTF_RULES) -> pd.DataFrame:
    """Computes features on higher timeframes and merges them back."""
    logger.info("Computing Higher Timeframe (HTF) features...")
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'open_time' in df.columns:
            logger.warning("'open_time' column found, setting as index.")
            df['open_time'] = pd.to_datetime(df['open_time'])
            df = df.set_index('open_time')
        else:
            logger.error("DataFrame index is not DatetimeIndex and 'open_time' column not found.")
            raise ValueError("DataFrame must have a DatetimeIndex or 'open_time' column for HTF calculation.")

    original_cols = df.columns.tolist()
    htf_features_list = []

    for rule, prefix in rules.items():
        logger.debug(f"Resampling to {rule} ({prefix})...")
        try:
            # Resample OHLCV
            resampled_df = df.resample(rule).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            resampled_df = resampled_df.dropna() # Drop intervals with no trades

            if len(resampled_df) < config.HTF_MIN_ROWS:
                logger.warning(f"Skipping {rule} ({prefix}) due to insufficient data after resampling ({len(resampled_df)} rows < {config.HTF_MIN_ROWS})")
                continue

            # Calculate indicators on the resampled data
            htf_indicators = calculate_indicators(resampled_df.copy(), prefix=prefix)

            # Select only the newly added indicator columns (prefixed)
            htf_indicator_cols = [col for col in htf_indicators.columns if col.startswith(prefix)]
            htf_features_list.append(htf_indicators[htf_indicator_cols])

        except Exception as e:
            logger.error(f"Error processing HTF rule {rule} ({prefix}): {e}")
            # Continue with other rules if one fails
            continue

    if not htf_features_list:
        logger.warning("No HTF features were successfully computed.")
        return df # Return original df if no HTF features

    # Merge HTF features back to the original 1m dataframe
    logger.info("Merging HTF features back to original dataframe...")
    merged_df = df.reset_index()
    for htf_df in htf_features_list:
        htf_df = htf_df.reset_index()
        print("Original DF columns:", merged_df.columns.tolist())
        print("HTF DF columns:", htf_df.columns.tolist())
        merged_df = pd.merge_asof(
            merged_df.sort_values('index'),
            htf_df.sort_values('index'),
            on='index',
            direction='backward'
        )
    merged_df = merged_df.set_index('index')
    # Forward fill any remaining NaNs in HTF columns after the first available value
    htf_added_cols = [col for col in merged_df.columns if any(col.startswith(prefix) for prefix in rules.values())]
    merged_df[htf_added_cols] = merged_df[htf_added_cols].ffill()
    logger.info(f"Added {len(htf_added_cols)} HTF feature columns.")
    return merged_df

def main(input_file=None, output_file=None, sample=False):
    """Main function to load data, compute features, and save.
    
    Args:
        input_file (str, optional): Path to input data file. Defaults to None (uses config).
        output_file (str, optional): Path to output features file. Defaults to None (uses config).
        sample (bool, optional): Whether to use a sample of data. Defaults to False.
    """
    logger.info("Starting feature computation process...")
    raw_data_path = input_file if input_file is not None else config_manager.get('data.raw_data_path')
    enriched_data_path = output_file if output_file is not None else config_manager.get('data.enriched_data_path')
    
    # Convert string paths to Path objects if they're not already
    if isinstance(raw_data_path, str):
        raw_data_path = Path(raw_data_path)
    if isinstance(enriched_data_path, str):
        enriched_data_path = Path(enriched_data_path)

    logger.info(f"Using input data from: {raw_data_path}")
    logger.info(f"Will save output data to: {enriched_data_path}")

    # Ensure results directory exists
    try:
        os.makedirs(os.path.dirname(enriched_data_path), exist_ok=True)
        logger.debug(f"Ensured results directory exists: {os.path.dirname(enriched_data_path)}")
    except Exception as e:
        logger.error(f"Could not create results directory {os.path.dirname(enriched_data_path)}: {e}")
        return # Cannot proceed without results directory

    # Load raw data
    logger.info(f"Loading raw data from {raw_data_path}...")
    logger.info("Note: For very large datasets on memory-constrained environments (like Colab), consider loading data in chunks or using memory-efficient dtypes.")
    try:
        df = DataManager().load_data(raw_data_path, file_type='parquet', use_cache=True)
        if df is None:
            logger.error(f"Raw data file not found at {raw_data_path}. Run data fetching first.")
            return
        # Remove manual optimize_dataframe_dtypes, now handled by DataManager
        if 'open_time' not in df.columns:
            logger.error("Raw data file missing 'open_time' column.")
            return
        df['open_time'] = pd.to_datetime(df['open_time'])
        df = df.set_index('open_time')
        df = preprocess_data(df, sort_by=None)  # Already sorted by open_time after set_index

        # Process sample data if sample flag is set
        if sample and df is not None:
            sample_size = min(1000, len(df))  # Use 1000 rows or entire dataset if smaller
            df = df.iloc[:sample_size].copy()
            logger.info(f"Using sample data: {sample_size} rows")

        logger.info(f"Loaded {len(df)} raw data records.")
    except Exception as e:
        logger.error(f"Error loading raw data from {raw_data_path}: {e}")
        return

    # Calculate base 1m indicators
    logger.info("Calculating base 1m indicators...")
    try:
        df = calculate_indicators(df.copy()) # Pass copy to avoid modifying original df inplace during calculation
        # Select only the features defined in config to keep df clean
        base_cols_to_keep = ['open', 'high', 'low', 'close', 'volume'] + config_manager.get('data.features_1m', [])
        # Filter df to keep only necessary base columns + newly calculated 1m features
        # Handle potential missing columns gracefully
        missing_1m_features = [f for f in config.FEATURES_1M if f not in df.columns]
        if missing_1m_features:
            logger.warning(f"The following 1m features specified in config were not generated: {missing_1m_features}")
        actual_1m_features = [f for f in config.FEATURES_1M if f in df.columns]
        df = df[['open', 'high', 'low', 'close', 'volume'] + actual_1m_features]

    except Exception as e:
        logger.error(f"Error calculating base 1m indicators: {e}")
        return

    # Calculate HTF features
    try:
        df = compute_htf_features(df)
    except Exception as e:
        logger.error(f"Error computing HTF features: {e}")
        # Decide if you want to proceed without HTF features or stop
        logger.warning("Proceeding without HTF features due to error.")
        # return # Uncomment this to stop if HTF features are critical

    # Create binary target variable
    logger.info("Creating binary target variable...")
    try:
        df = make_binary_target(
            df, # Pass the dataframe
            future_window=config.TARGET_FUTURE_WINDOW,
            threshold_usd=config.TARGET_THRESHOLD_USD,
            target_col_name=config.TARGET_COLUMN_NAME
        )
    except Exception as e:
        logger.error(f"Error creating binary target: {e}")
        return

    # Drop initial rows with NaNs resulting from lookback periods
    # Use a larger drop based on FEATURE_LOOKBACK + TARGET_FUTURE_WINDOW for safety
    initial_len = len(df)
    required_lookback = max(config.FEATURE_LOOKBACK, config.TARGET_FUTURE_WINDOW) # Consider both feature and target lookbacks
    # Drop NaNs based on all feature columns and the target
    all_feature_cols = config.ALL_TREE_FEATURES + [config.TARGET_COLUMN_NAME]
    # Ensure only existing columns are used for dropping NaNs
    cols_to_check_for_nan = [col for col in all_feature_cols if col in df.columns]
    df = df.dropna(subset=cols_to_check_for_nan)
    # Alternatively, drop first N rows if confident about lookback calculation:
    # df = df.iloc[required_lookback:]
    logger.info(f"Dropped {initial_len - len(df)} rows due to NaN values after feature/target calculation.")

    # --- Data Quality Check ---
    quality_report = detect_missing_data(df, critical_cols=cols_to_check_for_nan)
    if quality_report['total_missing'] > 0:
        logger.warning(f"Data quality issue: {quality_report}")

    if df.empty:
        logger.error("DataFrame is empty after calculating features and dropping NaNs. Cannot save.")
        return

    # Save enriched data
    logger.info(f"Saving enriched data to {enriched_data_path}...")
    try:
        df.reset_index().to_parquet(enriched_data_path, index=False) # Save with open_time as column
        logger.info(f"Successfully saved {len(df)} enriched data records.")
    except Exception as e:
        logger.error(f"Error saving enriched data to {enriched_data_path}: {e}")

    logger.info("Feature computation process finished.")

if __name__ == '__main__':
    main()

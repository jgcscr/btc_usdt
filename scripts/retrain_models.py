"""
retrain_models.py
Entry point script to retrain models using the btc_usdt_pipeline package.
Refactored to use centralized config and helpers.
Run as: python -m scripts.retrain_models
"""
# Removed sys.path manipulation
import pandas as pd
import numpy as np

from btc_usdt_pipeline import config
from btc_usdt_pipeline.utils.helpers import split_data, setup_logger, make_binary_target, create_sequences
from btc_usdt_pipeline.models.train import (
    train_random_forest,
    train_xgboost,
    train_lstm,
    train_gru
    # Add train_lgbm if you implement it in models/train.py
)

# Setup logger using the centralized configuration
logger = setup_logger('retrain_models_script.log') # Use a script-specific log

# Features and sequence features are now imported from config
FEATURES = config.ALL_TREE_FEATURES
SEQ_FEATURES = config.SEQUENCE_FEATURES
TARGET = config.TARGET_COLUMN_NAME

def main():
    logger.info('--- Starting Model Retraining Script ---')
    logger.info(f'Loading enriched data from: {config.ENRICHED_DATA_PATH}')
    # Add note about memory usage
    logger.info("Note: For very large datasets on memory-constrained environments (like Colab), consider loading data in chunks or using memory-efficient dtypes.")
    try:
        df = pd.read_parquet(config.ENRICHED_DATA_PATH)
        # Cast numeric columns to memory-efficient types
        for col in df.select_dtypes(include=['float64', 'float']).columns:
            df[col] = df[col].astype('float32')
        for col in df.select_dtypes(include=['int64', 'int']).columns:
            df[col] = df[col].astype('int32')
        logger.info(f'Enriched data loaded: {df.shape[0]} rows')
    except FileNotFoundError:
        logger.error(f"Enriched data file not found at {config.ENRICHED_DATA_PATH}. Run feature computation first.")
        print(f"Error: Enriched data file not found at {config.ENRICHED_DATA_PATH}")
        return
    except Exception as e:
        logger.error(f"Error loading enriched data: {e}")
        print(f"Error loading enriched data: {e}")
        return

    # Ensure index is datetime for potential time-based operations if needed later
    if 'open_time' in df.columns:
        df['open_time'] = pd.to_datetime(df['open_time'])
        df = df.set_index('open_time')

    logger.info(f'Creating binary target \'{TARGET}\'...')
    # Use the helper function for target creation
    df = make_binary_target(df, future_window=config.TARGET_FUTURE_WINDOW, threshold_usd=config.TARGET_THRESHOLD_USD, target_col_name=TARGET) # Pass params

    # Drop rows with NaN in features or target before splitting
    # Ensure all features used by any model are included here
    all_features_needed = list(set(FEATURES + SEQ_FEATURES))
    df = df.dropna(subset=all_features_needed + [TARGET])
    logger.info(f'Data shape after NaN drop: {df.shape}')

    if df.empty:
        logger.error("DataFrame is empty after dropping NaNs. Cannot proceed with training.")
        print("Error: No data available for training after preprocessing.")
        return

    logger.info('Splitting data...')
    # Use the helper function for splitting
    train_df, val_df, test_df = split_data(df, train_frac=config.TRAIN_FRAC, val_frac=config.VAL_FRAC)
    logger.info(f'Train: {train_df.shape[0]}, Val: {val_df.shape[0]}, Test: {test_df.shape[0]}')

    # --- Tree Model Training ---
    logger.info('Preparing data for tree models...')
    X_train_tree = train_df[FEATURES]
    y_train_tree = train_df[TARGET]
    X_val_tree = val_df[FEATURES]
    y_val_tree = val_df[TARGET]

    if X_train_tree.empty or y_train_tree.empty:
        logger.error("Training data for tree models is empty. Skipping tree model training.")
    else:
        logger.info(f'X_train (tree): {X_train_tree.shape}, y_train (tree): {y_train_tree.shape}')
        try:
            # Ensure training functions handle model saving and directory creation
            train_random_forest(X_train_tree, y_train_tree)
            train_xgboost(X_train_tree, y_train_tree, X_val_tree, y_val_tree)
            # Add call to train_lgbm if implemented
        except Exception as e:
            logger.error(f"Error during tree model training: {e}", exc_info=True)
            print(f"Error during tree model training: {e}")

    # --- Sequence Model Training ---
    logger.info('Preparing data for sequence models...')
    # Use the helper function for sequence creation
    X_train_seq_data = train_df[SEQ_FEATURES].values
    y_train_seq_target = train_df[TARGET].values
    X_val_seq_data = val_df[SEQ_FEATURES].values
    y_val_seq_target = val_df[TARGET].values

    X_train_seq, y_train_seq = create_sequences(X_train_seq_data, y_train_seq_target, timesteps=config.SEQUENCE_TIMESTEPS)
    X_val_seq, y_val_seq = create_sequences(X_val_seq_data, y_val_seq_target, timesteps=config.SEQUENCE_TIMESTEPS)

    if X_train_seq.size == 0 or y_train_seq.size == 0 or X_val_seq.size == 0 or y_val_seq.size == 0:
        logger.error("Training or validation data for sequence models is empty. Skipping sequence model training.")
    else:
        logger.info(f'X_train_seq: {X_train_seq.shape}, y_train_seq: {y_train_seq.shape}')
        logger.info(f'X_val_seq: {X_val_seq.shape}, y_val_seq: {y_val_seq.shape}')
        try:
            # Ensure training functions handle model saving and directory creation
            train_lstm(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
            train_gru(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
        except ImportError as e:
             logger.error(f"ImportError during sequence model training: {e}. Is TensorFlow installed?")
             print(f"ImportError: {e}. Please ensure TensorFlow is installed (`pip install tensorflow`).")
        except Exception as e:
            logger.error(f"Error during sequence model training: {e}", exc_info=True)
            print(f"Error during sequence model training: {e}")

    logger.info('--- Model Retraining Script Complete. Models saved to %s ---', config.MODELS_DIR)

if __name__ == '__main__':
    main()

"""
train_additional_models.py
Entry point script to train additional models (e.g., LightGBM) using the btc_usdt_pipeline package.
Refactored for modularity and use of central configuration.
Run as: python -m scripts.train_additional_models
"""
import pandas as pd
import numpy as np
import joblib
import subprocess
import sys  # Keep sys for subprocess call
import os

# Use absolute imports from the package
from btc_usdt_pipeline import config
from btc_usdt_pipeline.utils.helpers import split_data, setup_logger, make_binary_target
from btc_usdt_pipeline.utils.data_processing import optimize_memory_usage

# Setup logger using the centralized configuration
logger = setup_logger('train_additional_models.log')  # Changed log filename for clarity

# Optional: install lightgbm if not already installed
try:
    import lightgbm as lgb
except ImportError:
    logger.warning("lightgbm not found, attempting to install...")
    try:
        # Ensure pip is available and use it to install
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'lightgbm'])
        import lightgbm as lgb
        logger.info("lightgbm installed successfully.")
    except Exception as e:
        logger.error(f"Failed to install lightgbm: {e}")
        lgb = None

# Features and target are now imported from config
features = config.ALL_TREE_FEATURES
target = config.TARGET_COLUMN_NAME

def load_and_prepare_data(data_path):
    logger.info(f"Loading data from {data_path}...")
    # Add note about memory usage
    logger.info("Note: For very large datasets on memory-constrained environments (like Colab), consider loading data in chunks or using memory-efficient dtypes.")
    try:
        df = pd.read_parquet(data_path)
        df = optimize_memory_usage(df, logger=logger)
    except FileNotFoundError:
        logger.error(f"Data file not found at {data_path}. Run feature computation first.")
        return None
    except Exception as e:
        logger.error(f"Error loading data from {data_path}: {e}")
        return None

    logger.info("Creating binary target...")
    # Use the helper function for target creation
    df = make_binary_target(df, future_window=config.TARGET_FUTURE_WINDOW, threshold_usd=config.TARGET_THRESHOLD_USD, target_col_name=target)

    # Drop rows with NaN in features or target
    df = df.dropna(subset=features + [target])
    logger.info(f"Data prepared. Shape: {df.shape}")
    return df

def train_lightgbm(X_train, y_train, X_val, y_val):
    if lgb is None:
        logger.error("LightGBM is not available. Skipping training.")
        return None

    logger.info("Training LightGBM classifier...")
    lgbm = lgb.LGBMClassifier(
        n_estimators=config.LGBM_N_ESTIMATORS,
        n_jobs=-1,
        random_state=42  # Added random_state
    )
    # Ensure y_train and y_val are integer type
    y_train_lgb = y_train.astype(int)
    y_val_lgb = y_val.astype(int)

    lgbm.fit(X_train, y_train_lgb,
             eval_set=[(X_val, y_val_lgb)],
             callbacks=[lgb.early_stopping(config.EARLY_STOPPING_PATIENCE, verbose=False)],  # Use config patience
             eval_metric='logloss'  # Specify eval_metric
            )
    logger.info("LightGBM training complete.")
    return lgbm

def save_model(model, model_name, models_dir):
    if model is not None:
        # Ensure directory exists before saving
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, model_name)
        try:
            joblib.dump(model, model_path)
            logger.info(f"Model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model {model_name} to {model_path}: {e}")
    else:
        logger.warning(f"Model {model_name} was not trained or failed, skipping save.")

def main():
    logger.info("--- Starting Additional Model Training (LightGBM) ---")
    df = load_and_prepare_data(config.ENRICHED_DATA_PATH)

    if df is None or df.empty:
        logger.error("Failed to load or prepare data. Aborting training.")
        return

    train_df, val_df, test_df = split_data(df, train_frac=config.TRAIN_FRAC, val_frac=config.VAL_FRAC)

    X_train = train_df[features]
    y_train = train_df[target]
    X_val = val_df[features]
    y_val = val_df[target]

    if X_train.empty or y_train.empty:
        logger.error("Training data is empty. Skipping LightGBM training.")
    else:
        # Train and save LightGBM
        lgbm_model = train_lightgbm(X_train, y_train, X_val, y_val)
        save_model(lgbm_model, config.LGBM_MODEL_NAME, config.MODELS_DIR)

    # --- Add calls to train other models here ---
    # Example:
    # logger.info("Training CatBoost model...")
    # catboost_model = train_catboost(X_train, y_train, X_val, y_val) # Define train_catboost function
    # save_model(catboost_model, 'catboost_model.joblib', config.MODELS_DIR)

    logger.info("--- Additional Model Training Finished ---")

if __name__ == '__main__':
    main()

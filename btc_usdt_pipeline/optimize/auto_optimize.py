# optimize/auto_optimize.py
"""Uses Optuna to optimize model hyperparameters and potentially feature subsets."""

import optuna
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split # Using simple split for optimization trials
from sklearn.metrics import accuracy_score, log_loss
import psutil

# Import necessary models and helpers
from btc_usdt_pipeline import config
from btc_usdt_pipeline.utils.helpers import setup_logger, make_binary_target
from btc_usdt_pipeline.models.train import train_random_forest, train_xgboost # Assuming these can be adapted or new ones created for optuna
from btc_usdt_pipeline.utils.data_manager import DataManager
from btc_usdt_pipeline.utils.colab_utils import check_memory_usage, save_checkpoint, memory_safe
from btc_usdt_pipeline.utils.data_processing import optimize_dataframe_dtypes
from btc_usdt_pipeline.io.serialization import load_json, save_json
# Need adaptable training functions or direct model instantiation here
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

logger = setup_logger('auto_optimize.log')

def load_data_once():
    """Loads and prepares data, using DataManager for caching and loading."""
    dm = DataManager()
    df = dm.load_data(config.ENRICHED_DATA_PATH, file_type='parquet', use_cache=True)
    if df is None:
        logger.error(f"Enriched data file not found at {config.ENRICHED_DATA_PATH}. Cannot run optimization.")
        return None

    # Optimize DataFrame dtypes for Colab/memory efficiency
    df = optimize_dataframe_dtypes(df)

    if 'open_time' in df.columns:
        df['open_time'] = pd.to_datetime(df['open_time'])
        df = df.set_index('open_time')

    # Ensure target exists or create it
    if config.TARGET_COLUMN_NAME not in df.columns:
        logger.warning(f"Target column '{config.TARGET_COLUMN_NAME}' not found. Creating it now.")
        df = make_binary_target(df, config.TARGET_FUTURE_WINDOW, config.TARGET_THRESHOLD_USD, config.TARGET_COLUMN_NAME)

    # Drop NaNs based on features and target
    features_to_use = config.ALL_TREE_FEATURES
    cols_to_check = features_to_use + [config.TARGET_COLUMN_NAME]
    df = df.dropna(subset=[col for col in cols_to_check if col in df.columns])

    if df.empty:
        logger.error("Data is empty after loading and cleaning. Cannot run optimization.")
        return None

    logger.info(f"Data loaded successfully for optimization. Shape: {df.shape}")
    return df

def get_safe_n_jobs(requested_n_jobs: int, min_free_gb: float = 2.0) -> int:
    """
    Returns a safe n_jobs value based on available RAM.
    If available RAM is below min_free_gb, returns 1 (serial execution).
    Otherwise, returns requested_n_jobs.
    """
    mem = psutil.virtual_memory()
    avail_gb = mem.available / 1024**3
    if avail_gb < min_free_gb:
        logger.warning(f"Low available RAM ({avail_gb:.2f} GB). Forcing n_jobs=1 to avoid OOM.")
        return 1
    return requested_n_jobs

def objective(trial: optuna.Trial) -> float:
    """Optuna objective function to train and evaluate a model."""
    df = load_data_once()
    if df is None:
        # Return a large value to indicate failure if data loading failed
        return float('inf')

    features = config.ALL_TREE_FEATURES
    target = config.TARGET_COLUMN_NAME

    # Chronological split to avoid lookahead bias
    n = len(df)
    train_end = int(n * config.TRAIN_FRAC)
    val_end = int(n * (config.TRAIN_FRAC + config.VAL_FRAC))
    X_train = df[features].iloc[:train_end]
    y_train = df[target].iloc[:train_end]
    X_val = df[features].iloc[train_end:val_end]
    y_val = df[target].iloc[train_end:val_end]

    if X_train.empty or X_val.empty:
        logger.warning(f"Empty train or validation set in trial {trial.number}. Skipping.")
        return float('inf')

    # --- Hyperparameter Suggestion ---
    model_type = trial.suggest_categorical("model_type", ["RandomForest", "XGBoost"])

    if model_type == "RandomForest":
        n_estimators = trial.suggest_int("rf_n_estimators", 50, 300)
        max_depth = trial.suggest_int("rf_max_depth", 5, 50)
        min_samples_split = trial.suggest_int("rf_min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("rf_min_samples_leaf", 1, 10)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=config.RANDOM_STATE,
            n_jobs=1 # Limit jobs within trial to avoid oversubscribing CPUs with parallel trials
        )
    elif model_type == "XGBoost":
        n_estimators = trial.suggest_int("xgb_n_estimators", 50, 300)
        max_depth = trial.suggest_int("xgb_max_depth", 3, 10)
        learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 0.3, log=True)
        subsample = trial.suggest_float("xgb_subsample", 0.5, 1.0)
        colsample_bytree = trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0)

        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=config.RANDOM_STATE,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=1 # Limit jobs within trial
        )
    else:
        # Should not happen with categorical suggestion
        return float('inf')

    # --- Train and Evaluate ---
    try:
        # Note: XGBoost early stopping is harder to integrate directly here unless
        # you use the functional API or a more complex setup.
        # For simplicity, we fit without it in this basic optimization loop.
        model.fit(X_train, y_train)
        preds_proba = model.predict_proba(X_val)[:, 1]
        # Use log loss as the objective to minimize (lower is better)
        loss = log_loss(y_val, preds_proba)
        # Accuracy can be logged as a user attribute
        preds_binary = model.predict(X_val)
        accuracy = accuracy_score(y_val, preds_binary)
        trial.set_user_attr("accuracy", accuracy)
        logger.debug(f"Trial {trial.number} ({model_type}): Loss={loss:.4f}, Acc={accuracy:.4f}, Params={trial.params}")

    except Exception as e:
        logger.error(f"Error during training/evaluation in trial {trial.number}: {e}")
        loss = float('inf') # Penalize failed trials

    return loss # Optuna minimizes the objective function

@memory_safe(min_free_percent=10)
def main(n_trials=None, n_jobs=None):
    """
    Runs the Optuna optimization study with memory checks and checkpointing.
    Parallelization strategy:
    - n_jobs controls the number of parallel Optuna trials (default: config.OPTUNA_N_JOBS)
    - Each model is trained with n_jobs=1 to avoid nested parallelism
    - If available RAM is low, n_jobs is forced to 1 for safety
    """
    logger.info("--- Starting Auto Optimization --- ")

    # Load data once before starting study
    if load_data_once() is None:
        logger.error("Failed to load data. Aborting optimization.")
        return

    study_name = "btc_usdt_auto_optimize"
    study = optuna.create_study(direction="minimize")

    n_trials = n_trials or config.OPTUNA_N_TRIALS_AUTO
    requested_n_jobs = n_jobs if n_jobs is not None else getattr(config, 'OPTUNA_N_JOBS', 4)
    safe_n_jobs = get_safe_n_jobs(requested_n_jobs)
    checkpoint_dir = config.RESULTS_DIR / "optuna_checkpoints"
    checkpoint_interval = 10  # Save every 10 trials

    try:
        for i in range(0, n_trials, checkpoint_interval):
            remaining = min(checkpoint_interval, n_trials - i)
            logger.info(f"Running trials {i+1} to {i+remaining} with n_jobs={safe_n_jobs}...")
            study.optimize(
                objective,
                n_trials=remaining,
                n_jobs=safe_n_jobs
            )
            check_memory_usage()
            save_checkpoint(study, i+remaining, checkpoint_dir)
    except Exception as e:
        logger.error(f"Optimization study failed: {e}", exc_info=True)
        return

    logger.info(f"Optimization finished. Number of trials: {len(study.trials)}")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"  Value (Log Loss): {study.best_value:.4f}")
    logger.info(f"  Params: {study.best_params}")
    if 'accuracy' in study.best_trial.user_attrs:
        logger.info(f"  Accuracy: {study.best_trial.user_attrs['accuracy']:.4f}")

    # Save best parameters
    results = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'best_trial_number': study.best_trial.number,
        'user_attrs': study.best_trial.user_attrs
    }
    save_path = config.AUTO_OPTIMIZE_RESULTS_PATH
    try:
        save_json(results, save_path)
        logger.info(f"Optimization results saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save optimization results to {save_path}: {e}", exc_info=True)

    logger.info("--- Auto Optimization Finished --- ")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_trials', type=int, default=None, help='Number of Optuna trials')
    parser.add_argument('--n_jobs', type=int, default=None, help='Number of parallel Optuna jobs')
    args = parser.parse_args()
    main(n_trials=args.n_trials, n_jobs=args.n_jobs)

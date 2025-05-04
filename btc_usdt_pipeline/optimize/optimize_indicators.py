# optimize/optimize_indicators.py
"""Uses Optuna to find optimal parameters for technical indicators used in signals."""

import optuna
import pandas as pd
import numpy as np
from typing import Dict, Any

from btc_usdt_pipeline import config
from btc_usdt_pipeline.utils.helpers import setup_logger, load_json, save_json, calculate_metrics
from btc_usdt_pipeline.trading.signals import generate_signals # Assuming a simple signal function for optimization
from btc_usdt_pipeline.trading.backtest import run_backtest # Use the main backtester
from btc_usdt_pipeline.features.compute_features import calculate_indicators # Reuse indicator calculation
from btc_usdt_pipeline.utils.data_manager import DataManager
from btc_usdt_pipeline.utils.data_processing import optimize_dataframe_dtypes
from btc_usdt_pipeline.utils.colab_utils import memory_safe

logger = setup_logger('optimize_indicators.log')

def load_data_for_indicator_opt():
    """
    Loads and prepares data for indicator optimization.
    Uses DataManager with caching and memory-efficient dtypes for Colab compatibility.
    Ensures all required columns are present and aligns predictions with data.
    Returns:
        Tuple[pd.DataFrame, np.ndarray]: Cleaned DataFrame and ensemble probabilities.
    """
    dm = DataManager()
    logger.info("Loading data for indicator optimization...")
    try:
        # Load enriched data (needs base OHLCV + ATR for backtest SL/TP)
        df = dm.load_data(config.ENRICHED_DATA_PATH, file_type='parquet', use_cache=True)
        if df is None:
            logger.error(f"Enriched data file not found at {config.ENRICHED_DATA_PATH}. Cannot run optimization.")
            return None, None
        # Optimize DataFrame dtypes for Colab/memory efficiency
        df = optimize_dataframe_dtypes(df)
        if 'open_time' in df.columns:
            df['open_time'] = pd.to_datetime(df['open_time'])
            df = df.set_index('open_time')
        else:
            raise ValueError("Enriched data must have 'open_time' column or index.")

        # Ensure required columns for backtest exist (e.g., ATR)
        if config.BACKTEST_ATR_COLUMN not in df.columns:
            logger.warning(f"Required ATR column '{config.BACKTEST_ATR_COLUMN}' not found. Calculating default ATR(14)...")
            df.ta.atr(length=14, append=True)
            if 'ATR_14' in df.columns and config.BACKTEST_ATR_COLUMN == 'atr_14':
                df.rename(columns={'ATR_14': 'atr_14'}, inplace=True)
            elif config.BACKTEST_ATR_COLUMN not in df.columns:
                raise ValueError(f"Failed to calculate or find required ATR column: {config.BACKTEST_ATR_COLUMN}")

        # Load predictions (assuming ensemble_prob is needed for signal generation)
        preds_data = load_json(config.MODEL_PREDICTIONS_PATH)
        if preds_data is None:
            raise FileNotFoundError(f"Predictions file not found at {config.MODEL_PREDICTIONS_PATH}")
        preds_df = pd.DataFrame(preds_data)
        if 'index' in preds_df.columns:
            preds_df['open_time'] = pd.to_datetime(preds_df['index'])
            preds_df = preds_df.set_index('open_time')
        if 'ensemble_prob' not in preds_df.columns:
            raise ValueError("'ensemble_prob' not found in predictions data.")

        # Align data and predictions
        common_index = df.index.intersection(preds_df.index)
        if common_index.empty:
            raise ValueError("No common index between enriched data and predictions.")
        df_aligned = df.loc[common_index]
        preds_aligned = preds_df.loc[common_index, ['ensemble_prob']]
        df_combined = pd.concat([df_aligned, preds_aligned], axis=1)
        df_combined = df_combined.dropna(subset=['open', 'high', 'low', 'close', 'volume', config.BACKTEST_ATR_COLUMN, 'ensemble_prob'])
        if df_combined.empty:
            logger.error("Data is empty after loading, aligning and cleaning. Cannot run optimization.")
            return None, None
        logger.info(f"Data loaded successfully for indicator optimization. Shape: {df_combined.shape}")
        return df_combined, df_combined['ensemble_prob'].values
    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e}. Cannot run optimization.")
        return None, None
    except Exception as e:
        logger.error(f"Error loading data for indicator optimization: {e}")
        return None, None

def objective_indicators(trial: optuna.Trial) -> float:
    """
    Optuna objective function for indicator parameters.
    Uses a fast proxy metric (mean ensemble probability for non-flat signals) for optimization speed.
    The full backtest is run only for the best trial(s) after optimization.
    Args:
        trial (optuna.Trial): The Optuna trial object.
    Returns:
        float: Proxy metric to maximize (higher is better).
    """
    df, ensemble_probs = load_data_for_indicator_opt()
    if df is None or ensemble_probs is None:
        return -float('inf') # Return poor value if data loading failed

    # --- Suggest Parameters ---
    prob_threshold = trial.suggest_float("prob_threshold", 0.5, 0.8)
    sl_multiplier = trial.suggest_float("sl_multiplier", 0.5, 3.0)
    tp_multiplier = trial.suggest_float("tp_multiplier", 1.0, 5.0)

    # --- Generate Signals ---
    signals = generate_signals(df, ensemble_probs, threshold=prob_threshold)

    # --- Proxy Metric for Fast Optimization ---
    # Use the mean ensemble_prob for non-flat signals as a proxy for signal quality
    is_trade = signals != "Flat"
    if np.sum(is_trade) == 0:
        return -float('inf')
    proxy_metric = float(np.mean(ensemble_probs[is_trade]))

    # NOTE: For efficiency, we use the proxy metric for most trials.
    # Run the full backtest only for the best trial(s) after optimization.
    return proxy_metric

@memory_safe(min_free_percent=10)
def main():
    """Runs the Optuna optimization study for indicators/signals."""
    logger.info("--- Starting Indicator/Signal Optimization --- ")

    # Memory management: The @memory_safe decorator ensures that each optimization run checks available memory
    # before starting, and can abort or skip trials if free memory is too low. This is especially important
    # for Colab or other resource-constrained environments.

    # Load data once before starting study
    if load_data_for_indicator_opt()[0] is None:
        logger.error("Failed to load data. Aborting optimization.")
        return

    # Ensure results directory exists
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    study_name = "btc_usdt_indicator_optimize"
    storage_path = config.RESULTS_DIR / "optuna_indicator_study.db"
    storage_str = f"sqlite:///{storage_path}"
    try:
        logger.info(f"Creating Optuna study with SQLite storage at {storage_path}")
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_str,
            load_if_exists=True,
            direction="maximize"
        )
    except Exception as e:
        logger.error(f"Failed to create Optuna study with SQLite storage: {e}")
        return

    try:
        study.optimize(
            objective_indicators,
            n_trials=config.OPTUNA_N_TRIALS_INDICATORS,
            n_jobs=config.OPTUNA_N_JOBS
        )
    except Exception as e:
        logger.error(f"Optimization study failed: {e}", exc_info=True)
        return

    logger.info(f"Optimization finished. Number of trials: {len(study.trials)}")
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"  Value (Objective Metric): {study.best_value:.4f}")
    logger.info(f"  Params: {study.best_params}")

    # Save best parameters
    results = {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'best_trial_number': study.best_trial.number
    }
    save_path = config.INDICATOR_OPTIMIZE_RESULTS_PATH
    try:
        save_json(results, save_path)
        logger.info(f"Optimization results saved to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save optimization results to {save_path}")

    # --- Run full backtest for the best trial ---
    logger.info("Running full backtest for the best trial parameters...")
    best_params = study.best_params
    df, ensemble_probs = load_data_for_indicator_opt()
    if df is not None and ensemble_probs is not None:
        signals = generate_signals(
            df,
            ensemble_probs,
            threshold=best_params.get("prob_threshold", 0.6)
        )
        sl_multiplier = best_params.get("sl_multiplier", 1.0)
        tp_multiplier = best_params.get("tp_multiplier", 2.0)
        equity_curve, trade_log = run_backtest(
            df=df,
            signals=signals,
            initial_equity=config.INITIAL_EQUITY,
            atr_col=config.BACKTEST_ATR_COLUMN,
            sl_multiplier=sl_multiplier,
            tp_multiplier=tp_multiplier,
            commission_rate=config.COMMISSION_RATE,
            slippage_points=config.SLIPPAGE_POINTS,
            risk_fraction=config.RISK_FRACTION
        )
        metrics = calculate_metrics(equity_curve, trade_log, config.INITIAL_EQUITY)
        logger.info(f"Full backtest metrics for best trial: {metrics}")
        # Optionally, save these metrics
        results['full_backtest_metrics'] = metrics
        try:
            save_json(results, save_path)
            logger.info(f"Full backtest metrics saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save full backtest metrics to {save_path}")
    else:
        logger.error("Could not reload data for full backtest.")

    logger.info("--- Indicator/Signal Optimization Finished --- ")

if __name__ == '__main__':
    main()

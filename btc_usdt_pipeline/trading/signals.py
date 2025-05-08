# trading/signals.py
"""
Generates trading signals based on model predictions and technical indicators.
Refactored from scripts/signals.py to use centralized config and helpers.
"""
import numpy as np
import pandas as pd
from typing import List, Optional

# Use absolute imports from the package
from btc_usdt_pipeline import config
from btc_usdt_pipeline.utils.logging_config import setup_logging

logger = setup_logging(log_filename='signals.log')

def multi_timeframe_signal_logic(row: pd.Series, ensemble_prob: float, prob_threshold: float = config.PROBABILITY_THRESHOLD, signal_threshold: float = config.SIGNAL_THRESHOLD) -> str:
    """
    Generates a trading signal based on multi-timeframe indicators and an ensemble probability.
    Only generates Long/Short if probability exceeds prob_threshold.
    Uses signal_threshold for the directional logic (currently placeholder, needs refinement).

    Args:
        row (pd.Series): A row of the DataFrame containing indicator data.
        ensemble_prob (float): The ensemble prediction probability for this row.
        prob_threshold (float): Minimum probability required to consider a trade.
        signal_threshold (float): Threshold used in the original multi-timeframe logic.
                                  Needs review based on how ensemble_prob relates to direction.

    Returns:
        str: "Long", "Short", or "Flat"
    """
    # Check if probability meets the minimum threshold to consider a trade
    if ensemble_prob < prob_threshold:
        return "Flat"

    try:
        # --- Multi-Timeframe Confirmation Logic ---
        # Define bullish/bearish conditions based on configured HTF features
        # Example using 4h and 1h EMA and RSI (adjust based on config.FEATURES_HTF)
        bullish_4h = (row['close'] > row['4h_ema_20']) and (row['4h_rsi_14'] > 50)
        bearish_4h = (row['close'] < row['4h_ema_20']) and (row['4h_rsi_14'] < 50)
        bullish_1h = (row['close'] > row['1h_ema_20']) and (row['1h_rsi_14'] > 50)
        bearish_1h = (row['close'] < row['1h_ema_20']) and (row['1h_rsi_14'] < 50)

        # --- Signal Generation ---
        # This part needs careful review. The original logic used `ensemble_pred > threshold`.
        # Now we have `ensemble_prob`. Assuming high probability indicates the predicted move is likely.
        # We need a way to determine the *direction* of the predicted move if the model only predicts magnitude probability.
        # Placeholder: If HTF is bullish and prob > threshold, go Long.
        # Placeholder: If HTF is bearish and prob > threshold, go Short.
        # This assumes the probability relates to the *strength* of the HTF signal.
        # A better approach might involve models predicting direction or separate directional indicators.

        if bullish_4h and bullish_1h:
            # If HTF is bullish and probability is high, consider Long
            logger.debug(f"Index {row.name}: Bullish HTF confirmation, Prob={ensemble_prob:.4f} -> Long Signal")
            return "Long"
        elif bearish_4h and bearish_1h:
            # If HTF is bearish and probability is high, consider Short
            logger.debug(f"Index {row.name}: Bearish HTF confirmation, Prob={ensemble_prob:.4f} -> Short Signal")
            return "Short"
        else:
            # HTF signals are mixed or neutral, even if probability is high
            logger.debug(f"Index {row.name}: Mixed HTF signals, Prob={ensemble_prob:.4f} -> Flat Signal")
            return "Flat"

    except KeyError as e:
        # Log missing columns required for the signal logic
        logger.error(f"Missing required column for signal generation at index {row.name}: {e}. Returning Flat.")
        return "Flat"
    except Exception as e:
        logger.error(f"Error generating signal at index {row.name}: {e}. Returning Flat.")
        return "Flat"

def generate_signals(df: pd.DataFrame, probabilities: np.ndarray, threshold: float = config.PROBABILITY_THRESHOLD) -> np.ndarray:
    """
    Generates trading signals based on model probabilities and optional filters.
    Uses threshold from config by default.

    Args:
        df (pd.DataFrame): DataFrame containing features (potentially for filters), indexed by time.
                           Must align with probabilities array.
        probabilities (np.ndarray): Array of predicted probabilities for the positive class (target=1).
        threshold (float): Probability threshold to generate a Long signal.

    Returns:
        np.ndarray: Array of signals ("Long", "Short", "Flat").
    """
    if len(df) != len(probabilities):
        raise ValueError(f"DataFrame length ({len(df)}) and probabilities length ({len(probabilities)}) must match.")

    logger.info(f"Generating signals with probability threshold: {threshold}")

    # Basic strategy: Long if probability > threshold, otherwise Flat.
    # Currently no Short signals are generated based solely on probability < threshold.
    # A more complex strategy could use two thresholds or other indicators for shorts.
    signals = np.where(probabilities > threshold, "Long", "Flat")

    # --- Optional: Add Filters --- 
    # Example: Require EMA(9) > EMA(20) for Long signal confirmation
    # if 'ema_9' in df.columns and 'ema_20' in df.columns:
    #     ema_filter = df['ema_9'] > df['ema_20']
    #     signals = np.where((signals == "Long") & ema_filter, "Long", "Flat")
    #     logger.info("Applied EMA filter to Long signals.")
    # else:
    #     logger.warning("EMA columns not found for signal filtering.")
    # -----------------------------

    unique_signals, counts = np.unique(signals, return_counts=True)
    logger.info(f"Signal generation complete. Counts: {dict(zip(unique_signals, counts))}")

    return signals

# Example usage (can be run standalone for testing)
if __name__ == '__main__':
    logger.info("--- Signal Generation Test --- ")
    try:
        # Load enriched data
        df_test = pd.read_parquet(config.ENRICHED_DATA_PATH)
        # Cast numeric columns to memory-efficient types
        for col in df_test.select_dtypes(include=['float64', 'float']).columns:
            df_test[col] = df_test[col].astype('float32')
        for col in df_test.select_dtypes(include=['int64', 'int']).columns:
            df_test[col] = df_test[col].astype('int32')
        if 'open_time' in df_test.columns:
            df_test['open_time'] = pd.to_datetime(df_test['open_time'])
            df_test = df_test.set_index('open_time')
        logger.info(f"Loaded test data: {df_test.shape}")

        # Load or create dummy predictions
        try:
            predictions = pd.read_json(config.MODEL_PREDICTIONS_PATH)
            # Ensure predictions align with the end of the dataframe
            if len(predictions['ensemble_probs']) > len(df_test):
                 dummy_probs = np.array(predictions['ensemble_probs'][-len(df_test):])
            elif len(predictions['ensemble_probs']) < len(df_test):
                 # Pad with 0.5 if predictions are shorter (adjust as needed)
                 padding = np.full(len(df_test) - len(predictions['ensemble_probs']), 0.5)
                 dummy_probs = np.concatenate((padding, predictions['ensemble_probs']))
            else:
                 dummy_probs = np.array(predictions['ensemble_probs'])
            logger.info(f"Loaded predictions: {len(dummy_probs)} probabilities")
        except (FileNotFoundError, KeyError, Exception) as e:
            logger.warning(f"Could not load predictions from {config.MODEL_PREDICTIONS_PATH}: {e}. Using dummy probabilities.")
            # Create dummy probabilities (e.g., random or fixed)
            dummy_probs = np.random.rand(len(df_test)) * 0.4 + 0.3 # Random probs between 0.3 and 0.7

        if not df_test.empty and len(dummy_probs) == len(df_test):
            # Generate signals
            generated_signals = generate_signals(df_test, dummy_probs)
            print(f"Generated {len(generated_signals)} signals.")
            unique_signals, counts = np.unique(generated_signals, return_counts=True)
            print("Signal Counts:", dict(zip(unique_signals, counts)))
        elif len(dummy_probs) != len(df_test):
             logger.error(f"Test data length ({len(df_test)}) and prediction length ({len(dummy_probs)}) mismatch.")
        else:
            logger.warning("Test data is empty, skipping signal generation test.")

    except FileNotFoundError:
        logger.error(f"Test data file not found at {config.ENRICHED_DATA_PATH}. Cannot run example.")
    except Exception as e:
        logger.error(f"Error during signal generation test: {e}", exc_info=True)

    logger.info("--- Signal Generation Test Finished --- ")

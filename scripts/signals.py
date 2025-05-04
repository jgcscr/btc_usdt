"""
signals.py
Entry point script to generate trading signals using the btc_usdt_pipeline package.
Run as: python -m scripts.signals
"""
# Removed sys.path manipulation
import pandas as pd
import numpy as np
import json

from btc_usdt_pipeline import config
from btc_usdt_pipeline.trading.signals import generate_signals # Import the core function
from btc_usdt_pipeline.utils.helpers import setup_logger, load_json # Changed import

# Use a specific logger for this script
logger = setup_logger('generate_signals_script.log')

def main():
    logger.info("--- Running Signal Generation Script ---")
    try:
        # Load enriched data
        logger.info(f"Loading enriched data from: {config.ENRICHED_DATA_PATH}")
        # Add note about memory usage
        logger.info("Note: For very large datasets on memory-constrained environments (like Colab), consider loading data in chunks or using memory-efficient dtypes.")
        df = pd.read_parquet(config.ENRICHED_DATA_PATH)
        # Cast numeric columns to memory-efficient types
        for col in df.select_dtypes(include=['float64', 'float']).columns:
            df[col] = df[col].astype('float32')
        for col in df.select_dtypes(include=['int64', 'int']).columns:
            df[col] = df[col].astype('int32')
        if 'open_time' in df.columns:
            df['open_time'] = pd.to_datetime(df['open_time'])
            df = df.set_index('open_time')
        logger.info(f"Loaded enriched data: {df.shape}")

        # Load model predictions using the helper
        logger.info(f"Loading predictions from: {config.MODEL_PREDICTIONS_PATH}")
        # Use load_json helper which handles file not found
        predictions_data = load_json(config.MODEL_PREDICTIONS_PATH)

        if predictions_data is None:
             logger.error(f"Could not load predictions from {config.MODEL_PREDICTIONS_PATH}. Run model prediction first.")
             print(f"Error: Could not load predictions from {config.MODEL_PREDICTIONS_PATH}. Run model prediction first.")
             return

        # Convert loaded JSON data (dict of lists) to DataFrame
        # Assuming the JSON structure is like {'index': [...], 'ensemble_prob': [...], ...}
        # Adjust this based on the actual structure saved by ModelManager.predict_and_save
        try:
            predictions_df = pd.DataFrame(predictions_data)
            if 'index' in predictions_df.columns:
                 # Assuming 'index' column contains ISO format timestamps
                 predictions_df['open_time'] = pd.to_datetime(predictions_df['index'])
                 predictions_df = predictions_df.set_index('open_time')
            else:
                 logger.warning(f"Prediction data from {config.MODEL_PREDICTIONS_PATH} does not contain an 'index' column for alignment.")
                 # Attempt alignment based on matching lengths if appropriate, otherwise error
                 if len(predictions_df) != len(df):
                     logger.error("Prediction data length mismatch and no index for alignment.")
                     print("Error: Prediction data length mismatch and no index for alignment.")
                     return
                 # If lengths match, assume alignment (use with caution)
                 predictions_df.index = df.index[-len(predictions_df):] # Align with the end of df

        except Exception as e:
            logger.error(f"Error converting prediction JSON data to DataFrame: {e}")
            print(f"Error processing prediction data: {e}")
            return

        if 'ensemble_prob' not in predictions_df.columns:
            logger.error(f"'ensemble_prob' column not found in loaded predictions from {config.MODEL_PREDICTIONS_PATH}. Cannot generate signals.")
            print(f"Error: 'ensemble_prob' column not found in predictions data.")
            return

        logger.info(f"Loaded {len(predictions_df)} predictions.")

        # Align data and predictions.
        common_index = df.index.intersection(predictions_df.index)
        if common_index.empty:
            logger.error("No common index found between enriched data and predictions. Cannot align.")
            print("Error: Cannot align data and predictions.")
            return

        df_aligned = df.loc[common_index]
        ensemble_probs_aligned = predictions_df.loc[common_index, 'ensemble_prob'].values

        logger.info(f"Data and predictions aligned. Using {len(df_aligned)} rows for signal generation.")

        if df_aligned.empty:
            logger.error("Aligned data subset for signal generation is empty.")
            print("Error: Aligned data subset for signal generation is empty.")
            return

        # Generate signals using the function from the package
        signals = generate_signals(
            df_aligned,
            ensemble_probs_aligned,
            threshold=config.PROBABILITY_THRESHOLD # Use config threshold
        )

        # Save signals aligned with the subset dataframe
        signals_df = pd.DataFrame({'signal': signals}, index=df_aligned.index)
        signals_path = config.RESULTS_DIR / 'generated_signals.csv'
        # Ensure directory exists before saving
        signals_path.parent.mkdir(parents=True, exist_ok=True)
        signals_df.to_csv(signals_path)

        unique_signals, counts = np.unique(signals, return_counts=True)
        signal_counts = dict(zip(unique_signals, counts))
        logger.info(f"Generated Signal Counts: {signal_counts}")
        logger.info(f"Signals saved to {signals_path}")
        print("Generated Signal Counts:", signal_counts)
        print(f"Signals saved to {signals_path}")


    except FileNotFoundError as e:
        # This might catch the parquet load error if not handled earlier
        logger.error(f"Required file not found: {e}. Cannot generate signals.")
        print(f"Error: Required file not found ({e}). Cannot generate signals.")
    except Exception as e:
        logger.error(f"An error occurred during signal generation script: {e}", exc_info=True)
        print(f"An error occurred during signal generation script: {e}")

    logger.info("--- Signal Generation Script Finished ---")

if __name__ == '__main__':
    main()

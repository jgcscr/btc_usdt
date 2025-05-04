"""
models.py
Entry point script to run model predictions using the ModelManager.
Run as: python -m scripts.models
"""
# Removed sys.path manipulation
import pandas as pd

from btc_usdt_pipeline import config
from btc_usdt_pipeline.models.manage import ModelManager
from btc_usdt_pipeline.utils.helpers import setup_logger

# Use a specific logger for this script
logger = setup_logger('predict_script.log')

def main():
    logger.info("--- Running Model Prediction Script ---")
    try:
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

        if df.empty:
            logger.error("Enriched data is empty. Cannot run predictions.")
            print("Error: Enriched data is empty.")
            return

        # Initialize the model manager (loads models)
        logger.info("Initializing Model Manager...")
        manager = ModelManager()

        # Predict on all available data for backtesting/signal generation consistency
        # Consider predicting only on the most recent data if memory is a concern
        N_predictions = len(df)
        logger.info(f"Generating predictions for the latest {N_predictions} data points...")
        # The predict_and_save method now handles saving internally (ensure it creates dirs)
        predictions_df = manager.predict_and_save(df, N=N_predictions)

        if predictions_df is not None and not predictions_df.empty:
            num_preds = len(predictions_df)
            logger.info(f"Successfully generated and saved {num_preds} predictions to {config.MODEL_PREDICTIONS_PATH}")
            print(f"Generated and saved {num_preds} predictions to {config.MODEL_PREDICTIONS_PATH}")
        else:
            logger.warning("Prediction process completed, but no predictions were generated or saved.")
            print("Prediction process completed, but no predictions were generated or saved.")

    except FileNotFoundError:
        logger.error(f"Enriched data file not found at {config.ENRICHED_DATA_PATH}. Run feature computation first.")
        print(f"Error: Enriched data file not found at {config.ENRICHED_DATA_PATH}")
    except Exception as e:
        logger.error(f"An error occurred during prediction script: {e}", exc_info=True)
        print(f"An error occurred during prediction script: {e}")

    logger.info("--- Model Prediction Script Finished ---")

if __name__ == '__main__':
    main()
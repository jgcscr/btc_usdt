# models/manage.py
"""Manages loading models and making predictions."""

import os
import joblib
import json
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from pathlib import Path

# Sequence models (optional import)
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

from btc_usdt_pipeline import config
from btc_usdt_pipeline.utils.helpers import setup_logger, create_sequences
from btc_usdt_pipeline.io.serialization import save_json, load_json

logger = setup_logger('model_management.log')

class ModelManager:
    """Loads trained models and generates predictions."""
    def __init__(self):
        self.models: Dict[str, object] = {}
        self.load_models()

    def load_models(self):
        """Loads all models specified in the config."""
        logger.info("Loading models...")
        model_paths = {
            'rf': config.MODELS_DIR / config.RF_MODEL_NAME,
            'xgb': config.MODELS_DIR / config.XGB_MODEL_NAME,
            'lgbm': config.MODELS_DIR / config.LGBM_MODEL_NAME, # Added LGBM
            'lstm': config.MODELS_DIR / config.LSTM_MODEL_NAME,
            'gru': config.MODELS_DIR / config.GRU_MODEL_NAME,
        }

        for name, path in model_paths.items():
            if not path.exists():
                logger.warning(f"Model file not found for '{name}' at {path}. Skipping.")
                continue
            try:
                if name in ['lstm', 'gru']:
                    if TF_AVAILABLE:
                        self.models[name] = tf.keras.models.load_model(path)
                        logger.info(f"Loaded Keras model '{name}' from {path}")
                    else:
                        logger.warning(f"TensorFlow not available. Cannot load Keras model '{name}'.")
                else: # Sklearn / Joblib models
                    self.models[name] = joblib.load(path)
                    logger.info(f"Loaded model '{name}' from {path}")
            except Exception as e:
                logger.error(f"Error loading model '{name}' from {path}: {e}")

        if not self.models:
            logger.error("No models were successfully loaded. Prediction will not be possible.")

    def predict(self, df: pd.DataFrame, N: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Generates predictions from all loaded models for the last N rows of df."""
        if not self.models:
            logger.error("No models loaded, cannot make predictions.")
            return None

        if N is None:
            predict_df = df
        else:
            if N > len(df):
                logger.warning(f"Requested N={N} predictions, but only {len(df)} rows available. Predicting on all available rows.")
                predict_df = df
            else:
                predict_df = df.iloc[-N:]

        if predict_df.empty:
            logger.warning("Input DataFrame for prediction is empty.")
            return None

        predictions: Dict[str, np.ndarray] = {}
        tree_features = config.ALL_TREE_FEATURES
        seq_features = config.SEQUENCE_FEATURES
        timesteps = config.SEQUENCE_TIMESTEPS

        # Ensure required feature columns exist
        missing_tree_features = [f for f in tree_features if f not in predict_df.columns]
        missing_seq_features = [f for f in seq_features if f not in predict_df.columns]

        if missing_tree_features and any(m in self.models for m in ['rf', 'xgb', 'lgbm']):
            logger.error(f"Missing required tree features for prediction: {missing_tree_features}")
        if missing_seq_features and any(m in self.models for m in ['lstm', 'gru']):
            logger.error(f"Missing required sequence features for prediction: {missing_seq_features}")

        # Prepare data for tree models
        X_tree = predict_df[tree_features].dropna() # Drop rows with NaNs in features needed for tree models
        valid_tree_indices = X_tree.index

        # Prepare data for sequence models
        X_seq_data = predict_df[seq_features].values
        dummy_targets = np.zeros(len(X_seq_data))
        X_seq, _ = create_sequences(X_seq_data, dummy_targets, timesteps=timesteps)
        if X_seq.shape[0] > 0:
            valid_seq_indices = predict_df.index[timesteps - 1:]
            if len(valid_seq_indices) != X_seq.shape[0]:
                logger.warning(f"Sequence index alignment mismatch. Expected {X_seq.shape[0]} indices, got {len(valid_seq_indices)}. Check create_sequences logic.")
                valid_seq_indices = valid_seq_indices[:X_seq.shape[0]] # Truncate if too long
        else:
            valid_seq_indices = pd.Index([])

        for name, model in self.models.items():
            logger.debug(f"Predicting with model: {name}")
            try:
                if name in ['rf', 'xgb', 'lgbm']:
                    if not X_tree.empty and not missing_tree_features:
                        if hasattr(model, 'predict_proba'):
                            probs = model.predict_proba(X_tree)[:, 1]
                            predictions[name] = pd.Series(probs, index=valid_tree_indices)
                    else:
                        logger.warning(f"Skipping prediction for {name} due to missing features or empty data after NaN drop.")

                elif name in ['lstm', 'gru']:
                    if X_seq.shape[0] > 0 and not missing_seq_features:
                        if hasattr(model, 'predict'):
                            probs = model.predict(X_seq).flatten()
                            predictions[name] = pd.Series(probs, index=valid_seq_indices)
                    else:
                        logger.warning(f"Skipping prediction for {name} due to insufficient data for sequences or missing features.")

            except Exception as e:
                logger.error(f"Error during prediction with model '{name}': {e}")

        if not predictions:
            logger.error("No predictions were generated by any model.")
            return None

        results_df = pd.DataFrame(predictions)

        valid_model_preds = [col for col in results_df.columns if results_df[col].notna().any()]
        if valid_model_preds:
            results_df['ensemble_prob'] = results_df[valid_model_preds].mean(axis=1)
        else:
            results_df['ensemble_prob'] = np.nan
            logger.warning("Could not calculate ensemble probability as no models produced predictions.")

        results_df = results_df.reindex(predict_df.index)

        logger.info(f"Generated predictions for {len(results_df.dropna(subset=['ensemble_prob']))} timestamps.")
        return results_df

    def predict_and_save(self, df: pd.DataFrame, N: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Generates predictions and saves them to a JSON file."""
        predictions_df = self.predict(df, N)

        if predictions_df is None or predictions_df.empty:
            logger.error("Prediction failed or resulted in an empty DataFrame. Nothing saved.")
            return None

        output_df = predictions_df.copy()
        output_df = output_df.replace({np.nan: None})
        output_df['index'] = output_df.index.astype(str)
        predictions_dict = output_df.to_dict(orient='list')

        save_path = str(config.MODEL_PREDICTIONS_PATH)
        try:
            save_json(predictions_dict, save_path)
            logger.info(f"Predictions saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save predictions dictionary to {save_path}.")

        return predictions_df

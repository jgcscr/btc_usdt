# models/train.py
"""
Contains functions for training different model types.
Refactored from scripts/retrain_models.py to use centralized config and helpers.
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from tensorflow import keras
from pathlib import Path
from typing import Tuple, Any, Optional

# Use absolute imports from the package
from btc_usdt_pipeline import config
from btc_usdt_pipeline.utils.helpers import setup_logger, create_sequences
from btc_usdt_pipeline.types import ModelInputType, ModelOutputType, MetricsDict

logger = setup_logger('model_train.log')

# Features and sequence features are now imported from config
FEATURES = config.ALL_TREE_FEATURES
SEQ_FEATURES = config.SEQUENCE_FEATURES
TIMESTEPS = config.SEQUENCE_TIMESTEPS
TARGET = config.TARGET_COLUMN_NAME

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

def save_model(model: object, model_path: Path) -> None:
    """Helper function to save a model, ensuring directory exists."""
    try:
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(model, 'save'):  # Keras model
            model.save(model_path)
        else:  # Sklearn model
            joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
    except Exception as e:
        logger.error(f"Error saving model to {model_path}: {e}")

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Any, MetricsDict]:
    """Trains and saves a Random Forest model."""
    logger.info('Training Random Forest Classifier...')
    model = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        n_jobs=-1,
        random_state=config.RANDOM_STATE
    )
    try:
        model.fit(X_train, y_train)
        logger.info('Random Forest Classifier training complete.')
        model_path = config.MODELS_DIR / config.RF_MODEL_NAME
        save_model(model, model_path)
        metrics: MetricsDict = {"accuracy": model.score(X_train, y_train)}
        return model, metrics
    except Exception as e:
        logger.error(f"Error training Random Forest: {e}", exc_info=True)
        return None, {}

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None) -> Tuple[Any, MetricsDict]:
    """Trains and saves an XGBoost model."""
    logger.info('Training XGBoost Classifier...')
    # Ensure y_train and y_val are integer type for XGBoost
    y_train_xgb = y_train.astype(int)
    model = XGBClassifier(
        n_estimators=config.XGB_N_ESTIMATORS,
        n_jobs=-1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=config.RANDOM_STATE
    )

    fit_params = {}
    if X_val is not None and y_val is not None:
        y_val_xgb = y_val.astype(int)
        fit_params['eval_set'] = [(X_val, y_val_xgb)]
        fit_params['early_stopping_rounds'] = config.EARLY_STOPPING_PATIENCE  # Use same patience
        fit_params['verbose'] = False
        logger.info("Using validation set for XGBoost early stopping.")

    try:
        model.fit(X_train, y_train_xgb, **fit_params)
        logger.info('XGBoost Classifier training complete.')
        model_path = config.MODELS_DIR / config.XGB_MODEL_NAME
        save_model(model, model_path)
        metrics: MetricsDict = {"accuracy": model.score(X_train, y_train_xgb)}
        return model, metrics
    except Exception as e:
        logger.error(f"Error training XGBoost: {e}", exc_info=True)
        return None, {}

def train_lstm(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Any, MetricsDict]:
    """Trains an LSTM model."""
    if not TF_AVAILABLE:
        logger.error("TensorFlow not available. Skipping LSTM training.")
        return None, {}

    logger.info("Training LSTM model...")
    model = Sequential([
        LSTM(config.LSTM_UNITS, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),  # Added dropout for regularization
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=config.EARLY_STOPPING_PATIENCE, restore_best_weights=True)

    try:
        # Use sequence-specific config variables directly
        epochs = config.SEQUENCE_EPOCHS
        batch_size = config.SEQUENCE_BATCH_SIZE

        logger.info(f"Training LSTM with epochs={epochs}, batch_size={batch_size}, patience={config.EARLY_STOPPING_PATIENCE}")
        history = model.fit(X_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(X_val, y_val),
                            callbacks=[early_stopping],
                            verbose=1)  # Set verbose to 1 or 2 to see progress

        model_path = config.MODELS_DIR / config.LSTM_MODEL_NAME
        save_model(model, model_path)
        metrics: MetricsDict = {"accuracy": float(history.history['accuracy'][-1])}
        return model, metrics
    except Exception as e:
        logger.error(f"Error training or saving LSTM model: {e}", exc_info=True)
        return None, {}

def train_gru(X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> Tuple[Any, MetricsDict]:
    """Trains a GRU model."""
    if not TF_AVAILABLE:
        logger.error("TensorFlow not available. Skipping GRU training.")
        return None, {}

    logger.info("Training GRU model...")
    model = Sequential([
        GRU(config.GRU_UNITS, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),  # Added dropout for regularization
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=config.EARLY_STOPPING_PATIENCE, restore_best_weights=True)

    try:
        # Use sequence-specific config variables directly
        epochs = config.SEQUENCE_EPOCHS
        batch_size = config.SEQUENCE_BATCH_SIZE

        logger.info(f"Training GRU with epochs={epochs}, batch_size={batch_size}, patience={config.EARLY_STOPPING_PATIENCE}")
        history = model.fit(X_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(X_val, y_val),
                            callbacks=[early_stopping],
                            verbose=1)  # Set verbose to 1 or 2 to see progress

        model_path = config.MODELS_DIR / config.GRU_MODEL_NAME
        save_model(model, model_path)
        metrics: MetricsDict = {"accuracy": float(history.history['accuracy'][-1])}
        return model, metrics
    except Exception as e:
        logger.error(f"Error training or saving GRU model: {e}", exc_info=True)
        return None, {}

def predict_model(model: Any, X: ModelInputType) -> ModelOutputType:
    """Generates predictions using a trained model."""
    try:
        predictions = model.predict(X)
        return predictions
    except Exception as e:
        logger.error(f"Error during model prediction: {e}", exc_info=True)
        return []

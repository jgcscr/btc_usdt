from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def objective(trial):
    """
    Standalone objective function for RandomForest optimization (for testing).
    Mimics the structure expected by the test.
    """
    import numpy as np
    from btc_usdt_pipeline import config
    df = load_data_once()
    features = getattr(config, 'ALL_TREE_FEATURES', ['feature1'])
    target_col = getattr(config, 'TARGET_COLUMN_NAME', 'target')
    X = df[features].values
    y = df[target_col].values
    # Simple train/val split
    n_val = int(0.3 * len(X))
    X_train, X_val = X[:-n_val], X[-n_val:]
    y_train, y_val = y[:-n_val], y[-n_val:]
    # Get hyperparameters from trial
    n_estimators = trial.suggest_int('n_estimators', 10, 100)
    max_depth = trial.suggest_int('max_depth', 2, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    trial.set_user_attr('accuracy', acc)
    return acc
from btc_usdt_pipeline.utils.data_manager import DataManager
# Global cache for loaded data
_loaded_df_cache = None

def load_data_once():
    """
    Loads and caches the main data DataFrame for optimization. Only supports parquet files.
    Returns:
        pd.DataFrame or None
    """
    global _loaded_df_cache
    if _loaded_df_cache is not None:
        return _loaded_df_cache
    dm = DataManager()
    # You may want to adjust the path and config as needed
    from btc_usdt_pipeline import config
    df = dm.load_data(getattr(config, 'ENRICHED_DATA_PATH', 'data/1m_btcusdt_enriched.parquet'), file_type='parquet')
    _loaded_df_cache = df
    return df
"""
Auto-optimization module for ML models and trading strategies.
Provides automated hyperparameter tuning and strategy selection.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime
import joblib
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import matplotlib.pyplot as plt

from btc_usdt_pipeline.utils.logging_config import setup_logging
from btc_usdt_pipeline.models.model_metrics import ModelMetricsTracker

logger = setup_logging('auto_optimize.log')

class ModelOptimizer:
    """
    Automated optimization for machine learning models.
    Uses Optuna for hyperparameter optimization with support for different model types.
    """
    
    def __init__(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        metric: str = 'accuracy',
        n_trials: int = 50,
        timeout: Optional[int] = None,
        study_name: Optional[str] = None,
        direction: str = 'maximize',
        storage: Optional[str] = None
    ):
        """
        Initialize the model optimizer.
        
        Args:
            model_type: Type of model to optimize ('xgboost', 'lightgbm', 'lstm', etc.)
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            metric: Performance metric to optimize
            n_trials: Number of optimization trials
            timeout: Optimization timeout in seconds
            study_name: Name for the optimization study
            direction: Optimization direction ('maximize' or 'minimize')
            storage: Optuna storage URL (for persistence)
        """
        self.model_type = model_type
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.metric = metric
        self.n_trials = n_trials
        self.timeout = timeout
        self.direction = direction
        
        # Set up Optuna study
        self.study_name = study_name or f"{model_type}_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.storage = storage
        
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction=direction,
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
            storage=storage,
            load_if_exists=True
        )
        
        # Initialize best model and params
        self.best_model = None
        self.best_params = None
        self.best_score = None
        
        # Setup model-specific optimization methods
        self._setup_objective_func()
        
    def _setup_objective_func(self):
        """Set up the appropriate objective function based on model type."""
        model_objectives = {
            'xgboost': self._objective_xgboost,
            'lightgbm': self._objective_lightgbm,
            'random_forest': self._objective_random_forest,
            'lstm': self._objective_lstm,
            'gru': self._objective_gru,
            'tcn': self._objective_tcn
        }
        
        if self.model_type not in model_objectives:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        self.objective_func = model_objectives[self.model_type]
        
    def _objective_xgboost(self, trial):
        """Objective function for XGBoost optimization."""
        from xgboost import XGBClassifier
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        
        # Define hyperparameters to tune
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 10.0),
            'use_label_encoder': False,
            'eval_metric': 'logloss',
            'random_state': 42
        }
        
        # Create and train model
        model = XGBClassifier(**params)
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Evaluate on validation set
        y_pred = model.predict(self.X_val)
        y_pred_proba = model.predict_proba(self.X_val)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_val, y_pred),
            'f1': f1_score(self.y_val, y_pred),
            'precision': precision_score(self.y_val, y_pred),
            'recall': recall_score(self.y_val, y_pred),
            'roc_auc': roc_auc_score(self.y_val, y_pred_proba)
        }
        
        # Return the specified metric
        if self.metric in metrics:
            return metrics[self.metric]
        else:
            # Default to accuracy if specified metric not available
            return metrics['accuracy']
            
    def _objective_lightgbm(self, trial):
        """Objective function for LightGBM optimization."""
        from lightgbm import LGBMClassifier
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        
        # Define hyperparameters to tune
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
            'num_leaves': trial.suggest_int('num_leaves', 10, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
            'class_weight': 'balanced',
            'random_state': 42
        }
        
        # Create and train model
        model = LGBMClassifier(**params)
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Evaluate on validation set
        y_pred = model.predict(self.X_val)
        y_pred_proba = model.predict_proba(self.X_val)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_val, y_pred),
            'f1': f1_score(self.y_val, y_pred),
            'precision': precision_score(self.y_val, y_pred),
            'recall': recall_score(self.y_val, y_pred),
            'roc_auc': roc_auc_score(self.y_val, y_pred_proba)
        }
        
        # Return the specified metric
        if self.metric in metrics:
            return metrics[self.metric]
        else:
            return metrics['accuracy']
            
    def _objective_random_forest(self, trial):
        """Objective function for Random Forest optimization."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        
        # Define hyperparameters to tune
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'class_weight': 'balanced',
            'random_state': 42
        }
        
        # Create and train model
        model = RandomForestClassifier(**params)
        model.fit(self.X_train, self.y_train)
        
        # Evaluate on validation set
        y_pred = model.predict(self.X_val)
        y_pred_proba = model.predict_proba(self.X_val)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_val, y_pred),
            'f1': f1_score(self.y_val, y_pred),
            'precision': precision_score(self.y_val, y_pred),
            'recall': recall_score(self.y_val, y_pred),
            'roc_auc': roc_auc_score(self.y_val, y_pred_proba)
        }
        
        # Return the specified metric
        if self.metric in metrics:
            return metrics[self.metric]
        else:
            return metrics['accuracy']
            
    def _objective_lstm(self, trial):
        """Objective function for LSTM model optimization."""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        
        # Check if input data is sequence data (3D)
        if len(self.X_train.shape) != 3:
            logger.error("Input data for LSTM must be 3D (samples, timesteps, features)")
            raise ValueError("Input data for LSTM must be 3D (samples, timesteps, features)")
        
        # Define hyperparameters to tune
        units = trial.suggest_int('units', 32, 256)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        recurrent_dropout = trial.suggest_float('recurrent_dropout', 0.0, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
        
        # Create LSTM model
        model = Sequential()
        model.add(LSTM(
            units=units,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            return_sequences=False,
            input_shape=(self.X_train.shape[1], self.X_train.shape[2])
        ))
        
        if use_batch_norm:
            model.add(BatchNormalization())
            
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Define early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            self.X_train, self.y_train,
            epochs=100,
            batch_size=batch_size,
            validation_data=(self.X_val, self.y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate on validation set
        y_pred_proba = model.predict(self.X_val).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_val, y_pred),
            'f1': f1_score(self.y_val, y_pred),
            'precision': precision_score(self.y_val, y_pred),
            'recall': recall_score(self.y_val, y_pred),
            'roc_auc': roc_auc_score(self.y_val, y_pred_proba)
        }
        
        # Clean up to prevent memory leaks
        tf.keras.backend.clear_session()
        
        # Return the specified metric
        if self.metric in metrics:
            return metrics[self.metric]
        else:
            return metrics['accuracy']
            
    def _objective_gru(self, trial):
        """Objective function for GRU model optimization."""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        
        # Check if input data is sequence data (3D)
        if len(self.X_train.shape) != 3:
            logger.error("Input data for GRU must be 3D (samples, timesteps, features)")
            raise ValueError("Input data for GRU must be 3D (samples, timesteps, features)")
        
        # Define hyperparameters to tune
        units = trial.suggest_int('units', 32, 256)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        recurrent_dropout = trial.suggest_float('recurrent_dropout', 0.0, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
        
        # Create GRU model
        model = Sequential()
        model.add(GRU(
            units=units,
            dropout=dropout_rate,
            recurrent_dropout=recurrent_dropout,
            return_sequences=False,
            input_shape=(self.X_train.shape[1], self.X_train.shape[2])
        ))
        
        if use_batch_norm:
            model.add(BatchNormalization())
            
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Define early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            self.X_train, self.y_train,
            epochs=100,
            batch_size=batch_size,
            validation_data=(self.X_val, self.y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate on validation set
        y_pred_proba = model.predict(self.X_val).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_val, y_pred),
            'f1': f1_score(self.y_val, y_pred),
            'precision': precision_score(self.y_val, y_pred),
            'recall': recall_score(self.y_val, y_pred),
            'roc_auc': roc_auc_score(self.y_val, y_pred_proba)
        }
        
        # Clean up to prevent memory leaks
        tf.keras.backend.clear_session()
        
        # Return the specified metric
        if self.metric in metrics:
            return metrics[self.metric]
        else:
            return metrics['accuracy']
            
    def _objective_tcn(self, trial):
        """Objective function for Temporal Convolutional Network optimization."""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.callbacks import EarlyStopping
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
        
        # Try to import TCN layers
        try:
            from tcn import TCN
        except ImportError:
            logger.error("TCN package not installed. Install with: pip install keras-tcn")
            raise ImportError("TCN package not installed. Install with: pip install keras-tcn")
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        
        # Check if input data is sequence data (3D)
        if len(self.X_train.shape) != 3:
            logger.error("Input data for TCN must be 3D (samples, timesteps, features)")
            raise ValueError("Input data for TCN must be 3D (samples, timesteps, features)")
        
        # Define hyperparameters to tune
        nb_filters = trial.suggest_int('nb_filters', 32, 128)
        kernel_size = trial.suggest_int('kernel_size', 2, 8)
        nb_stacks = trial.suggest_int('nb_stacks', 1, 4)
        dilations = [2 ** i for i in range(trial.suggest_int('dilations_power', 3, 8))]
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        
        # Create TCN model
        model = Sequential()
        model.add(TCN(
            nb_filters=nb_filters,
            kernel_size=kernel_size,
            nb_stacks=nb_stacks,
            dilations=dilations,
            dropout_rate=dropout_rate,
            return_sequences=False,
            input_shape=(self.X_train.shape[1], self.X_train.shape[2])
        ))
        
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Define early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        history = model.fit(
            self.X_train, self.y_train,
            epochs=100,
            batch_size=batch_size,
            validation_data=(self.X_val, self.y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Evaluate on validation set
        y_pred_proba = model.predict(self.X_val).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_val, y_pred),
            'f1': f1_score(self.y_val, y_pred),
            'precision': precision_score(self.y_val, y_pred),
            'recall': recall_score(self.y_val, y_pred),
            'roc_auc': roc_auc_score(self.y_val, y_pred_proba)
        }
        
        # Clean up to prevent memory leaks
        tf.keras.backend.clear_session()
        
        # Return the specified metric
        if self.metric in metrics:
            return metrics[self.metric]
        else:
            return metrics['accuracy']
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run the optimization process.
        
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting optimization for {self.model_type} model...")
        logger.info(f"Metric to optimize: {self.metric}")
        logger.info(f"Number of trials: {self.n_trials}")
        
        # Run optimization
        self.study.optimize(
            self.objective_func,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Get best parameters and trial
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        logger.info(f"Optimization complete. Best {self.metric}: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        # Train final model with best parameters
        self._train_best_model()
        
        # Return results
        return {
            'model_type': self.model_type,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_history': self._get_optimization_history(),
            'study_name': self.study_name
        }
    
    def _train_best_model(self):
        """Train the final model using the best parameters."""
        logger.info("Training final model with best parameters...")
        
        if self.model_type == 'xgboost':
            from xgboost import XGBClassifier
            self.best_model = XGBClassifier(
                **self.best_params,
                use_label_encoder=False,
                eval_metric='logloss',
                random_state=42
            )
            self.best_model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
            
        elif self.model_type == 'lightgbm':
            from lightgbm import LGBMClassifier
            self.best_model = LGBMClassifier(
                **self.best_params,
                random_state=42
            )
            self.best_model.fit(
                self.X_train, self.y_train,
                eval_set=[(self.X_val, self.y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
            
        elif self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            self.best_model = RandomForestClassifier(
                **self.best_params,
                random_state=42
            )
            self.best_model.fit(self.X_train, self.y_train)
            
        elif self.model_type in ['lstm', 'gru', 'tcn']:
            # For deep learning models, we don't retrain here since they take longer
            # and the best model is already saved during optimization via callbacks
            logger.info(f"Best {self.model_type} model must be trained separately due to complexity")
            self.best_model = None
            
        logger.info("Final model training complete.")
        
    def _get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get the history of all optimization trials."""
        trials = []
        
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trials.append({
                    'number': trial.number,
                    'params': trial.params,
                    'value': trial.value,
                    'datetime': trial.datetime.isoformat() if trial.datetime else None
                })
                
        return trials
        
    def save_model(self, file_path: str) -> bool:
        """
        Save the best model to disk.
        
        Args:
            file_path: Path to save the model
            
        Returns:
            Success status
        """
        if self.best_model is None:
            logger.warning("No best model to save")
            return False
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save model
            if self.model_type in ['xgboost', 'lightgbm', 'random_forest']:
                joblib.dump(self.best_model, file_path)
            elif self.model_type in ['lstm', 'gru', 'tcn']:
                # For TensorFlow models
                self.best_model.save(file_path)
                
            logger.info(f"Model saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
            
    def save_study(self, file_path: str) -> bool:
        """
        Save the optimization study to disk.
        
        Args:
            file_path: Path to save the study
            
        Returns:
            Success status
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save study results
            study_data = {
                'study_name': self.study_name,
                'direction': self.direction,
                'model_type': self.model_type,
                'best_params': self.best_params,
                'best_score': self.best_score,
                'optimization_history': self._get_optimization_history(),
                'datetime': datetime.now().isoformat()
            }
            
            with open(file_path, 'w') as f:
                json.dump(study_data, f, indent=2)
                
            logger.info(f"Optimization study saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving study: {e}")
            return False
            
    def plot_optimization_history(self, file_path: Optional[str] = None):
        """
        Plot the optimization history.
        
        Args:
            file_path: Path to save the plot (None to display)
        """
        # Create optimization visualization
        fig = optuna.visualization.plot_optimization_history(self.study)
        
        if file_path:
            # Convert plotly figure to matplotlib figure and save
            import plotly.io as pio
            pio.write_image(fig, file_path)
            logger.info(f"Optimization history plot saved to {file_path}")
        else:
            fig.show()
            
    def plot_param_importances(self, file_path: Optional[str] = None):
        """
        Plot parameter importances.
        
        Args:
            file_path: Path to save the plot (None to display)
        """
        # Create parameter importance visualization
        fig = optuna.visualization.plot_param_importances(self.study)
        
        if file_path:
            # Convert plotly figure to matplotlib figure and save
            import plotly.io as pio
            pio.write_image(fig, file_path)
            logger.info(f"Parameter importance plot saved to {file_path}")
        else:
            fig.show()

class StrategyOptimizer:
    """
    Automated optimization for trading strategies.
    Tunes strategy parameters based on backtest results.
    """
    
    def __init__(
        self,
        strategy_class,
        data: pd.DataFrame,
        metric: str = 'sharpe_ratio',
        n_trials: int = 50,
        timeout: Optional[int] = None,
        direction: str = 'maximize',
        initial_capital: float = 10000.0
    ):
        """
        Initialize the strategy optimizer.
        
        Args:
            strategy_class: Trading strategy class to optimize
            data: Market data for backtesting
            metric: Performance metric to optimize
            n_trials: Number of optimization trials
            timeout: Optimization timeout in seconds
            direction: Optimization direction ('maximize' or 'minimize')
            initial_capital: Initial capital for backtesting
        """
        self.strategy_class = strategy_class
        self.data = data
        self.metric = metric
        self.n_trials = n_trials
        self.timeout = timeout
        self.direction = direction
        self.initial_capital = initial_capital
        
        # Set up Optuna study
        self.study_name = f"strategy_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction=direction,
            sampler=TPESampler(seed=42)
        )
        
        # Initialize best strategy and params
        self.best_strategy = None
        self.best_params = None
        self.best_score = None
        self.best_backtest_results = None
        
    def objective(self, trial):
        """Objective function for strategy optimization."""
        # Define strategy parameters to tune
        # This should be customized based on the specific strategy
        params = self.strategy_class.suggest_params(trial)
        
        # Create strategy instance with trial parameters
        strategy = self.strategy_class(**params)
        
        # Run backtest
        from btc_usdt_pipeline.trading.backtest import Backtest
        backtest = Backtest(
            data=self.data,
            strategy=strategy,
            initial_capital=self.initial_capital
        )
        
        results = backtest.run()
        
        # Extract the specified metric from results
        if self.metric not in results:
            logger.warning(f"Metric {self.metric} not found in backtest results. Available metrics: {list(results.keys())}")
            return 0.0  # Return worst possible value
            
        # Store results for best trial
        if self.best_score is None or \
           (self.direction == 'maximize' and results[self.metric] > self.best_score) or \
           (self.direction == 'minimize' and results[self.metric] < self.best_score):
            self.best_score = results[self.metric]
            self.best_params = params
            self.best_strategy = strategy
            self.best_backtest_results = results
            
        return results[self.metric]
        
    def optimize(self) -> Dict[str, Any]:
        """
        Run the optimization process.
        
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting optimization for trading strategy...")
        logger.info(f"Metric to optimize: {self.metric}")
        logger.info(f"Number of trials: {self.n_trials}")
        
        # Run optimization
        self.study.optimize(
            self.objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Get best parameters and trial
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        logger.info(f"Optimization complete. Best {self.metric}: {self.best_score:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        # Return results
        return {
            'strategy_name': self.strategy_class.__name__,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'optimization_history': self._get_optimization_history(),
            'study_name': self.study_name
        }
        
    def _get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get the history of all optimization trials."""
        trials = []
        
        for trial in self.study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trials.append({
                    'number': trial.number,
                    'params': trial.params,
                    'value': trial.value,
                    'datetime': trial.datetime.isoformat() if trial.datetime else None
                })
                
        return trials
        
    def save_strategy(self, file_path: str) -> bool:
        """
        Save the best strategy to disk.
        
        Args:
            file_path: Path to save the strategy
            
        Returns:
            Success status
        """
        if self.best_strategy is None:
            logger.warning("No best strategy to save")
            return False
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save strategy parameters
            strategy_data = {
                'strategy_name': self.strategy_class.__name__,
                'params': self.best_params,
                'metric': self.metric,
                'score': self.best_score,
                'backtest_results': self.best_backtest_results,
                'datetime': datetime.now().isoformat()
            }
            
            with open(file_path, 'w') as f:
                json.dump(strategy_data, f, indent=2)
                
            logger.info(f"Strategy saved to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving strategy: {e}")
            return False
            
    def plot_optimization_history(self, file_path: Optional[str] = None):
        """
        Plot the optimization history.
        
        Args:
            file_path: Path to save the plot (None to display)
        """
        # Create optimization visualization
        fig = optuna.visualization.plot_optimization_history(self.study)
        
        if file_path:
            # Convert plotly figure to matplotlib figure and save
            import plotly.io as pio
            pio.write_image(fig, file_path)
            logger.info(f"Optimization history plot saved to {file_path}")
        else:
            fig.show()
            
    def plot_param_importances(self, file_path: Optional[str] = None):
        """
        Plot parameter importances.
        
        Args:
            file_path: Path to save the plot (None to display)
        """
        # Create parameter importance visualization
        fig = optuna.visualization.plot_param_importances(self.study)
        
        if file_path:
            # Convert plotly figure to matplotlib figure and save
            import plotly.io as pio
            pio.write_image(fig, file_path)
            logger.info(f"Parameter importance plot saved to {file_path}")
        else:
            fig.show()

def optimize_models(
    data_path: str,
    model_types: List[str],
    target_column: str = 'target',
    feature_columns: Optional[List[str]] = None,
    test_size: float = 0.2,
    val_size: float = 0.2,
    metric: str = 'f1',
    n_trials: int = 50,
    output_dir: str = 'models/optimized',
    track_metrics: bool = True
) -> Dict[str, Any]:
    """
    Optimize multiple model types on the same dataset.
    
    Args:
        data_path: Path to data file
        model_types: List of model types to optimize
        target_column: Target column name
        feature_columns: Feature column names (None for all except target)
        test_size: Proportion of data for test set
        val_size: Proportion of training data for validation
        metric: Performance metric to optimize
        n_trials: Number of optimization trials per model
        output_dir: Directory to save optimized models
        track_metrics: Whether to track metrics in ModelMetricsTracker
        
    Returns:
        Dictionary with optimization results for each model type
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    data = pd.read_parquet(data_path)
    
    # Determine feature columns
    if feature_columns is None:
        feature_columns = [col for col in data.columns if col != target_column]
        
    # Split data into features and target
    X = data[feature_columns].values
    y = data[target_column].values
    
    # Split into train+val and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=False
    )
    
    # Split train+val into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=42, shuffle=False
    )
    
    # Normalize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # For RNN models, reshape input data
    X_train_seq = None
    X_val_seq = None
    X_test_seq = None
    
    if any(model_type in ['lstm', 'gru', 'tcn'] for model_type in model_types):
        # Need to reshape for sequence models
        # Assuming we want to use 10 time steps for each prediction
        from btc_usdt_pipeline.utils.helpers import create_sequences
        
        # Create sequence data for training, validation, and test sets
        sequence_length = 10
        X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
        X_val_seq, y_val_seq = create_sequences(X_val, y_val, sequence_length)
        X_test_seq, y_test_seq = create_sequences(X_test, y_test, sequence_length)
    
    # Initialize metrics tracker if needed
    metrics_tracker = None
    if track_metrics:
        metrics_tracker = ModelMetricsTracker()
    
    # Optimize each model type
    results = {}
    
    for model_type in model_types:
        logger.info(f"Optimizing model type: {model_type}")
        
        # Prepare data based on model type
        if model_type in ['lstm', 'gru', 'tcn']:
            # Use sequence data for RNN models
            if X_train_seq is None:
                logger.error(f"Sequence data not available for {model_type}")
                continue
                
            optimizer = ModelOptimizer(
                model_type=model_type,
                X_train=X_train_seq,
                y_train=y_train_seq,
                X_val=X_val_seq,
                y_val=y_val_seq,
                metric=metric,
                n_trials=n_trials,
                direction='maximize'
            )
        else:
            # Use regular data for other models
            optimizer = ModelOptimizer(
                model_type=model_type,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                metric=metric,
                n_trials=n_trials,
                direction='maximize'
            )
            
        # Run optimization
        opt_result = optimizer.optimize()
        results[model_type] = opt_result
        
        # Save model
        model_id = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_dir = os.path.join(output_dir, model_type)
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f"{model_id}.joblib")
        study_path = os.path.join(model_dir, f"{model_id}_study.json")
        
        optimizer.save_model(model_path)
        optimizer.save_study(study_path)
        
        # Save optimization plots
        plots_dir = os.path.join(model_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        optimizer.plot_optimization_history(
            os.path.join(plots_dir, f"{model_id}_history.png")
        )
        optimizer.plot_param_importances(
            os.path.join(plots_dir, f"{model_id}_importance.png")
        )
        
        # Track metrics
        if metrics_tracker:
            # Calculate test metrics using the best model
            if model_type in ['lstm', 'gru', 'tcn']:
                # For deep learning models
                if optimizer.best_model:
                    y_pred_proba = optimizer.best_model.predict(X_test_seq).flatten()
                    y_pred = (y_pred_proba > 0.5).astype(int)
                    
                    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
                    test_metrics = {
                        'accuracy': accuracy_score(y_test_seq, y_pred),
                        'f1': f1_score(y_test_seq, y_pred),
                        'precision': precision_score(y_test_seq, y_pred),
                        'recall': recall_score(y_test_seq, y_pred),
                        'roc_auc': roc_auc_score(y_test_seq, y_pred_proba)
                    }
            else:
                # For traditional ML models
                if optimizer.best_model:
                    y_pred = optimizer.best_model.predict(X_test)
                    y_pred_proba = optimizer.best_model.predict_proba(X_test)[:, 1]
                    
                    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
                    test_metrics = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'f1': f1_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred),
                        'recall': recall_score(y_test, y_pred),
                        'roc_auc': roc_auc_score(y_test, y_pred_proba)
                    }
            
            # Add metrics to tracker
            if 'test_metrics' in locals():
                metrics_tracker.add_model_metrics(
                    model_id=model_id,
                    model_type=model_type,
                    metrics=test_metrics,
                    parameters=optimizer.best_params,
                    training_date=datetime.now(),
                    dataset_info={
                        'data_path': data_path,
                        'num_features': len(feature_columns),
                        'train_samples': len(X_train),
                        'val_samples': len(X_val),
                        'test_samples': len(X_test)
                    }
                )
                
    return results

def main(args):
    """Main function for auto-optimization module."""
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description='Auto-optimize models and trading strategies')
    parser.add_argument('--target', type=str, default='target',
                      help='Target column name (default: target)')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to data file (parquet format)')
    parser.add_argument('--models', nargs='+', type=str, 
                      default=['xgboost', 'lightgbm', 'random_forest'],
                      help='Model types to optimize')
    parser.add_argument('--metric', type=str, default='f1',
                      help='Metric to optimize (default: f1)')
    parser.add_argument('--trials', type=int, default=50,
                      help='Number of optimization trials (default: 50)')
    parser.add_argument('--output', type=str, default='models/optimized',
                      help='Output directory (default: models/optimized)')
    
    args = parser.parse_args(args)
    
    # Run model optimization
    optimize_models(
        data_path=args.data,
        model_types=args.models,
        target_column=args.target,
        metric=args.metric,
        n_trials=args.trials,
        output_dir=args.output
    )

if __name__ == '__main__':
    main(sys.argv[1:])

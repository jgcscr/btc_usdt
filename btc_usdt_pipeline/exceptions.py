"""
exceptions.py
Centralized custom exception hierarchy for the BTC/USDT pipeline project.
"""

class BTCUSDTError(Exception):
    """
    Base exception for all custom errors in the BTC/USDT pipeline.
    """
    pass

# --- Data Errors ---
class DataError(BTCUSDTError):
    """
    Base exception for data-related errors.
    """
    pass

class DataAlignmentError(DataError):
    """
    Raised when data and signals are misaligned or cannot be aligned.
    """
    pass

# --- Backtest Errors ---
class BacktestError(BTCUSDTError):
    """
    Base exception for backtesting errors.
    """
    pass

class ParameterValidationError(BacktestError):
    """
    Raised when invalid parameters are provided to backtest or related functions.
    """
    pass

# --- Optimization Errors ---
class OptimizationError(BTCUSDTError):
    """
    Raised for errors during optimization routines.
    """
    pass

# --- Model Errors ---
class ModelError(BTCUSDTError):
    """
    Base exception for model-related errors (training, saving, loading, etc.).
    """
    pass

class ModelTrainingError(ModelError):
    """
    Raised when model training fails.
    """
    pass

class ModelSavingError(ModelError):
    """
    Raised when saving a model fails.
    """
    pass

class ModelLoadingError(ModelError):
    """
    Raised when loading a model fails.
    """
    pass

# --- Configuration Errors ---
class ConfigError(BTCUSDTError):
    """
    Raised for configuration-related issues.
    """
    pass

# --- Workflow/Task Errors ---
class WorkflowError(BTCUSDTError):
    """
    Raised for errors in workflow or pipeline tasks.
    """
    pass

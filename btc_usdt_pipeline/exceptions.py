"""
exceptions.py
Centralized custom exception hierarchy for the BTC/USDT pipeline project.
"""

class BTCUSDTError(Exception):
    """Base exception for all custom errors in the BTC/USDT pipeline."""

# --- Data Errors ---
class DataError(BTCUSDTError):
    """Base exception for data-related errors."""

class DataAlignmentError(DataError):
    """Raised when data and signals are misaligned or cannot be aligned."""

# --- Backtest Errors ---
class BacktestError(BTCUSDTError):
    """Base exception for backtesting errors."""

class ParameterValidationError(BacktestError):
    """Raised when invalid parameters are provided to backtest or related functions."""

# --- Optimization Errors ---
class OptimizationError(BTCUSDTError):
    """
    Raised for errors during optimization routines.
    """
    """Raised for errors during optimization routines."""

# --- Model Errors ---
class ModelError(BTCUSDTError):
    """
    Base exception for model-related errors (training, saving, loading, etc.).
    """
    """Base exception for model-related errors (training, saving, loading, etc.)."""

class ModelTrainingError(ModelError):
    """
    Raised when model training fails.
    """
    """Raised when model training fails."""

class ModelSavingError(ModelError):
    """
    Raised when saving a model fails.
    """
    """Raised when saving a model fails."""

class ModelLoadingError(ModelError):
    """
    Raised when loading a model fails.
    """
    """Raised when loading a model fails."""

# --- Configuration Errors ---
class ConfigError(BTCUSDTError):
    """
    Raised for configuration-related issues.
    """
    """Raised for configuration-related issues."""

# --- Workflow/Task Errors ---
class WorkflowError(BTCUSDTError):
    """
    Raised for errors in workflow or pipeline tasks.
    """
    """Raised for errors in workflow or pipeline tasks."""

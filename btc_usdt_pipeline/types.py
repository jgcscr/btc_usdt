"""
Custom type aliases for the BTC/USDT trading pipeline.
"""
from typing import Any, Dict, List, Tuple, Union, Optional, TypeVar, Callable
import numpy as np
import pandas as pd

# Type aliases for common structures
DataFrame = pd.DataFrame
Series = pd.Series
ArrayLike = Union[np.ndarray, List[float], List[int]]

# SignalType: Used for model signals, can be float or array
SignalType = Dict[str, Union[float, np.ndarray]]

# Trade log entry and trade log
TradeLogEntry = Dict[str, Any]
TradeLogType = List[TradeLogEntry]

# Metrics dictionary
MetricsDict = Dict[str, Union[float, int, str, None]]

# Model input/output
ModelInputType = Union[pd.DataFrame, np.ndarray]
ModelOutputType = Union[np.ndarray, List[float], List[int]]

# Feature dictionary
FeatureDict = Dict[str, Union[float, int, np.ndarray, pd.Series]]

# Context for workflow pipelines
ContextType = Dict[str, Any]

# Cross-validation results
CVResultsType = List[Dict[str, Any]]

# TypeVars for generics
T = TypeVar('T')
U = TypeVar('U')

# Callable for generic model training
TrainFunc = Callable[..., Tuple[Any, MetricsDict]]

# Callable for generic prediction
PredictFunc = Callable[[Any, ModelInputType], ModelOutputType]

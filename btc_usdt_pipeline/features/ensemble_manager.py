"""
EnsembleManager: Combines predictions from multiple models based on their performance in the current market regime.
"""
import logging
from typing import Dict, Any, List, Optional, Callable, Union
import numpy as np
import pandas as pd
from .market_regime_detector import MarketRegimeDetector
from .model_selector_by_regime import ModelSelectorByRegime

logger = logging.getLogger("EnsembleManager")

class EnsembleManager:
    """
    Manages an ensemble of models, combining their predictions based on regime-aware performance.
    Supports: simple average, weighted average, voting.
    """
    def __init__(self,
                 regime_detector: Optional[MarketRegimeDetector] = None,
                 model_selector: Optional[ModelSelectorByRegime] = None,
                 ensemble_method: str = 'weighted_average'):
        self.regime_detector = regime_detector or MarketRegimeDetector()
        self.model_selector = model_selector or ModelSelectorByRegime(self.regime_detector)
        self.ensemble_method = ensemble_method
        # regime -> model_name -> {'model': model_obj, 'performance': float, 'trades': int}
        self.models: Dict[str, Dict[str, Any]] = {}  # model_name -> {'model': model_obj, ...}

    def add_model(self, model_name: str, model: Any):
        """Add a model to the ensemble."""
        self.models[model_name] = {'model': model}
        logger.info(f"Added model '{model_name}' to ensemble.")

    def remove_model(self, model_name: str):
        """Remove a model from the ensemble."""
        if model_name in self.models:
            del self.models[model_name]
            logger.info(f"Removed model '{model_name}' from ensemble.")
        else:
            logger.warning(f"Model '{model_name}' not found in ensemble.")

    def _get_regime(self, market_data: pd.DataFrame) -> str:
        try:
            regime = self.regime_detector.classify_regime(market_data.tail(50)).iloc[-1]
            return regime
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return 'normal'

    def _get_model_performances(self, regime: str) -> Dict[str, float]:
        """Get performance scores for all models in the current regime."""
        perf = {}
        regime_models = self.model_selector.regime_model_performance.get(regime, {})
        for model_name in self.models:
            if model_name in regime_models:
                perf[model_name] = regime_models[model_name]['performance']
            else:
                perf[model_name] = 0.0  # Default to 0 if no data
        return perf

    def predict(self, market_data: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """
        Get ensemble prediction and confidence score.
        Returns: {'prediction': value, 'confidence': float, 'details': ...}
        """
        regime = self._get_regime(market_data)
        performances = self._get_model_performances(regime)
        preds = {}
        for model_name, model_info in self.models.items():
            model = model_info['model']
            try:
                # Each model must implement a predict method
                pred = model.predict(market_data, **kwargs)
                preds[model_name] = float(pred)
            except Exception as e:
                logger.error(f"Model '{model_name}' prediction error: {e}")
        if not preds:
            logger.error("No model predictions available.")
            return {'prediction': None, 'confidence': 0.0, 'details': {}}
        method = self.ensemble_method
        if method == 'simple_average':
            ensemble_pred = np.mean(list(preds.values()))
            confidence = 1.0 / len(preds)
        elif method == 'weighted_average':
            weights = np.array([performances.get(m, 0.0) for m in preds])
            if np.sum(weights) == 0:
                weights = np.ones(len(preds)) / len(preds)
            else:
                weights = weights / np.sum(weights)
            ensemble_pred = float(np.dot(list(preds.values()), weights))
            confidence = float(np.max(weights))
        elif method == 'voting':
            # For classification: majority vote
            votes = list(preds.values())
            ensemble_pred = max(set(votes), key=votes.count)
            confidence = votes.count(ensemble_pred) / len(votes)
        else:
            logger.warning(f"Unknown ensemble method '{method}', defaulting to weighted_average.")
            weights = np.array([performances.get(m, 0.0) for m in preds])
            if np.sum(weights) == 0:
                weights = np.ones(len(preds)) / len(preds)
            else:
                weights = weights / np.sum(weights)
            ensemble_pred = float(np.dot(list(preds.values()), weights))
            confidence = float(np.max(weights))
        return {
            'prediction': ensemble_pred,
            'confidence': confidence,
            'details': {
                'regime': regime,
                'model_predictions': preds,
                'model_performances': performances,
                'ensemble_method': method,
                'weights': {m: float(w) for m, w in zip(preds, weights)} if method in ['weighted_average', 'unknown'] else None
            }
        }

    def set_ensemble_method(self, method: str):
        """Set the ensemble method (simple_average, weighted_average, voting)."""
        if method not in ['simple_average', 'weighted_average', 'voting']:
            logger.warning(f"Unknown ensemble method '{method}', defaulting to weighted_average.")
            self.ensemble_method = 'weighted_average'
        else:
            self.ensemble_method = method
        logger.info(f"Set ensemble method to '{self.ensemble_method}'.")

"""
ModelSelectorByRegime: Dynamically selects the best model for the current market regime using MarketRegimeDetector.
"""
import logging
from typing import Dict, Any, Optional, Callable
import pandas as pd
from .market_regime_detector import MarketRegimeDetector

logger = logging.getLogger("ModelSelectorByRegime")

class ModelSelectorByRegime:
    """
    Maintains mapping of market regimes to model performance, allows model registration, selection, and feedback.
    """
    def __init__(self, regime_detector: Optional[MarketRegimeDetector] = None):
        self.regime_detector = regime_detector or MarketRegimeDetector()
        # regime -> model_name -> {'model': model_obj, 'performance': float, 'trades': int}
        self.regime_model_performance: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def register_model(self, regime: str, model_name: str, model: Any, performance: float, trades: int = 0):
        """
        Register a model and its historical performance for a regime.
        """
        if regime not in self.regime_model_performance:
            self.regime_model_performance[regime] = {}
        self.regime_model_performance[regime][model_name] = {
            'model': model,
            'performance': performance,
            'trades': trades
        }
        logger.info(f"Registered model '{model_name}' for regime '{regime}' with performance {performance}.")

    def select_best_model(self, market_data: pd.DataFrame) -> Optional[Any]:
        """
        Detects current regime and returns the best model for it.
        Returns None if no model is registered for the detected regime.
        """
        try:
            regime = self.regime_detector.classify_regime(market_data.tail(50)).iloc[-1]
        except Exception as e:
            logger.error(f"Error detecting regime: {e}")
            return None
        models = self.regime_model_performance.get(regime, {})
        if not models:
            logger.warning(f"No models registered for regime '{regime}'.")
            return None
        # Select model with best (highest) performance
        best = max(models.items(), key=lambda x: x[1]['performance'])
        logger.info(f"Selected model '{best[0]}' for regime '{regime}' with performance {best[1]['performance']}.")
        return best[1]['model']

    def update_performance(self, regime: str, model_name: str, new_performance: float, trade_result: float):
        """
        Update model performance after a trade.
        Args:
            regime: The regime during the trade.
            model_name: The model used.
            new_performance: Updated performance metric (e.g., Sharpe, win rate).
            trade_result: PnL or other trade outcome.
        """
        try:
            model_info = self.regime_model_performance[regime][model_name]
            # Simple running average update
            n = model_info['trades']
            prev_perf = model_info['performance']
            updated_perf = (prev_perf * n + new_performance) / (n + 1)
            model_info['performance'] = updated_perf
            model_info['trades'] += 1
            logger.info(f"Updated performance for model '{model_name}' in regime '{regime}': {updated_perf:.4f} after trade result {trade_result}.")
        except KeyError:
            logger.error(f"Model '{model_name}' not registered for regime '{regime}'. Cannot update performance.")

    def get_registered_models(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Returns the current mapping of regimes to registered models and their stats.
        """
        return self.regime_model_performance

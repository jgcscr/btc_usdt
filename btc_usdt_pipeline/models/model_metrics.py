"""
Model performance tracking module for monitoring and comparing trained models.
"""
import os
import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from btc_usdt_pipeline.utils.logging_config import setup_logging

logger = setup_logging('model_metrics.log')

class ModelMetricsTracker:
    """
    Tracks and compares performance metrics for different models over time.
    Provides functionality to:
    - Store metrics for each model version
    - Track performance across market conditions
    - Select best models based on different criteria
    - Visualize performance trends
    """
    
    def __init__(self, metrics_db_path: str = 'data/model_metrics.json'):
        """
        Initialize the metrics tracker.
        
        Args:
            metrics_db_path: Path to the JSON file storing model metrics
        """
        self.metrics_db_path = metrics_db_path
        self.metrics_db = self._load_metrics_db()
        
    def _load_metrics_db(self) -> Dict[str, Any]:
        """Load metrics database from disk or create a new one."""
        try:
            if os.path.exists(self.metrics_db_path):
                with open(self.metrics_db_path, 'r') as f:
                    return json.load(f)
            else:
                # Create initial structure
                return {
                    'models': {},
                    'meta': {
                        'created_at': datetime.now().isoformat(),
                        'last_updated': datetime.now().isoformat(),
                        'version': '1.0'
                    }
                }
        except Exception as e:
            logger.error(f"Error loading metrics database: {e}")
            # Return empty database if loading fails
            return {
                'models': {},
                'meta': {
                    'created_at': datetime.now().isoformat(),
                    'last_updated': datetime.now().isoformat(),
                    'version': '1.0'
                }
            }
            
    def _save_metrics_db(self) -> bool:
        """Save metrics database to disk."""
        try:
            # Update last_updated timestamp
            self.metrics_db['meta']['last_updated'] = datetime.now().isoformat()
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.metrics_db_path), exist_ok=True)
            
            # Save the database
            with open(self.metrics_db_path, 'w') as f:
                json.dump(self.metrics_db, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error saving metrics database: {e}")
            return False
            
    def add_model_metrics(
        self,
        model_id: str,
        model_type: str,
        metrics: Dict[str, float],
        parameters: Dict[str, Any],
        training_date: Optional[datetime] = None,
        market_conditions: Optional[Dict[str, Any]] = None,
        dataset_info: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add performance metrics for a model.
        
        Args:
            model_id: Unique identifier for the model
            model_type: Type of model (e.g., 'xgboost', 'lstm')
            metrics: Dictionary of performance metrics
            parameters: Model hyperparameters and configuration
            training_date: When the model was trained
            market_conditions: Market conditions during training/evaluation
            dataset_info: Information about the dataset used
            
        Returns:
            Success status
        """
        try:
            # Initialize if this is the first entry for this model
            if model_id not in self.metrics_db['models']:
                self.metrics_db['models'][model_id] = {
                    'model_type': model_type,
                    'versions': []
                }
                
            # Create a new version entry
            version_data = {
                'version': len(self.metrics_db['models'][model_id]['versions']) + 1,
                'training_date': (training_date or datetime.now()).isoformat(),
                'metrics': metrics,
                'parameters': parameters
            }
            
            if market_conditions:
                version_data['market_conditions'] = market_conditions
                
            if dataset_info:
                version_data['dataset_info'] = dataset_info
                
            # Add the version data
            self.metrics_db['models'][model_id]['versions'].append(version_data)
            
            # Save the updated database
            return self._save_metrics_db()
            
        except Exception as e:
            logger.error(f"Error adding model metrics: {e}")
            return False
            
    def get_model_metrics(self, model_id: str, version: Optional[int] = None) -> Dict[str, Any]:
        """
        Retrieve metrics for a specific model and version.
        
        Args:
            model_id: Unique identifier for the model
            version: Specific version to retrieve (None for latest)
            
        Returns:
            Model metrics data
        """
        if model_id not in self.metrics_db['models']:
            logger.warning(f"Model {model_id} not found in metrics database")
            return {}
            
        versions = self.metrics_db['models'][model_id]['versions']
        
        if not versions:
            return {}
            
        if version is None:
            # Return the latest version
            return versions[-1]
        elif 1 <= version <= len(versions):
            # Return the specified version
            return versions[version - 1]
        else:
            logger.warning(f"Version {version} not found for model {model_id}")
            return {}
            
    def get_best_model(
        self,
        metric: str = 'accuracy',
        model_type: Optional[str] = None,
        min_training_date: Optional[datetime] = None,
        max_training_date: Optional[datetime] = None,
        market_condition: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Find the best model based on a specific metric and optional filters.
        
        Args:
            metric: Metric to use for comparison
            model_type: Filter by model type
            min_training_date: Minimum training date
            max_training_date: Maximum training date
            market_condition: Filter by market condition
            
        Returns:
            Best model information
        """
        best_model = None
        best_metric_value = None
        is_higher_better = metric not in ['loss', 'error', 'mse', 'rmse', 'mae']
        
        for model_id, model_data in self.metrics_db['models'].items():
            # Apply model type filter
            if model_type and model_data['model_type'] != model_type:
                continue
                
            for version_data in model_data['versions']:
                # Apply date filters
                if min_training_date or max_training_date:
                    training_date = datetime.fromisoformat(version_data['training_date'])
                    
                    if min_training_date and training_date < min_training_date:
                        continue
                        
                    if max_training_date and training_date > max_training_date:
                        continue
                        
                # Apply market condition filter
                if market_condition and 'market_conditions' in version_data:
                    # Simple exact match for now - could be enhanced with fuzzy matching
                    if not all(version_data['market_conditions'].get(k) == v 
                               for k, v in market_condition.items()):
                        continue
                        
                # Check if metric exists
                if metric not in version_data['metrics']:
                    continue
                    
                metric_value = version_data['metrics'][metric]
                
                # Update best model if this one is better
                if best_metric_value is None or \
                   (is_higher_better and metric_value > best_metric_value) or \
                   (not is_higher_better and metric_value < best_metric_value):
                    best_metric_value = metric_value
                    best_model = {
                        'model_id': model_id,
                        'model_type': model_data['model_type'],
                        'version': version_data['version'],
                        'metric_value': metric_value,
                        'training_date': version_data['training_date'],
                        'parameters': version_data['parameters']
                    }
                    
        return best_model or {}
    
    def compare_models(
        self,
        model_ids: List[str],
        metrics: List[str],
        versions: Optional[Dict[str, int]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple models across specified metrics.
        
        Args:
            model_ids: List of model IDs to compare
            metrics: List of metrics to compare
            versions: Dictionary mapping model_ids to versions (None for latest)
            
        Returns:
            DataFrame with comparison results
        """
        comparison = []
        
        for model_id in model_ids:
            if model_id not in self.metrics_db['models']:
                logger.warning(f"Model {model_id} not found, skipping")
                continue
                
            version = versions.get(model_id) if versions else None
            model_data = self.get_model_metrics(model_id, version)
            
            if not model_data:
                continue
                
            row = {
                'model_id': model_id,
                'model_type': self.metrics_db['models'][model_id]['model_type'],
                'version': model_data['version'],
                'training_date': model_data['training_date']
            }
            
            # Add metrics
            for metric in metrics:
                if metric in model_data['metrics']:
                    row[metric] = model_data['metrics'][metric]
                else:
                    row[metric] = None
                    
            comparison.append(row)
            
        return pd.DataFrame(comparison)
        
    def plot_metric_history(
        self,
        model_id: str,
        metric: str,
        save_path: Optional[str] = None
    ):
        """
        Plot the history of a specific metric for a model.
        
        Args:
            model_id: Model ID to plot
            metric: Metric to plot
            save_path: Path to save the plot (None to display)
        """
        if model_id not in self.metrics_db['models']:
            logger.warning(f"Model {model_id} not found")
            return
            
        versions = self.metrics_db['models'][model_id]['versions']
        
        if not versions:
            logger.warning(f"No versions found for model {model_id}")
            return
            
        # Extract data for plotting
        version_nums = []
        metric_values = []
        dates = []
        
        for version_data in versions:
            if metric in version_data['metrics']:
                version_nums.append(version_data['version'])
                metric_values.append(version_data['metrics'][metric])
                dates.append(datetime.fromisoformat(version_data['training_date']))
                
        if not version_nums:
            logger.warning(f"Metric {metric} not found in any version of model {model_id}")
            return
            
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(version_nums, metric_values, 'o-', linewidth=2)
        plt.title(f"{metric.capitalize()} History for Model {model_id}")
        plt.xlabel("Model Version")
        plt.ylabel(metric.capitalize())
        plt.xticks(version_nums)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add data labels
        for i, (v, m) in enumerate(zip(version_nums, metric_values)):
            plt.annotate(f"{m:.4f}", (v, m), textcoords="offset points", 
                         xytext=(0, 5), ha='center')
            
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    def compare_model_parameters(
        self, 
        model_ids: List[str],
        versions: Optional[Dict[str, int]] = None
    ) -> pd.DataFrame:
        """
        Compare parameters across different models.
        
        Args:
            model_ids: List of model IDs to compare
            versions: Dictionary mapping model_ids to versions (None for latest)
            
        Returns:
            DataFrame with parameter comparison
        """
        params_data = []
        
        for model_id in model_ids:
            if model_id not in self.metrics_db['models']:
                continue
                
            version = versions.get(model_id) if versions else None
            model_data = self.get_model_metrics(model_id, version)
            
            if not model_data or 'parameters' not in model_data:
                continue
                
            # Flatten parameters for this model
            flat_params = self._flatten_dict(model_data['parameters'], prefix='')
            flat_params['model_id'] = model_id
            flat_params['version'] = model_data['version']
            flat_params['model_type'] = self.metrics_db['models'][model_id]['model_type']
            
            params_data.append(flat_params)
            
        return pd.DataFrame(params_data)
    
    def _flatten_dict(self, d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        """Helper method to flatten nested dictionary for parameter comparison."""
        result = {}
        
        for k, v in d.items():
            key = f"{prefix}{k}" if prefix else k
            
            if isinstance(v, dict):
                # Recursive flattening for nested dicts
                nested = self._flatten_dict(v, f"{key}.")
                result.update(nested)
            else:
                result[key] = v
                
        return result
        
    def get_performance_by_market_condition(
        self,
        condition_key: str,
        metrics: List[str]
    ) -> pd.DataFrame:
        """
        Analyze model performance across different market conditions.
        
        Args:
            condition_key: Market condition key to analyze
            metrics: List of metrics to include
            
        Returns:
            DataFrame with performance by market condition
        """
        condition_data = []
        
        for model_id, model_data in self.metrics_db['models'].items():
            for version_data in model_data['versions']:
                if 'market_conditions' in version_data and condition_key in version_data['market_conditions']:
                    condition_value = version_data['market_conditions'][condition_key]
                    
                    row = {
                        'model_id': model_id,
                        'model_type': model_data['model_type'],
                        'version': version_data['version'],
                        'condition_value': condition_value
                    }
                    
                    # Add metrics
                    for metric in metrics:
                        if metric in version_data['metrics']:
                            row[metric] = version_data['metrics'][metric]
                            
                    condition_data.append(row)
                    
        return pd.DataFrame(condition_data)
        
    def plot_performance_by_condition(
        self,
        condition_key: str,
        metric: str,
        model_type: Optional[str] = None,
        save_path: Optional[str] = None
    ):
        """
        Plot model performance by market condition.
        
        Args:
            condition_key: Market condition to analyze
            metric: Performance metric to plot
            model_type: Filter by model type
            save_path: Path to save the plot (None to display)
        """
        df = self.get_performance_by_market_condition(condition_key, [metric])
        
        if df.empty:
            logger.warning(f"No data found for condition {condition_key}")
            return
            
        if model_type:
            df = df[df['model_type'] == model_type]
            
        # Group by condition value and model type, calculate mean performance
        grouped = df.groupby(['condition_value', 'model_type'])[metric].mean().reset_index()
        
        plt.figure(figsize=(12, 6))
        
        # Plot as grouped bar chart
        sns.barplot(x='condition_value', y=metric, hue='model_type', data=grouped)
        
        plt.title(f"{metric.capitalize()} by {condition_key.capitalize()} Condition")
        plt.xlabel(condition_key.capitalize())
        plt.ylabel(metric.capitalize())
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
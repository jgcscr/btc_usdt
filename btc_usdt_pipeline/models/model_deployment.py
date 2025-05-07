import os
import pickle
import json
import time
from typing import Any, Dict, Optional, List
import numpy as np
import pandas as pd
from btc_usdt_pipeline.features.feature_pipeline import FeaturePipeline
from btc_usdt_pipeline.utils.serialization import to_json
from btc_usdt_pipeline.utils.helpers import setup_logger
from btc_usdt_pipeline.types import FeatureDict, MetricsDict

logger = setup_logger('model_deployment.log')

class ModelDeploymentError(Exception):
    pass

class ModelPackage:
    """
    Packages a trained model with its preprocessing pipeline and metadata.
    Supports versioning and A/B testing.
    """
    def __init__(self, model: Any, pipeline: FeaturePipeline, version: str, metadata: Optional[Dict[str, Any]] = None):
        self.model: Any = model
        self.pipeline: FeaturePipeline = pipeline
        self.version: str = version
        self.metadata: Dict[str, Any] = metadata or {}

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        try:
            X = self.pipeline.transform(df)
            preds = self.model.predict(X)
            return preds
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise ModelDeploymentError(f"Prediction failed: {e}")

    def save(self, path: str) -> None:
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"ModelPackage saved to {path}")

    @staticmethod
    def load(path: str) -> 'ModelPackage':
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        logger.info(f"ModelPackage loaded from {path}")
        return obj

class ModelRegistry:
    """
    Handles model versioning, A/B testing, and rollback.
    """
    REGISTRY_PATH: str = os.path.join(os.path.dirname(__file__), '../../models/registry.json')

    def __init__(self) -> None:
        os.makedirs(os.path.dirname(self.REGISTRY_PATH), exist_ok=True)
        self._load_registry()

    def _load_registry(self) -> None:
        if os.path.exists(self.REGISTRY_PATH):
            with open(self.REGISTRY_PATH, 'r') as f:
                self.registry = json.load(f)
        else:
            self.registry = {'active': None, 'versions': {}, 'ab_test': {}}

    def _save_registry(self) -> None:
        with open(self.REGISTRY_PATH, 'w') as f:
            json.dump(self.registry, f, indent=2)

    def register_model(self, version: str, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.registry['versions'][version] = {'path': path, 'metadata': metadata or {}, 'timestamp': time.time()}
        self._save_registry()
        logger.info(f"Registered model version {version} at {path}")

    def set_active(self, version: str) -> None:
        if version not in self.registry['versions']:
            raise ModelDeploymentError(f"Version {version} not found in registry.")
        self.registry['active'] = version
        self._save_registry()
        logger.info(f"Set active model version to {version}")

    def get_active(self) -> Optional[str]:
        return self.registry.get('active')

    def get_model_path(self, version: Optional[str] = None) -> Optional[str]:
        version = version or self.get_active()
        if version and version in self.registry['versions']:
            return self.registry['versions'][version]['path']
        return None

    def rollback(self, to_version: str) -> None:
        self.set_active(to_version)
        logger.info(f"Rolled back to model version {to_version}")

    def list_versions(self) -> List[str]:
        return list(self.registry['versions'].keys())

    def set_ab_test(self, version_a: str, version_b: str, ratio: float = 0.5) -> None:
        self.registry['ab_test'] = {'A': version_a, 'B': version_b, 'ratio': ratio}
        self._save_registry()
        logger.info(f"A/B test set: {version_a} vs {version_b} (ratio {ratio})")

    def get_ab_test(self) -> Dict[str, Any]:
        return self.registry.get('ab_test', {})

# --- Prediction API ---
def predict_api(input_data: pd.DataFrame, registry: ModelRegistry) -> Dict[str, Any]:
    """
    Clean prediction API with error handling and A/B testing support.
    """
    ab_test = registry.get_ab_test()
    if ab_test and ab_test.get('A') and ab_test.get('B'):
        # Simple random assignment for A/B
        version = np.random.choice([ab_test['A'], ab_test['B']], p=[ab_test['ratio'], 1-ab_test['ratio']])
    else:
        version = registry.get_active()
    model_path = registry.get_model_path(version)
    if not model_path:
        logger.error("No active model found for prediction.")
        return {'error': 'No active model'}
    model_pkg = ModelPackage.load(model_path)
    try:
        preds = model_pkg.predict(input_data)
        return {'predictions': preds.tolist(), 'version': version}
    except Exception as e:
        logger.error(f"Prediction API error: {e}")
        return {'error': str(e)}

# --- Health Check Endpoint ---
def health_check(registry: ModelRegistry) -> Dict[str, Any]:
    active = registry.get_active()
    versions = registry.list_versions()
    return {'status': 'ok', 'active_version': active, 'available_versions': versions}

# --- CLI ---
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Model Deployment CLI")
    parser.add_argument('--deploy', type=str, help='Path to model package to deploy')
    parser.add_argument('--version', type=str, help='Model version for deployment/rollback')
    parser.add_argument('--set-active', type=str, help='Set active model version')
    parser.add_argument('--rollback', type=str, help='Rollback to previous version')
    parser.add_argument('--list', action='store_true', help='List all model versions')
    parser.add_argument('--ab-test', nargs=2, help='Set A/B test: version_a version_b')
    parser.add_argument('--health', action='store_true', help='Health check endpoint')
    args = parser.parse_args()
    registry = ModelRegistry()
    if args.deploy and args.version:
        registry.register_model(args.version, args.deploy)
        registry.set_active(args.version)
        print(f"Deployed and activated model version {args.version}")
    if args.set_active:
        registry.set_active(args.set_active)
        print(f"Set active model version to {args.set_active}")
    if args.rollback:
        registry.rollback(args.rollback)
        print(f"Rolled back to model version {args.rollback}")
    if args.list:
        print("Available model versions:", registry.list_versions())
    if args.ab_test:
        registry.set_ab_test(args.ab_test[0], args.ab_test[1])
        print(f"A/B test set: {args.ab_test[0]} vs {args.ab_test[1]}")
    if args.health:
        print(health_check(registry))

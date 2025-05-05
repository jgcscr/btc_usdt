import os
import json
import time
import logging
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from datetime import datetime

logger = logging.getLogger("btc_usdt_pipeline.models.model_monitoring")

class PredictionMonitor:
    """
    Logs predictions, features, and actual outcomes for monitoring and auditing.
    """
    def __init__(self, log_path: str = "./logs/prediction_log.jsonl"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def log_prediction(self, prediction: Any, features: Dict[str, Any]) -> str:
        pred_id = f"pred_{int(time.time() * 1000)}_{np.random.randint(1e6)}"
        entry = {
            "id": pred_id,
            "timestamp": datetime.utcnow().isoformat(),
            "prediction": prediction,
            "features": features
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info(f"Logged prediction {pred_id}")
        return pred_id

    def log_outcome(self, prediction_id: str, actual_outcome: Any):
        # For simplicity, append outcome to a separate file
        outcome_path = self.log_path.replace("prediction_log", "outcome_log")
        entry = {
            "id": prediction_id,
            "timestamp": datetime.utcnow().isoformat(),
            "actual": actual_outcome
        }
        with open(outcome_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        logger.info(f"Logged outcome for {prediction_id}")

    def load_logs(self, period: str = "day") -> pd.DataFrame:
        # Loads predictions and outcomes for the given period ("day", "week", etc.)
        if not os.path.exists(self.log_path):
            return pd.DataFrame()
        df = pd.read_json(self.log_path, lines=True)
        if period == "day":
            cutoff = pd.Timestamp.utcnow().normalize()
            df = df[pd.to_datetime(df["timestamp"]) >= cutoff]
        return df

class FeatureDriftDetector:
    """
    Detects feature drift using statistical tests (KS-test for continuous, Chi-squared for categorical).
    """
    def detect_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame, threshold: float = 0.05) -> Dict[str, Any]:
        drift_report = {}
        for col in reference_data.columns:
            if pd.api.types.is_numeric_dtype(reference_data[col]):
                stat, p = ks_2samp(reference_data[col].dropna(), current_data[col].dropna())
                drift = p < threshold
                drift_report[col] = {"test": "ks", "p_value": p, "drift": drift}
            else:
                # Categorical: Chi-squared
                ref_counts = reference_data[col].value_counts()
                cur_counts = current_data[col].value_counts()
                all_cats = ref_counts.index.union(cur_counts.index)
                ref_freq = ref_counts.reindex(all_cats, fill_value=0)
                cur_freq = cur_counts.reindex(all_cats, fill_value=0)
                stat, p, _, _ = chi2_contingency([ref_freq, cur_freq])
                drift = p < threshold
                drift_report[col] = {"test": "chi2", "p_value": p, "drift": drift}
        return drift_report

class PerformanceTracker:
    """
    Tracks and computes model performance metrics over time.
    """
    def __init__(self, monitor: PredictionMonitor):
        self.monitor = monitor

    def calculate_metrics(self, period: str = "day") -> Dict[str, Any]:
        df = self.monitor.load_logs(period=period)
        if df.empty:
            return {"count": 0}
        # Assume binary classification for example
        preds = df["prediction"].apply(lambda x: x if isinstance(x, int) else x[0] if isinstance(x, (list, np.ndarray)) else None)
        actuals = df.get("actual")
        if actuals is None or actuals.isnull().all():
            return {"count": len(df)}
        acc = np.mean(preds == actuals)
        return {"count": len(df), "accuracy": acc}

# --- Alerting Mechanism ---
def send_alert(message: str, context: Optional[Dict[str, Any]] = None, method: str = "log"):
    alert_msg = f"ALERT: {message} | Context: {context}"
    if method == "log":
        logger.warning(alert_msg)
    elif method == "email":
        # Placeholder: integrate with email system
        logger.warning(f"EMAIL ALERT: {alert_msg}")
    elif method == "webhook":
        # Placeholder: integrate with webhook system
        logger.warning(f"WEBHOOK ALERT: {alert_msg}")
    else:
        logger.warning(f"UNKNOWN ALERT METHOD: {alert_msg}")

# --- Periodic Check Utility ---
def periodic_monitoring_check(monitor: PredictionMonitor, tracker: PerformanceTracker, drift_detector: FeatureDriftDetector, reference_data: pd.DataFrame, metric_threshold: float = 0.7, drift_threshold: float = 0.05):
    metrics = tracker.calculate_metrics()
    if metrics.get("accuracy", 1.0) < metric_threshold:
        send_alert(f"Model accuracy dropped below threshold: {metrics}", context=metrics)
    # Load current data for drift detection
    df = monitor.load_logs()
    if not df.empty:
        current_data = pd.DataFrame(list(df["features"]))
        drift_report = drift_detector.detect_drift(reference_data, current_data, threshold=drift_threshold)
        drifted = {k: v for k, v in drift_report.items() if v["drift"]}
        if drifted:
            send_alert(f"Feature drift detected: {drifted}", context=drift_report)
    logger.info(f"Periodic monitoring check complete. Metrics: {metrics}")

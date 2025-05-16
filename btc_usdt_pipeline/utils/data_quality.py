import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional
import matplotlib.pyplot as plt

logger = logging.getLogger("btc_usdt_pipeline.utils.data_quality")

# --- Data Quality Checks ---
def detect_missing_data(df: pd.DataFrame, critical_cols: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Checks for missing values and completeness in the DataFrame.
    Returns a dict with missing counts and completeness ratios.
    """
    critical_cols = critical_cols or df.columns.tolist()
    missing = df[critical_cols].isnull().sum()
    completeness = 1 - missing / len(df)
    result = {
        'missing_counts': missing.to_dict(),
        'completeness': completeness.to_dict(),
        'total_missing': int(missing.sum()),
        'rows_with_any_missing': int(df[critical_cols].isnull().any(axis=1).sum())
    }
    if result['total_missing'] > 0:
        logger.warning(f"Missing data detected: {result}")
    return result

def detect_price_anomalies(df: pd.DataFrame, price_col: str = 'close', threshold: float = 5.0) -> Dict[str, Any]:
    """
    Detects price anomalies: gaps, spikes, and zero/negative values.
    Returns a dict with anomaly indices and summary.
    """
    anomalies = {}
    # Zero or negative prices
    zero_idx = df.index[df[price_col] <= 0].tolist()
    # Spikes: price change > threshold * std
    price_diff = df[price_col].diff().abs()
    spike_std = price_diff.std()
    spike_idx = df.index[price_diff > threshold * spike_std].tolist()
    # Gaps: large time difference between consecutive rows
    if isinstance(df.index, pd.DatetimeIndex):
        time_deltas = df.index.to_series().diff().dt.total_seconds().fillna(0)
        median_delta = time_deltas[time_deltas > 0].median() if (time_deltas > 0).any() else 0
        gap_idx = df.index[time_deltas > 3 * median_delta].tolist()
    else:
        gap_idx = []
    anomalies['zero_or_negative'] = zero_idx
    anomalies['spikes'] = spike_idx
    anomalies['gaps'] = gap_idx
    anomalies['summary'] = {k: len(v) for k, v in anomalies.items() if isinstance(v, list)}
    if any(len(v) > 0 for v in anomalies.values() if isinstance(v, list)):
        logger.warning(f"Price anomalies detected: {anomalies['summary']}")
    return anomalies

def analyze_feature_distributions(df: pd.DataFrame, reference: Optional[pd.DataFrame] = None, threshold: float = 0.1) -> Dict[str, Any]:
    """
    Compares feature distributions to a reference (or to themselves) to detect shifts.
    Returns a dict of features with significant distribution changes.
    """
    from scipy.stats import ks_2samp
    shifts = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        if reference is not None and col in reference.columns:
            stat, p = ks_2samp(df[col].dropna(), reference[col].dropna())
            if p < threshold:
                shifts[col] = {'ks_stat': stat, 'p_value': p}
    if shifts:
        logger.warning(f"Feature distribution shifts detected: {shifts}")
    return shifts

def verify_time_continuity(df: pd.DataFrame, freq: Optional[str] = None) -> Dict[str, Any]:
    """
    Checks for time series continuity and consistent sampling intervals.
    Returns a dict with gap info and sampling stats.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return {'error': 'Index is not DatetimeIndex'}
    time_deltas = df.index.to_series().diff().dt.total_seconds().fillna(0)
    median_delta = time_deltas[time_deltas > 0].median() if (time_deltas > 0).any() else 0
    gaps = df.index[time_deltas > 3 * median_delta].tolist()
    stats = {
        'median_delta_sec': median_delta,
        'max_delta_sec': time_deltas.max(),
        'num_gaps': len(gaps),
        'gap_indices': gaps
    }
    if stats['num_gaps'] > 0:
        logger.warning(f"Time continuity issues detected: {stats}")
    return stats

# --- Data Repair Utilities ---
def fill_gaps(df: pd.DataFrame, method: str = 'linear') -> pd.DataFrame:
    """
    Fills missing values and time gaps using interpolation.
    """
    df_filled = df.copy()
    if isinstance(df_filled.index, pd.DatetimeIndex):
        full_range = pd.date_range(df_filled.index.min(), df_filled.index.max(), freq=pd.infer_freq(df_filled.index) or 'min')
        df_filled = df_filled.reindex(full_range)
    df_filled = df_filled.interpolate(method=method).fillna(method='bfill').fillna(method='ffill')
    return df_filled

def smooth_outliers(df: pd.DataFrame, cols: Optional[List[str]] = None, z_thresh: float = 4.0) -> pd.DataFrame:
    """
    Smooths outliers in specified columns using z-score thresholding.
    """
    df_sm = df.copy()
    cols = cols or df_sm.select_dtypes(include=[np.number]).columns.tolist()
    for col in cols:
        vals = df_sm[col]
        z = (vals - vals.mean()) / (vals.std() + 1e-9)
        outliers = z.abs() > z_thresh
        df_sm.loc[outliers, col] = np.nan
    df_sm = df_sm.interpolate().fillna(method='bfill').fillna(method='ffill')
    return df_sm

def standardize_sampling(df: pd.DataFrame, freq: str = '1min') -> pd.DataFrame:
    """
    Resamples the DataFrame to a consistent time interval.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError('Index must be DatetimeIndex for resampling.')
    df_resampled = df.resample(freq).ffill()
    return df_resampled

# --- Quality Assessment Reports ---
def plot_data_quality_metrics(metrics: Dict[str, Any], title: str = 'Data Quality Metrics'):
    """
    Visualizes data quality metrics over time.
    """
    if 'completeness' in metrics:
        plt.figure(figsize=(10, 4))
        plt.plot(list(metrics['completeness'].keys()), list(metrics['completeness'].values()), marker='o')
        plt.title(title)
        plt.xlabel('Column')
        plt.ylabel('Completeness')
        plt.ylim(0, 1.05)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

def highlight_problem_periods(df: pd.DataFrame, anomalies: Dict[str, Any], title: str = 'Problematic Data Periods'):
    """
    Highlights periods with detected anomalies on a price/time chart.
    """
    plt.figure(figsize=(14, 5))
    if 'close' in df.columns:
        plt.plot(df.index, df['close'], label='Close Price')
    for k, idxs in anomalies.items():
        if isinstance(idxs, list) and idxs:
            plt.scatter(idxs, df.loc[idxs, 'close'] if 'close' in df.columns else [np.nan]*len(idxs), label=f'{k} anomaly', marker='x')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.tight_layout()
    plt.show()

def recommend_cleaning_steps(metrics: Dict[str, Any], anomalies: Dict[str, Any]) -> List[str]:
    """
    Recommends data cleaning steps based on detected issues.
    """
    recs = []
    if metrics.get('total_missing', 0) > 0:
        recs.append('Fill or interpolate missing values.')
    if any(anomalies.get(k) for k in ['spikes', 'zero_or_negative']):
        recs.append('Smooth outliers and remove zero/negative prices.')
    if anomalies.get('gaps'):
        recs.append('Fill time gaps and standardize sampling.')
    if not recs:
        recs.append('No major issues detected. Proceed to modeling.')
    return recs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterator, Tuple, List, Dict, Any, Callable, Optional

class TimeSeriesSplit:
    """
    Time series cross-validator that provides train/test indices for each split.
    Splits data so that the training set is always before the test set in time.
    """
    def __init__(self, n_splits: int = 5, test_size: Optional[int] = None, max_train_size: Optional[int] = None, step_size: int = 1):
        self.n_splits = n_splits
        self.test_size = test_size
        self.max_train_size = max_train_size
        self.step_size = step_size

    def split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        n_samples = len(X)
        test_size = self.test_size or n_samples // (self.n_splits + 1)
        for i in range(self.n_splits):
            train_end = test_size + i * self.step_size
            test_start = train_end
            test_end = test_start + test_size
            if self.max_train_size:
                train_start = max(0, train_end - self.max_train_size)
            else:
                train_start = 0
            if test_end > n_samples:
                break
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            yield train_idx, test_idx

# Rolling window split generator
def rolling_window_split(df: pd.DataFrame, window_size: int, step_size: int, test_size: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    n = len(df)
    for start in range(0, n - window_size - test_size + 1, step_size):
        train_idx = np.arange(start, start + window_size)
        test_idx = np.arange(start + window_size, start + window_size + test_size)
        yield train_idx, test_idx

# Expanding window split generator
def expanding_window_split(df: pd.DataFrame, initial_train_size: int, test_size: int, step_size: int) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    n = len(df)
    for end in range(initial_train_size, n - test_size + 1, step_size):
        train_idx = np.arange(0, end)
        test_idx = np.arange(end, end + test_size)
        yield train_idx, test_idx

def walk_forward_validation(model_func: Callable, df: pd.DataFrame, window_generator: Callable, param_grid: Dict[str, List[Any]], target_col: str, features: List[str]) -> List[Dict[str, Any]]:
    """
    Perform walk-forward validation using a model function and parameter grid.
    Returns a list of dicts with performance for each split/parameter set.
    """
    from sklearn.model_selection import ParameterGrid
    results = []
    for params in ParameterGrid(param_grid):
        for train_idx, test_idx in window_generator(df):
            train, test = df.iloc[train_idx], df.iloc[test_idx]
            model = model_func(**params)
            X_train, y_train = train[features], train[target_col]
            X_test, y_test = test[features], test[target_col]
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            score = np.mean(preds == y_test)
            results.append({
                'params': params,
                'train_idx': train_idx,
                'test_idx': test_idx,
                'score': score
            })
    return results

def walk_forward_backtest(model_func: Callable, df: pd.DataFrame, window_generator: Callable, param_grid: Dict[str, List[Any]], target_col: str, features: List[str]) -> List[Dict[str, Any]]:
    """
    Run walk-forward backtest using the provided model function and window generator.
    Returns a list of dicts with performance and parameters for each split.
    """
    return walk_forward_validation(model_func, df, window_generator, param_grid, target_col, features)

def calculate_stability_metrics(cv_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate performance stability metrics across CV splits.
    Returns mean, std, min, max, and coefficient of variation for scores.
    """
    scores = [r['score'] for r in cv_results]
    return {
        'mean_score': np.mean(scores),
        'std_score': np.std(scores),
        'min_score': np.min(scores),
        'max_score': np.max(scores),
        'cv_score': np.std(scores) / np.mean(scores) if np.mean(scores) != 0 else np.nan
    }

def plot_cv_splits(df: pd.DataFrame, splits: List[Tuple[np.ndarray, np.ndarray]], title: str = "CV Splits"):
    plt.figure(figsize=(12, 2 + len(splits)))
    for i, (train_idx, test_idx) in enumerate(splits):
        plt.scatter(train_idx, [i]*len(train_idx), color='blue', marker='|', label='Train' if i==0 else "")
        plt.scatter(test_idx, [i]*len(test_idx), color='red', marker='|', label='Test' if i==0 else "")
    plt.title(title)
    plt.xlabel('Time Index')
    plt.ylabel('Split')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_cv_performance(cv_results: List[Dict[str, Any]], title: str = "CV Performance"):
    scores = [r['score'] for r in cv_results]
    plt.figure(figsize=(10, 4))
    plt.plot(scores, marker='o')
    plt.title(title)
    plt.xlabel('Split')
    plt.ylabel('Score')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_parameter_stability(cv_results: List[Dict[str, Any]], param_name: str, title: str = "Parameter Stability"):
    params = [r['params'][param_name] for r in cv_results]
    scores = [r['score'] for r in cv_results]
    plt.figure(figsize=(10, 4))
    plt.scatter(params, scores, c=scores, cmap='viridis', marker='o')
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel('Score')
    plt.colorbar(label='Score')
    plt.tight_layout()
    plt.show()

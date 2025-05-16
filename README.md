# BTC/USDT Trading Pipeline

## Project Purpose
This project provides a robust, modular pipeline for quantitative trading research and backtesting on the BTC/USDT pair. It supports data fetching, feature engineering, model training, optimization, and backtesting, with a focus on maintainability, reproducibility, and extensibility.

## Architecture Overview
- **Data Layer**: Fetches and manages historical market data (see `btc_usdt_pipeline/data/`).
- **Feature Engineering**: Computes technical indicators and engineered features (`features/compute_features.py`).
- **Model Layer**: Trains and manages ML models for signal generation (`models/`).
- **Optimization**: Hyperparameter and feature optimization using Optuna (`optimize/`).
- **Backtesting**: Event-driven backtest engine for evaluating strategies (`trading/backtest.py`).
- **Utilities**: Helpers for logging, serialization, memory optimization, and more (`utils/`).

## Main Workflow
1. **Fetch Data**: Download historical data from Binance using `scripts/fetch_data.py`.
2. **Compute Features**: Generate features and save enriched data with `scripts/compute_features.py`.
3. **Train Models**: Train ML models using `scripts/train_additional_models.py` or `models/train.py`.
4. **Optimize**: Run hyperparameter optimization with `scripts/auto_optimize.py`.
5. **Backtest**: Evaluate strategies using `scripts/backtest.py`.
6. **Analyze Results**: Review logs, equity curves, and trade logs in the `logs/` and `results/` folders.

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Fetch data
python -m scripts.fetch_data

# Compute features
python -m scripts.compute_features

# Train models
python -m scripts.train_additional_models

# Run optimization
python -m scripts.auto_optimize

# Backtest
python -m scripts.backtest
```

## Dependencies
- Python 3.8+
- pandas, numpy, scikit-learn, xgboost, optuna, joblib, matplotlib, requests, etc.
- See `requirements.txt` for the full list.

## Installation
```bash
git clone <repo-url>
cd btc_usdt
pip install -r requirements.txt
```

## Notes
- All logs are saved in the `logs/` directory.
- Data files are stored in the `data/` directory.
- For Colab/Google Drive support, see `utils/colab_utils.py`.

---

For more details, see the code documentation and comments in each module.

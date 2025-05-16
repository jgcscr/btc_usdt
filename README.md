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
- **Vertex AI & KFP Integration**: End-to-end ML pipeline orchestration using Google Cloud Vertex AI Pipelines and Kubeflow Pipelines (KFP). Artifacts and configs are managed in Google Cloud Storage (GCS).

## Main Workflow
### Local/Scripted Workflow
1. **Fetch Data**: Download historical data from Binance using `scripts/fetch_data.py`.
2. **Compute Features**: Generate features and save enriched data with `scripts/compute_features.py`.
3. **Train Models**: Train ML models using `scripts/train_additional_models.py` or `models/train.py`.
4. **Optimize**: Run hyperparameter optimization with `scripts/auto_optimize.py`.
5. **Backtest**: Evaluate strategies using `scripts/backtest.py`.
6. **Analyze Results**: Review logs, equity curves, and trade logs in the `logs/` and `results/` folders.

### Vertex AI / KFP Workflow
1. **Setup**: Configure GCP project, region, and GCS bucket in the Jupyter notebook (`vertex_ai_trading_pipeline.ipynb`).
2. **Upload Configs**: Place your YAML config files in the GCS bucket (e.g., `gs://<your-bucket>/configs/`).
3. **KFP Component**: Use the custom KFP component for Binance kline data fetching (`btc_usdt_pipeline/components/fetch_binance_data_component.py`).
4. **Pipeline Definition**: Define and compile the pipeline in the notebook, using the KFP DSL and referencing the GCS pipeline root.
5. **Run Pipeline**: Submit and monitor pipeline runs via Vertex AI Pipelines in the Google Cloud Console.
6. **Artifacts**: All data, models, and pipeline outputs are stored in GCS (see `CONFIG_GCS_PATH`, `RAW_DATA_GCS_PATH`, etc.).

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# (Local) Fetch data
python -m scripts.fetch_data

# (Local) Compute features
python -m scripts.compute_features

# (Local) Train models
python -m scripts.train_additional_models

# (Local) Run optimization
python -m scripts.auto_optimize

# (Local) Backtest
python -m scripts.backtest
```

### Vertex AI / KFP Pipeline (Cloud)
1. Open `vertex_ai_trading_pipeline.ipynb` in Vertex AI Workbench.
2. Run the setup cells to initialize Vertex AI, GCS, and configuration management.
3. Compile and submit the pipeline using the provided notebook cells.
4. Monitor pipeline runs and view outputs in the Vertex AI Console.

## Dependencies
- Python 3.8+
- pandas, numpy, scikit-learn, xgboost, optuna, joblib, matplotlib, requests, pyarrow, google-cloud-aiplatform, kfp, etc.
- See `requirements.txt` for the full list.

## Installation
```bash
git clone <repo-url>
cd btc_usdt
pip install -r requirements.txt
```

## Notes
- All logs are saved in the `logs/` directory.
- Data files are stored in the `data/` directory (local) or in GCS (cloud pipeline).
- For Colab/Google Drive support, see `utils/colab_utils.py`.
- For Vertex AI/KFP, see the notebook and `btc_usdt_pipeline/components/` for reusable pipeline components.

For more details, see the code documentation and comments in each module.
# Test modification

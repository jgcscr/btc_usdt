import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from btc_usdt_pipeline.models.train import save_model

# Basic test for save_model function (using a dummy object)
def test_save_model(tmp_path):
    dummy_model = {"param": 1}
    model_path = tmp_path / "test_model.joblib"
    save_model(dummy_model, model_path)
    assert model_path.exists()

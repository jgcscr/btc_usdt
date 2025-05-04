import pytest
from btc_usdt_pipeline.models.manage import ModelManager

# Basic test to check if ModelManager initializes
def test_model_manager_init(mocker):
    # Mock load_models to prevent actual file loading during init
    mocker.patch.object(ModelManager, 'load_models', return_value=None)
    manager = ModelManager()
    assert isinstance(manager.models, dict)

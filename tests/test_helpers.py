import pytest
import pandas as pd
import numpy as np
from btc_usdt_pipeline.utils.helpers import split_data

# Basic test for split_data function
def test_split_data():
    df = pd.DataFrame(np.random.rand(100, 3), columns=['A', 'B', 'C'])
    train_df, val_df, test_df = split_data(df, train_frac=0.7, val_frac=0.15)
    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15
    assert train_df.index.max() < val_df.index.min()
    assert val_df.index.max() < test_df.index.min()

# Test split_data with invalid fractions
def test_split_data_invalid_frac():
    df = pd.DataFrame(np.random.rand(10, 1))
    with pytest.raises(ValueError):
        split_data(df, train_frac=0.8, val_frac=0.3) # Sum > 1
    with pytest.raises(ValueError):
        split_data(df, train_frac=1.1, val_frac=0.1) # train_frac > 1

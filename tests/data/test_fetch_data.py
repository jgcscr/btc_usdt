import pytest
import pandas as pd
from datetime import datetime
from btc_usdt_pipeline.data.fetch_data import get_klines, fetch_historical_data
import requests

# Basic test for get_klines (mocking requests.get)
def test_get_klines_success(mocker):
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    # Simulate Binance API response format
    mock_response.json.return_value = [
        [1678886400000, "25000", "25100", "24900", "25050", "100", 1678886459999, "2505000", 50, "60", "1503000", "0"],
        [1678886460000, "25050", "25150", "25000", "25100", "120", 1678886519999, "3015000", 60, "70", "1758500", "0"]
    ]
    mocker.patch('requests.get', return_value=mock_response)

    klines = get_klines("BTCUSDT", "1m", start_time=1678886400000, limit=2)
    assert klines is not None
    assert len(klines) == 2
    assert klines[0][0] == 1678886400000
    assert klines[1][4] == "25100"

def test_get_klines_api_error(mocker):
    mock_response = mocker.Mock()
    mock_response.raise_for_status.side_effect = requests.exceptions.RequestException("API Error")
    mocker.patch('requests.get', return_value=mock_response)

    klines = get_klines("BTCUSDT", "1m")
    assert klines is None

# Very basic test for fetch_historical_data structure (mocking get_klines)
def test_fetch_historical_data_structure(mocker):
     # Mock get_klines to return data the first time, then empty list
    mock_data = [
        [1678886400000, "25000", "25100", "24900", "25050", "100", 1678886459999, "2505000", 50, "60", "1503000", "0"]
    ]
    mocker.patch('btc_usdt_pipeline.data.fetch_data.get_klines', side_effect=[mock_data, []]) # Return data once, then stop

    start_date = datetime(2023, 3, 15)
    end_date = datetime(2023, 3, 15, 0, 2) # Fetch 2 minutes

    df = fetch_historical_data("BTCUSDT", "1m", start_date, end_date)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'open_time' in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df['open_time'])
    assert 'close' in df.columns
    assert pd.api.types.is_numeric_dtype(df['close'])

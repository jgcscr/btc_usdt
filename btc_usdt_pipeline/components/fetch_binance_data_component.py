from kfp.v2.dsl import component, OutputPath
import requests
import pandas as pd
from datetime import datetime, timedelta

@component(
    base_image="python:3.9-slim",
    packages_to_install=["pandas", "requests", "pyarrow"]
)
def fetch_binance_klines_component(
    symbol: str,
    interval: str,
    start_date_str: str,
    end_date_str: str,
    api_url: str = "https://api.binance.com/api/v3/klines",
    output_data: OutputPath("Dataset") = None
):
    """
    Fetch historical kline (candlestick) data from Binance API and save as Parquet.
    """
    print(f"Fetching data for {symbol} from {start_date_str} to {end_date_str} (interval: {interval})")
    
    # Convert date strings to datetime and then to milliseconds
    start_dt = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date_str, "%Y-%m-%d")
    start_ms = int(start_dt.timestamp() * 1000)
    # Binance endTime is exclusive, so add 1 day to include the last day
    end_ms = int((end_dt + timedelta(days=1)).timestamp() * 1000) - 1
    
    all_klines = []
    limit = 1000
    curr_start = start_ms
    
    while curr_start < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": curr_start,
            "endTime": end_ms,
            "limit": limit
        }
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        klines = response.json()
        if not klines:
            break
        all_klines.extend(klines)
        # Next start time: 1 ms after last returned kline's open time
        last_open_time = klines[-1][0]
        next_start = last_open_time + 1
        if next_start <= curr_start:
            break  # Prevent infinite loop
        curr_start = next_start
        if len(klines) < limit:
            break  # No more data
    
    if not all_klines:
        print("No data fetched for the given parameters.")
        return
    
    # Columns as per Binance API
    columns = [
        "Open_Time", "Open", "High", "Low", "Close", "Volume", "Close_Time",
        "Quote_Asset_Volume", "Number_of_Trades", "Taker_Buy_Base_Asset_Volume",
        "Taker_Buy_Quote_Asset_Volume", "Ignore"
    ]
    df = pd.DataFrame(all_klines, columns=columns)
    # Keep only relevant columns
    df = df[["Open_Time", "Open", "High", "Low", "Close", "Volume"]]
    df["Open_Time"] = pd.to_datetime(df["Open_Time"], unit="ms")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)
    # Filter strictly between start and end date
    mask = (df["Open_Time"] >= pd.to_datetime(start_date_str)) & (df["Open_Time"] < pd.to_datetime(end_date_str) + pd.Timedelta(days=1))
    df = df.loc[mask]
    df.to_parquet(output_data, index=False, engine="pyarrow")
    print(f"Fetched {len(df)} rows. Data saved to {output_data}")

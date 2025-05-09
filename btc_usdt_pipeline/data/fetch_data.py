# data/fetch_data.py
"""Fetches historical kline data from Binance API."""

import os
import time
import pandas as pd
import requests
import json
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from btc_usdt_pipeline import config
from btc_usdt_pipeline.utils.logging_config import setup_logging
from btc_usdt_pipeline.io.data_manager import DataManager
from btc_usdt_pipeline.utils.data_processing import optimize_memory_usage

logger = setup_logging(log_filename='fetch_data.log')

def get_klines(symbol: str, interval: str, start_time: Optional[int] = None, end_time: Optional[int] = None, limit: int = config.FETCH_LIMIT) -> Optional[List[List[Any]]]:
    """Fetches klines from Binance API with error handling."""
    params = {
        'symbol': str(symbol),
        'interval': str(interval),
        'limit': str(limit)
    }
    if start_time:
        params['startTime'] = str(start_time)
    if end_time:
        params['endTime'] = str(end_time)

    try:
        response = requests.get(config.BINANCE_API_URL, params=params, timeout=10) # Added timeout
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode API response: {e}. Response text: {response.text[:200]}") # Log part of the response
        return None

def fetch_historical_data(symbol: str, interval: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
    """Fetches historical data between two dates, handling pagination and rate limits."""
    all_data = []
    start_time = int(start_date.timestamp() * 1000)
    end_time = int(end_date.timestamp() * 1000)
    current_time = start_time

    logger.info(f"Fetching data for {symbol} ({interval}) from {start_date} to {end_date}")

    while current_time < end_time:
        logger.debug(f"Fetching batch starting from {datetime.fromtimestamp(current_time / 1000)}")
        klines = get_klines(symbol, interval, start_time=current_time, limit=config.FETCH_LIMIT)

        if klines is None:
            logger.warning("Failed to fetch a batch, stopping further requests for this run.")
            # Depending on the error, you might want to retry or handle differently
            return None # Or return partial data: pd.DataFrame(all_data, columns=columns) if all_data else None

        if not klines:
            logger.info("No more data returned from API.")
            break

        all_data.extend(klines)
        last_kline_time = klines[-1][0]
        current_time = last_kline_time + 1 # Move to the next millisecond after the last kline

        # Respect potential rate limits
        time.sleep(config.FETCH_DELAY_SECONDS)

    if not all_data:
        logger.warning("No data fetched for the specified period.")
        return None

    columns = ['open_time', 'open', 'high', 'low', 'close', 'volume',
               'close_time', 'quote_asset_volume', 'number_of_trades',
               'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    df = pd.DataFrame(all_data, columns=columns)

    # Data Type Conversion and Cleaning
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
                    'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col])
    df = df.drop(columns=['ignore'])
    df = df.sort_values('open_time').reset_index(drop=True)
    # Remove duplicates just in case API returns overlapping data
    df = df.drop_duplicates(subset=['open_time'], keep='first')

    # Optimize memory usage
    df = optimize_memory_usage(df, logger=logger)

    logger.info(f"Fetched {len(df)} klines.")
    return df

def update_data(existing_df: pd.DataFrame, symbol: str, interval: str) -> Optional[pd.DataFrame]:
    """Fetches new data since the last timestamp in the existing DataFrame."""
    if existing_df.empty or 'open_time' not in existing_df.columns:
        logger.warning("Existing DataFrame is empty or missing 'open_time'. Cannot determine start date for update.")
        return None

    last_timestamp = existing_df['open_time'].max()
    start_date = last_timestamp + pd.Timedelta(milliseconds=1) # Start fetching right after the last record
    end_date = datetime.utcnow() # Fetch up to now

    if start_date >= end_date:
        logger.info("Data is already up-to-date.")
        return existing_df # Return the original df if no new time range exists

    logger.info(f"Updating data from {start_date}...")
    new_data_df = fetch_historical_data(symbol, interval, start_date, end_date)

    if new_data_df is None or new_data_df.empty:
        logger.info("No new data fetched during update.")
        return existing_df # Return original if no new data

    # Combine and remove duplicates
    combined_df = pd.concat([existing_df, new_data_df], ignore_index=True)
    combined_df = combined_df.sort_values('open_time').drop_duplicates(subset=['open_time'], keep='last').reset_index(drop=True)
    logger.info(f"Added {len(new_data_df)} new records. Total records now: {len(combined_df)}")
    return combined_df

def main(years=None):
    """Main function to fetch or update data and save to parquet.
    Args:
        years (int, optional): Number of years of data to fetch. Overrides config.FETCH_DAYS if provided.
    """
    logger.info("Starting data fetch/update process...")
    raw_data_path = config.RAW_DATA_PATH
    symbol = config.SYMBOL
    interval = config.INTERVAL

    # Ensure data directory exists
    try:
        raw_data_path.parent.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured data directory exists: {raw_data_path.parent}")
    except Exception as e:
        logger.error(f"Could not create data directory {raw_data_path.parent}: {e}")
        return # Cannot proceed without data directory

    df = None
    if raw_data_path.exists():
        logger.info(f"Found existing data file: {raw_data_path}. Attempting to load and update.")
        try:
            dm = DataManager()
            existing_df = dm.load_data(raw_data_path, file_type='parquet', use_cache=False)
            # Remove manual dtype casting, handled by memory optimizer
            if 'open_time' not in existing_df.columns:
                 logger.error("Existing data file is missing 'open_time' column. Cannot update. Please check or delete the file.")
                 return
            existing_df['open_time'] = pd.to_datetime(existing_df['open_time']) # Ensure datetime type
            df = update_data(existing_df, symbol, interval)
        except Exception as e:
            logger.error(f"Error loading or updating existing data from {raw_data_path}: {e}. Will attempt full fetch.")
            df = None # Reset df to trigger full fetch
    else:
        logger.info(f"No existing data file found at {raw_data_path}. Performing initial fetch.")

    if df is None: # Trigger full fetch if update failed or no existing file
        end_date = datetime.utcnow()
        if years is not None:
            start_date = end_date - timedelta(days=365*years)
        else:
            start_date = end_date - timedelta(days=config.FETCH_DAYS)
        df = fetch_historical_data(symbol, interval, start_date, end_date)

    if df is not None and not df.empty:
        # Ensure numeric columns are float before saving
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_asset_volume', 'number_of_trades',
                          'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        for col in numeric_columns:
            if col in df.columns:
                # First attempt: standard numeric conversion
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isna().any():
                    logger.warning(f"Column '{col}' has {df[col].isna().sum()} NaN values after type conversion.")
        
        logger.info(f"Dtypes before saving: {df.dtypes}")
        
        # Detect if we're running in Google Colab (check if path contains 'drive')
        is_gdrive = 'drive' in str(raw_data_path).lower()
        
        try:
            # First try: standard Parquet save
            df.to_parquet(raw_data_path, index=False)
            logger.info(f"Successfully saved/updated data to {raw_data_path}")
        except Exception as e:
            logger.error(f"Error saving data to {raw_data_path}: {e}")
            
            try:
                # Second try: For Google Drive issues, try explicit string conversion
                if is_gdrive:
                    logger.info("Attempting to save with explicit string conversion for Google Drive compatibility")
                    for col in numeric_columns:
                        if col in df.columns:
                            # Convert to string to avoid pyarrow type issues with Google Drive
                            df[col] = df[col].astype(str)
                    
                    # Try with a different engine
                    df.to_parquet(raw_data_path, index=False, engine='pyarrow')
                    logger.info(f"Successfully saved data with string conversion to {raw_data_path}")
                    
            except Exception as e2:
                logger.error(f"Final fallback save attempt failed: {e2}")
                
                # Last resort: Save as CSV
                csv_path = str(raw_data_path).replace('.parquet', '.csv')
                try:
                    logger.info(f"Attempting to save as CSV to {csv_path}")
                    df.to_csv(csv_path, index=False)
                    logger.info(f"Successfully saved data as CSV to {csv_path}")
                except Exception as e3:
                    logger.error(f"All save attempts failed. Last error: {e3}")
    elif df is not None and df.empty:
        logger.warning("Fetched data frame is empty, nothing to save.")
    else:
        logger.error("Data fetching failed, no data frame produced.")

    logger.info("Data fetch/update process finished.")

if __name__ == '__main__':
    main()

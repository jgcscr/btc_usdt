"""
fetch_data.py
Entry point script to fetch data using the btc_usdt_pipeline package.
Run as: python -m scripts.fetch_data
"""
# Removed sys.path manipulation

import argparse
from btc_usdt_pipeline.data.fetch_data import main as fetch_data_main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch BTC/USDT data from Binance')
    parser.add_argument('--years', type=int, default=None, help='Number of years of data to fetch (overrides default days)')
    args = parser.parse_args()
    print("--- Running Data Fetch Script ---")
    fetch_data_main(years=args.years)
    print("--- Data Fetch Script Finished ---")
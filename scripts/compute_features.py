"""
compute_features.py
Entry point script to compute features using the btc_usdt_pipeline package.
Run as: python -m scripts.compute_features
"""
import argparse
from btc_usdt_pipeline.utils.data_manager import DataManager
from btc_usdt_pipeline.features.compute_features import main as compute_features_main

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Compute features from raw data')
    parser.add_argument('--input_file', type=str, help='Path to input data file')
    parser.add_argument('--output_file', type=str, help='Path to output features file')
    parser.add_argument('--sample', action='store_true', help='Use a small sample of data for testing')
    args = parser.parse_args()
    
    print("--- Running Feature Computation Script ---")
    compute_features_main(input_file=args.input_file, output_file=args.output_file, sample=args.sample)
    print("--- Feature Computation Script Finished ---")

if __name__ == '__main__':
    main()
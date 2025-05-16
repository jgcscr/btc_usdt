#!/usr/bin/env python
"""
Test script for validating the Docker environment and pipeline components.
Runs a quick test of all pipeline components with a small sample of data.
"""
import os
import sys
import time
import argparse
import logging
from datetime import datetime, timedelta

from btc_usdt_pipeline.utils.logging_config import setup_logging
from btc_usdt_pipeline.workflow.pipeline import Pipeline, TaskRunner
from btc_usdt_pipeline.workflow.trading_pipeline import (
    create_trading_pipeline,
    create_data_fetching_task,
    create_feature_computation_task,
    create_model_training_task,
    create_backtest_task
)

logger = setup_logging('docker_test.log')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test the BTC-USDT pipeline in Docker environment')
    
    parser.add_argument(
        '--components', 
        type=str,
        nargs='+',
        choices=['fetch', 'features', 'train', 'backtest', 'all'],
        default=['all'],
        help='Pipeline components to test (default: all)'
    )
    
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading symbol (default: BTCUSDT)')
    parser.add_argument('--interval', type=str, default='1m', help='Candlestick interval (default: 1m)')
    parser.add_argument('--limit', type=int, default=1000, help='Number of data points to fetch (default: 1000)')
    parser.add_argument('--output-dir', type=str, default='data/test', help='Directory for test outputs')
    parser.add_argument('--gpu', action='store_true', help='Test GPU availability')
    
    return parser.parse_args()

def test_gpu_availability():
    """Test if GPU is available in the container."""
    logger.info("Testing GPU availability...")
    
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            logger.info(f"GPU is available. Found {len(gpus)} GPU devices:")
            for gpu in gpus:
                logger.info(f"  - {gpu}")
            return True
        else:
            logger.warning("No GPU devices found by TensorFlow.")
            return False
    except Exception as e:
        logger.error(f"Error checking GPU availability: {e}")
        return False

def run_component_test(component, context):
    """Run a test for a specific pipeline component."""
    component_functions = {
        'fetch': create_data_fetching_task,
        'features': create_feature_computation_task,
        'train': create_model_training_task,
        'backtest': create_backtest_task
    }
    
    if component not in component_functions:
        logger.error(f"Unknown component: {component}")
        return False
    
    logger.info(f"Testing component: {component}")
    start_time = time.time()
    
    try:
        result = component_functions[component](context)
        elapsed = time.time() - start_time
        logger.info(f"Component {component} test completed in {elapsed:.2f} seconds")
        return True
    except Exception as e:
        logger.error(f"Error testing component {component}: {e}", exc_info=True)
        return False

def run_full_pipeline_test(context):
    """Run a test of the complete pipeline."""
    logger.info("Testing full pipeline...")
    
    try:
        pipeline = create_trading_pipeline()
        start_time = time.time()
        result = pipeline.run(initial_context=context)
        elapsed = time.time() - start_time
        
        logger.info(f"Full pipeline test completed in {elapsed:.2f} seconds with status: {pipeline.status}")
        
        # Save pipeline execution history
        os.makedirs('logs/pipeline_history', exist_ok=True)
        history_file = f"logs/pipeline_history/docker_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        pipeline.save_history(history_file)
        
        return pipeline.status == 'success'
    except Exception as e:
        logger.error(f"Error testing full pipeline: {e}", exc_info=True)
        return False

def main():
    """Run the Docker environment test."""
    args = parse_args()
    
    logger.info("=== Starting Docker Environment Test ===")
    logger.info(f"Test components: {args.components}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Test GPU if requested
    if args.gpu:
        gpu_available = test_gpu_availability()
        logger.info(f"GPU availability test {'passed' if gpu_available else 'failed'}")
    
    # Prepare context for tests
    # Use a short timeframe for quick testing
    end_date = datetime.now()
    start_date = end_date - timedelta(days=1)
    
    context = {
        'symbol': args.symbol,
        'interval': args.interval,
        'sample': True,  # Always use sample mode for tests
        'limit': args.limit,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'raw_data_path': f"{args.output_dir}/{args.symbol}_{args.interval}_raw_test.parquet",
        'enriched_data_path': f"{args.output_dir}/{args.symbol}_{args.interval}_enriched_test.parquet",
        'models_dir': f"{args.output_dir}/models",
        'results_dir': f"{args.output_dir}/results"
    }
    
    logger.info(f"Test context: {context}")
    
    # Create component directories
    os.makedirs(context['models_dir'], exist_ok=True)
    os.makedirs(context['results_dir'], exist_ok=True)
    
    # Track test results
    results = {}
    
    # Run tests
    if 'all' in args.components:
        # Test full pipeline
        results['full_pipeline'] = run_full_pipeline_test(context)
    else:
        # Test individual components
        for component in args.components:
            results[component] = run_component_test(component, context)
    
    # Report results
    logger.info("=== Docker Environment Test Results ===")
    all_passed = True
    
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        logger.info(f"{name}: {status}")
        all_passed = all_passed and passed
    
    logger.info(f"Overall test result: {'PASSED' if all_passed else 'FAILED'}")
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
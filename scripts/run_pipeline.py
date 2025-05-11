#!/usr/bin/env python
"""
Command-line script to run the BTC-USDT trading pipeline.
This script provides a convenient way to run the entire pipeline or individual components.
"""
import argparse
import logging
import sys
import time
from datetime import datetime

from btc_usdt_pipeline.workflow.pipeline import Pipeline
from btc_usdt_pipeline.workflow.trading_pipeline import (
    create_trading_pipeline, 
    create_data_fetching_task,
    create_feature_computation_task,
    create_model_training_task,
    create_backtest_task
)
from btc_usdt_pipeline.utils.logging_config import setup_logging

logger = setup_logging('run_pipeline.log')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the BTC-USDT trading pipeline')
    
    parser.add_argument(
        '--component', 
        type=str, 
        choices=['all', 'fetch', 'features', 'train', 'backtest'],
        default='all',
        help='Pipeline component to run (default: all)'
    )
    
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading symbol (default: BTCUSDT)')
    parser.add_argument('--interval', type=str, default='1m', help='Candlestick interval (default: 1m)')
    parser.add_argument('--sample', action='store_true', help='Use sample mode for faster processing')
    parser.add_argument('--raw-data', type=str, help='Path to raw data file')
    parser.add_argument('--enriched-data', type=str, help='Path to enriched data file')
    parser.add_argument('--models-dir', type=str, default='models', help='Directory for saving models')
    parser.add_argument('--results-dir', type=str, default='results', help='Directory for saving results')
    
    # Date range options for data fetching
    parser.add_argument('--start-date', type=str, help='Start date for data fetching (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date for data fetching (YYYY-MM-DD)')
    parser.add_argument('--limit', type=int, help='Limit number of candlesticks to fetch')
    
    return parser.parse_args()

def run_pipeline_component(component, context):
    """Run a specific component of the pipeline."""
    component_functions = {
        'fetch': create_data_fetching_task,
        'features': create_feature_computation_task,
        'train': create_model_training_task,
        'backtest': create_backtest_task
    }
    
    if component not in component_functions:
        logger.error(f"Unknown component: {component}")
        return None
    
    logger.info(f"Running pipeline component: {component}")
    start_time = time.time()
    
    try:
        result = component_functions[component](context)
        elapsed = time.time() - start_time
        logger.info(f"Component {component} completed in {elapsed:.2f} seconds")
        return result
    except Exception as e:
        logger.error(f"Error running component {component}: {e}", exc_info=True)
        return None

def main():
    """Run the trading pipeline based on command line arguments."""
    args = parse_args()
    
    # Build context from command line arguments
    context = {
        'symbol': args.symbol,
        'interval': args.interval,
        'sample': args.sample
    }
    
    if args.raw_data:
        context['raw_data_path'] = args.raw_data
    
    if args.enriched_data:
        context['enriched_data_path'] = args.enriched_data
    
    if args.models_dir:
        context['models_dir'] = args.models_dir
    
    if args.results_dir:
        context['results_dir'] = args.results_dir
    
    if args.start_date:
        context['start_date'] = args.start_date
    
    if args.end_date:
        context['end_date'] = args.end_date
    
    if args.limit:
        context['limit'] = args.limit
    
    # Log execution context
    logger.info(f"Running with context: {context}")
    
    if args.component == 'all':
        # Run the full pipeline
        pipeline = create_trading_pipeline()
        logger.info(f"Running full trading pipeline: {pipeline.name}")
        
        try:
            result = pipeline.run(initial_context=context)
            logger.info(f"Pipeline completed with status: {pipeline.status}")
            
            # Save pipeline execution history
            history_file = f"logs/pipeline_history/full_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            pipeline.save_history(history_file)
            logger.info(f"Pipeline execution history saved to {history_file}")
            
            return 0 if pipeline.status == 'success' else 1
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            return 1
    else:
        # Run individual component
        result = run_pipeline_component(args.component, context)
        return 0 if result is not None else 1

if __name__ == "__main__":
    sys.exit(main())
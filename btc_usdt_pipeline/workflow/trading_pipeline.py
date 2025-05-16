"""
Trading pipeline orchestrator for the BTC-USDT trading system.
Combines data fetching, feature computation, model training, and backtesting
into a cohesive workflow.
"""
import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from btc_usdt_pipeline.workflow.pipeline import Pipeline, TaskRunner
from btc_usdt_pipeline.workflow.scheduler import WorkflowScheduler
from btc_usdt_pipeline.utils.logging_config import setup_logging

logger = setup_logging('trading_pipeline.log')

def create_data_fetching_task(context: Dict[str, Any]):
    """Task function to fetch market data."""
    from btc_usdt_pipeline.data.fetch_data import main as fetch_data_main
    
    logger.info("Starting data fetching task...")
    
    # Extract parameters from context or use defaults
    symbol = context.get('symbol', 'BTCUSDT')
    interval = context.get('interval', '1m')
    output_file = context.get('raw_data_path', 'data/1m_btcusdt_raw.parquet')
    start_date = context.get('start_date', None)
    end_date = context.get('end_date', None)
    limit = context.get('limit', None)
    
    # Run the data fetching process
    fetch_data_main(
        symbol=symbol,
        interval=interval,
        output_file=output_file,
        start_date=start_date,
        end_date=end_date,
        limit=limit
    )
    
    logger.info(f"Data fetching completed successfully. Data saved to {output_file}")
    return {
        'raw_data_path': output_file,
        'symbol': symbol,
        'interval': interval,
        'fetch_timestamp': datetime.now().isoformat()
    }

def create_feature_computation_task(context: Dict[str, Any]):
    """Task function to compute features from raw data."""
    from btc_usdt_pipeline.features.compute_features import main as compute_features_main
    
    logger.info("Starting feature computation task...")
    
    # Extract parameters from context or use defaults
    input_file = context.get('raw_data_path', 'data/1m_btcusdt_raw.parquet')
    output_file = context.get('enriched_data_path', 'data/1m_btcusdt_enriched.parquet')
    sample = context.get('sample', False)
    
    # Run feature computation
    compute_features_main(
        input_file=input_file,
        output_file=output_file,
        sample=sample
    )
    
    logger.info(f"Feature computation completed successfully. Features saved to {output_file}")
    return {
        'enriched_data_path': output_file,
        'feature_computation_timestamp': datetime.now().isoformat()
    }

def create_model_training_task(context: Dict[str, Any]):
    """Task function to train prediction models."""
    from btc_usdt_pipeline.models.train import train_models
    
    logger.info("Starting model training task...")
    
    # Extract parameters from context or use defaults
    input_file = context.get('enriched_data_path', 'data/1m_btcusdt_enriched.parquet')
    models_dir = context.get('models_dir', 'models')
    
    # Ensure models directory exists
    os.makedirs(models_dir, exist_ok=True)
    
    # Run model training
    trained_models = train_models(
        data_path=input_file,
        models_dir=models_dir
    )
    
    logger.info(f"Model training completed successfully. {len(trained_models)} models trained.")
    return {
        'trained_models': trained_models,
        'models_dir': models_dir,
        'training_timestamp': datetime.now().isoformat()
    }

def create_backtest_task(context: Dict[str, Any]):
    """Task function to backtest trading strategies."""
    from btc_usdt_pipeline.trading.backtest import run_backtest
    
    logger.info("Starting backtest task...")
    
    # Extract parameters from context or use defaults
    input_file = context.get('enriched_data_path', 'data/1m_btcusdt_enriched.parquet')
    models_dir = context.get('models_dir', 'models')
    results_dir = context.get('results_dir', 'results/backtest')
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Run backtest
    backtest_results = run_backtest(
        data_path=input_file,
        models_dir=models_dir,
        results_dir=results_dir
    )
    
    logger.info(f"Backtest completed successfully. Results saved to {results_dir}")
    return {
        'backtest_results': backtest_results,
        'results_dir': results_dir,
        'backtest_timestamp': datetime.now().isoformat()
    }

def create_trading_pipeline() -> Pipeline:
    """Create the complete trading pipeline with all tasks."""
    # Create the pipeline
    pipeline = Pipeline(name="btc_usdt_trading_pipeline")
    
    # Add data fetching task
    data_task = TaskRunner(
        name="fetch_data",
        func=create_data_fetching_task,
        max_retries=3,
        retry_delay=60
    )
    pipeline.add_task(data_task)
    
    # Add feature computation task
    features_task = TaskRunner(
        name="compute_features",
        func=create_feature_computation_task,
        max_retries=3,
        retry_delay=60,
        dependencies=["fetch_data"]
    )
    pipeline.add_task(features_task)
    
    # Add model training task
    training_task = TaskRunner(
        name="train_models",
        func=create_model_training_task,
        max_retries=2,
        retry_delay=120,
        dependencies=["compute_features"]
    )
    pipeline.add_task(training_task)
    
    # Add backtest task
    backtest_task = TaskRunner(
        name="run_backtest",
        func=create_backtest_task,
        max_retries=2,
        retry_delay=120,
        dependencies=["train_models"]
    )
    pipeline.add_task(backtest_task)
    
    return pipeline

def schedule_trading_pipeline(scheduler: WorkflowScheduler, schedule_interval: int = 14400):
    """
    Schedule the trading pipeline to run at regular intervals.
    Default is every 4 hours (14400 seconds).
    """
    # Create the pipeline
    pipeline = create_trading_pipeline()
    
    # Schedule it to run at the specified interval
    schedule_id = scheduler.schedule(
        pipeline=pipeline,
        schedule_type='interval',
        interval=schedule_interval,
        initial_context={
            'symbol': 'BTCUSDT',
            'interval': '1m',
            'sample': False
        }
    )
    
    logger.info(f"Trading pipeline scheduled with ID: {schedule_id}, interval: {schedule_interval} seconds")
    return schedule_id

def main():
    """Run the trading pipeline orchestrator."""
    logger.info("Starting trading pipeline orchestrator...")
    
    # Create scheduler
    scheduler = WorkflowScheduler(check_interval=30)
    
    # Schedule the trading pipeline (every 4 hours)
    schedule_trading_pipeline(scheduler, schedule_interval=14400)
    
    # Start the scheduler
    scheduler.start()
    
    logger.info("Trading pipeline orchestrator running. Press Ctrl+C to stop.")
    
    # Keep the main thread running
    try:
        while True:
            import time
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Stopping trading pipeline orchestrator...")
        scheduler.stop()

if __name__ == "__main__":
    main()

"""Workflow utilities for the BTC/USDT pipeline."""
import time
import traceback
import logging
from typing import Callable, List, Dict, Any, Optional
from datetime import datetime
import threading
import sched
from btc_usdt_pipeline.utils.logging_config import setup_logging

logger = setup_logging(log_filename='workflow.log')

class TaskError(Exception):
    """Custom exception for workflow task errors."""

class TaskRunner:
    """Class for running workflow tasks with retry logic."""
    def __init__(self, func: Callable, name: str, max_retries: int = 3, retry_delay: int = 10, dependencies: Optional[List[str]] = None):
        self.func = func
        self.name = name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.dependencies = dependencies or []
        self.status: str = 'pending'
        self.last_error = None
        self.result = None

    def run(self, context: Dict[str, Any]):
        """Run the task with retry logic."""
        retries = 0
        while retries <= self.max_retries:
            try:
                logger.info("Running task: %s (attempt %d)", self.name, retries+1)
                self.status = 'running'
                self.result = self.func(context)
                self.status = 'success'
                logger.info("Task %s completed successfully.", self.name)
                return self.result
            except Exception as e:
                self.last_error = str(e)
                logger.error("Task %s failed: %s\n%s", self.name, e, traceback.format_exc())
                self.status = 'failed'
                retries += 1
                if retries > self.max_retries:
                    raise TaskError(f"Task {self.name} failed after {self.max_retries} retries.")
                time.sleep(self.retry_delay)

class Pipeline:
    def __init__(self, name: str):
        self.name: str = name
        self.status: str = ''  # Initialize as empty string to avoid Optional[str] assignment issues
        self.tasks: Dict[str, 'TaskRunner'] = {}
        self.task_order: List[str] = []
        self.context: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []

    def add_task(self, task: TaskRunner):
        self.tasks[task.name] = task
        self.task_order.append(task.name)

    def run(self):
        logger.info(f"Starting pipeline: {self.name}")
        self.status = 'running'
        for task_name in self.task_order:
            task = self.tasks[task_name]
            # Check dependencies
            for dep in task.dependencies:
                if self.tasks[dep].status != 'success':
                    logger.warning(f"Dependency {dep} for task {task_name} not satisfied. Skipping task.")
                    task.status = 'skipped'
                    break
            if task.status == 'skipped':
                continue
            try:
                result = task.run(self.context)
                self.context[task.name] = result
            except TaskError as e:
                logger.error(f"Pipeline {self.name} failed at task {task_name}: {e}")
                self.status = 'failed'
                self.history.append({'timestamp': datetime.utcnow().isoformat(), 'status': self.status, 'failed_task': task_name})
                return
        self.status = 'success'
        self.history.append({'timestamp': datetime.utcnow().isoformat(), 'status': self.status})
        logger.info(f"Pipeline {self.name} completed successfully.")

# --- Example Task Implementations ---
def DataIngestionTask(context):
    from btc_usdt_pipeline.data.fetch_data import fetch_historical_data
    from btc_usdt_pipeline.utils.data_quality import detect_missing_data
    logger.info("Fetching new data...")
    df = fetch_historical_data()
    quality = detect_missing_data(df)
    if quality['total_missing'] > 0:
        logger.warning("Data quality issue detected during ingestion.")
    return df

def FeatureEngineeringTask(context):
    from btc_usdt_pipeline.features.feature_pipeline import FeaturePipeline, RSICalculator, MACDCalculator, BollingerBandsCalculator
    logger.info("Applying feature engineering...")
    df = context['DataIngestionTask']
    pipeline = FeaturePipeline()
    pipeline.add_transformer(RSICalculator())
    pipeline.add_transformer(MACDCalculator())
    pipeline.add_transformer(BollingerBandsCalculator())
    df_feat = pipeline.fit_transform(df)
    context['feature_pipeline'] = pipeline
    return df_feat

def ModelTrainingTask(context):
    from btc_usdt_pipeline.models.train import train_random_forest
    logger.info("Training model...")
    df = context['FeatureEngineeringTask']
    model, metrics = train_random_forest(df)
    context['model'] = model
    return metrics

def BacktestTask(context):
    from btc_usdt_pipeline.trading.backtest import run_backtest
    logger.info("Running backtest...")
    df = context['FeatureEngineeringTask']
    signals = context.get('signals') or ['Flat'] * len(df)
    equity_curve, trade_log = run_backtest(df, signals)
    context['equity_curve'] = equity_curve
    context['trade_log'] = trade_log
    return {'equity_curve': equity_curve, 'trade_log': trade_log}

def ReportingTask(context):
    from btc_usdt_pipeline.utils.performance_reporting import calculate_performance_metrics, to_html_report
    logger.info("Generating performance report...")
    equity_curve = context['equity_curve']
    trade_log = context['trade_log']
    metrics = calculate_performance_metrics(equity_curve, trade_log)
    results = {'metrics': metrics}
    to_html_report(results, title="Performance Report", output_path="./results/performance_report.html")
    return results

# --- Scheduling ---
class WorkflowScheduler:
    def __init__(self):
        self.sched = sched.scheduler(time.time, time.sleep)
        self.scheduled: List[Dict[str, Any]] = []
        self.running: Dict[str, threading.Thread] = {}

    def schedule(self, pipeline: Pipeline, cron: str):
        # For demo: cron is a delay in seconds
        delay = int(cron)
        def run_pipeline():
            pipeline.run()
        event = self.sched.enter(delay, 1, run_pipeline)
        self.scheduled.append({'pipeline': pipeline, 'event': event, 'cron': cron})
        threading.Thread(target=self.sched.run, daemon=True).start()

# --- Workflow CLI ---
WORKFLOWS: Dict[str, Pipeline] = {}

def list_workflows() -> None:
    logger.info("Available workflows: %s", list(WORKFLOWS.keys()))

def run_workflow(name: str) -> None:
    if name not in WORKFLOWS:
        logger.error(f"Workflow {name} not found.")
        return
    WORKFLOWS[name].run()

def workflow_status(name: str) -> None:
    if name not in WORKFLOWS:
        logger.error(f"Workflow {name} not found.")
        return
    logger.info(f"Workflow {name} status: {WORKFLOWS[name].status}")

def schedule_workflow(name: str, cron: str) -> None:
    if name not in WORKFLOWS:
        logger.error(f"Workflow {name} not found.")
        return
    scheduler = WorkflowScheduler()
    scheduler.schedule(WORKFLOWS[name], cron)
    logger.info(f"Scheduled workflow {name} with cron '{cron}'")

# --- Register Example Workflow ---
def register_default_workflow():
    pipeline = Pipeline("default_workflow")
    pipeline.add_task(TaskRunner(DataIngestionTask, "DataIngestionTask"))
    pipeline.add_task(TaskRunner(FeatureEngineeringTask, "FeatureEngineeringTask", dependencies=["DataIngestionTask"]))
    pipeline.add_task(TaskRunner(ModelTrainingTask, "ModelTrainingTask", dependencies=["FeatureEngineeringTask"]))
    pipeline.add_task(TaskRunner(BacktestTask, "BacktestTask", dependencies=["FeatureEngineeringTask"]))
    pipeline.add_task(TaskRunner(ReportingTask, "ReportingTask", dependencies=["BacktestTask"]))
    WORKFLOWS[pipeline.name] = pipeline

register_default_workflow()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Workflow Orchestration CLI")
    parser.add_argument('command', choices=['list_workflows', 'run_workflow', 'workflow_status', 'schedule_workflow'])
    parser.add_argument('--name', type=str, help='Workflow name')
    parser.add_argument('--cron', type=str, help='Cron schedule (seconds for demo)')
    args = parser.parse_args()
    if args.command == 'list_workflows':
        list_workflows()
    elif args.command == 'run_workflow' and args.name:
        run_workflow(args.name)
    elif args.command == 'workflow_status' and args.name:
        workflow_status(args.name)
    elif args.command == 'schedule_workflow' and args.name and args.cron:
        schedule_workflow(args.name, args.cron)
    else:
        logger.error("Invalid command or missing arguments.")

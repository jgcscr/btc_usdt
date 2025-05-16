"""
Pipeline Orchestrator for running the trading system workflow.
Handles scheduling, dependency management, and execution tracking.
"""
import logging
import time
from datetime import datetime, timedelta
import os
import json
from typing import Dict, List, Any, Callable, Optional, Union
import traceback

from btc_usdt_pipeline.utils.logging_config import setup_logging
from btc_usdt_pipeline.utils.config_manager import config_manager

logger = setup_logging("pipeline.log")

class TaskError(Exception):
    """Exception raised when a task fails."""
    pass

class TaskRunner:
    """Executes a single task in the pipeline with retry logic and dependency tracking."""
    def __init__(
        self, 
        name: str, 
        func: Callable[[Dict[str, Any]], Any], 
        max_retries: int = 3, 
        retry_delay: int = 60,
        dependencies: Optional[List[str]] = None
    ):
        self.name = name
        self.func = func
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.dependencies = dependencies or []
        self.status: str = 'pending'
        self.last_error = None
        self.result = None
        self.start_time = None
        self.end_time = None

    def run(self, context: Dict[str, Any]):
        """Execute the task function with retry logic."""
        retries = 0
        self.start_time = datetime.utcnow()
        
        while retries <= self.max_retries:
            try:
                logger.info(f"Running task: {self.name} (attempt {retries+1})")
                self.status = 'running'
                self.result = self.func(context)
                self.status = 'success'
                logger.info(f"Task {self.name} completed successfully.")
                self.end_time = datetime.utcnow()
                return self.result
            except Exception as e:
                self.last_error = str(e)
                logger.error(f"Task {self.name} failed: {e}\n{traceback.format_exc()}")
                self.status = 'failed'
                retries += 1
                if retries > self.max_retries:
                    self.end_time = datetime.utcnow()
                    raise TaskError(f"Task {self.name} failed after {self.max_retries} retries.")
                time.sleep(self.retry_delay)
                
    def get_runtime(self) -> Optional[float]:
        """Get task runtime in seconds, or None if not completed."""
        if not self.start_time:
            return None
        end = self.end_time or datetime.utcnow()
        return (end - self.start_time).total_seconds()

class Pipeline:
    """Orchestrates the execution of a series of tasks in a workflow."""
    def __init__(self, name: str):
        self.name: str = name
        self.status: str = 'pending'
        self.tasks: Dict[str, TaskRunner] = {}
        self.task_order: List[str] = []
        self.context: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
        self.start_time = None
        self.end_time = None

    def add_task(self, task: TaskRunner):
        """Add a task to the pipeline."""
        self.tasks[task.name] = task
        if task.name not in self.task_order:
            self.task_order.append(task.name)
        return self

    def run(self, initial_context: Optional[Dict[str, Any]] = None):
        """Run all tasks in the pipeline in the defined order."""
        logger.info(f"Starting pipeline: {self.name}")
        self.start_time = datetime.utcnow()
        self.status = 'running'
        
        # Initialize or update context
        if initial_context:
            self.context.update(initial_context)
            
        for task_name in self.task_order:
            task = self.tasks[task_name]
            
            # Check dependencies
            deps_satisfied = True
            for dep in task.dependencies:
                if dep not in self.tasks:
                    logger.error(f"Dependency '{dep}' for task '{task_name}' not found in pipeline.")
                    task.status = 'skipped'
                    deps_satisfied = False
                    break
                if self.tasks[dep].status != 'success':
                    logger.warning(f"Dependency '{dep}' for task '{task_name}' not satisfied (status: {self.tasks[dep].status}). Skipping task.")
                    task.status = 'skipped'
                    deps_satisfied = False
                    break
                    
            if not deps_satisfied:
                continue
                
            try:
                logger.info(f"Executing task: {task_name}")
                result = task.run(self.context)
                self.context[task_name] = result
            except TaskError as e:
                logger.error(f"Pipeline {self.name} failed at task {task_name}: {e}")
                self.status = 'failed'
                self.end_time = datetime.utcnow()
                self.history.append({
                    'timestamp': self.end_time.isoformat(), 
                    'status': self.status, 
                    'failed_task': task_name,
                    'error': str(e),
                    'runtime': self.get_runtime()
                })
                return self.context
                
        self.status = 'success'
        self.end_time = datetime.utcnow()
        self.history.append({
            'timestamp': self.end_time.isoformat(), 
            'status': self.status,
            'runtime': self.get_runtime()
        })
        logger.info(f"Pipeline {self.name} completed successfully in {self.get_runtime():.2f} seconds.")
        return self.context
        
    def get_runtime(self) -> Optional[float]:
        """Get pipeline runtime in seconds, or None if not started."""
        if not self.start_time:
            return None
        end = self.end_time or datetime.utcnow()
        return (end - self.start_time).total_seconds()
        
    def get_task_status(self) -> Dict[str, str]:
        """Get the status of all tasks in the pipeline."""
        return {name: task.status for name, task in self.tasks.items()}
        
    def save_history(self, filepath: str):
        """Save pipeline execution history to a JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
            
    def reset(self):
        """Reset the pipeline to prepare for another run."""
        self.status = 'pending'
        for task in self.tasks.values():
            task.status = 'pending'
            task.last_error = None
            task.result = None
            task.start_time = None
            task.end_time = None
        self.start_time = None
        self.end_time = None
        return self

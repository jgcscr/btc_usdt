import time
import traceback
from typing import Callable, List, Dict, Any, Optional
from datetime import datetime
from btc_usdt_pipeline.utils.logging_config import setup_logging

logger = setup_logging(log_filename='workflow.log')

class TaskError(Exception):
    pass

class TaskRunner:
    def __init__(self, func: Callable, name: str, max_retries: int = 3, retry_delay: int = 10, dependencies: Optional[List[str]] = None):
        self.func = func
        self.name = name
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.dependencies = dependencies or []
        self.status: str = 'pending'  # Change from None to str type annotation
        self.last_error = None
        self.result = None

    def run(self, context: Dict[str, Any]):
        retries = 0
        while retries <= self.max_retries:
            try:
                logger.info(f"Running task: {self.name} (attempt {retries+1})")
                self.status = 'running'
                self.result = self.func(context)
                self.status = 'success'
                logger.info(f"Task {self.name} completed successfully.")
                return self.result
            except Exception as e:
                self.last_error = str(e)
                logger.error(f"Task {self.name} failed: {e}\n{traceback.format_exc()}")
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

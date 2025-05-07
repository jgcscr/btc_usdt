import threading
import sched
import time
from typing import List, Dict, Any
from btc_usdt_pipeline.workflow.pipeline import Pipeline
from btc_usdt_pipeline.utils.logging_config import setup_logging

logger = setup_logging(log_filename='workflow.log')

class WorkflowScheduler:
    def __init__(self) -> None:
        self.sched: sched.scheduler = sched.scheduler(time.time, time.sleep)
        self.scheduled: List[Dict[str, Any]] = []
        self.running: Dict[str, threading.Thread] = {}

    def schedule(self, pipeline: Pipeline, cron: str) -> None:
        delay: int = int(cron)
        def run_pipeline() -> None:
            pipeline.run()
        event = self.sched.enter(delay, 1, run_pipeline)
        self.scheduled.append({'pipeline': pipeline, 'event': event, 'cron': cron})
        threading.Thread(target=self.sched.run, daemon=True).start()

# CLI and workflow registry logic can be added here or in a separate cli.py if needed

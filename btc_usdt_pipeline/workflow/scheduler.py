"""
Scheduler for automated execution of trading system pipelines.
Supports interval and cron-style scheduling.
"""
import threading
import time
import logging
from datetime import datetime, timedelta
import signal
import sys
from typing import List, Dict, Any, Optional, Union, Callable
import json
import os
from croniter import croniter

from btc_usdt_pipeline.workflow.pipeline import Pipeline, TaskRunner
from btc_usdt_pipeline.utils.logging_config import setup_logging

logger = setup_logging(log_filename='scheduler.log')

class ScheduleConfig:
    """Configuration for a scheduled pipeline execution."""
    def __init__(
        self,
        pipeline: Pipeline,
        schedule_type: str,  # 'interval', 'cron', or 'once'
        interval: Optional[int] = None,  # seconds
        cron_expression: Optional[str] = None,
        next_run: Optional[datetime] = None,
        max_runs: Optional[int] = None,
        initial_context: Optional[Dict[str, Any]] = None,
        enabled: bool = True
    ):
        self.pipeline = pipeline
        self.schedule_type = schedule_type
        self.interval = interval
        self.cron_expression = cron_expression
        self.next_run = next_run or datetime.now()
        self.last_run: Optional[datetime] = None
        self.run_count = 0
        self.max_runs = max_runs
        self.enabled = enabled
        self.initial_context = initial_context or {}
        self.id = f"{pipeline.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Validate configuration
        self._validate()
        
        # Calculate initial next_run if not specified
        if not next_run:
            self._calculate_next_run()
    
    def _validate(self):
        """Validate schedule configuration."""
        if self.schedule_type not in ['interval', 'cron', 'once']:
            raise ValueError(f"Invalid schedule_type: {self.schedule_type}")
            
        if self.schedule_type == 'interval' and not self.interval:
            raise ValueError("Interval scheduling requires an interval value in seconds")
            
        if self.schedule_type == 'cron' and not self.cron_expression:
            raise ValueError("Cron scheduling requires a cron expression")
            
        if self.schedule_type == 'cron' and not croniter.is_valid(self.cron_expression):
            raise ValueError(f"Invalid cron expression: {self.cron_expression}")
    
    def _calculate_next_run(self):
        """Calculate the next run time based on schedule configuration."""
        now = datetime.now()
        
        if self.schedule_type == 'interval':
            if self.last_run:
                self.next_run = self.last_run + timedelta(seconds=self.interval)
            else:
                self.next_run = now
                
        elif self.schedule_type == 'cron':
            if self.cron_expression:
                cron = croniter(self.cron_expression, now)
                self.next_run = cron.get_next(datetime)
                
        elif self.schedule_type == 'once':
            # For 'once' scheduling, if it has already run, don't schedule again
            if self.run_count > 0:
                self.next_run = None
        
    def should_run(self) -> bool:
        """Determine if the pipeline should run now."""
        if not self.enabled:
            return False
            
        if self.max_runs is not None and self.run_count >= self.max_runs:
            return False
            
        if not self.next_run:
            return False
            
        return datetime.now() >= self.next_run
    
    def update_after_run(self):
        """Update schedule state after a pipeline run."""
        self.last_run = datetime.now()
        self.run_count += 1
        self._calculate_next_run()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert schedule configuration to a dictionary for serialization."""
        return {
            'id': self.id,
            'pipeline_name': self.pipeline.name,
            'schedule_type': self.schedule_type,
            'interval': self.interval,
            'cron_expression': self.cron_expression,
            'next_run': self.next_run.isoformat() if self.next_run else None,
            'last_run': self.last_run.isoformat() if self.last_run else None,
            'run_count': self.run_count,
            'max_runs': self.max_runs,
            'enabled': self.enabled
        }

class WorkflowScheduler:
    """Scheduler for automated execution of trading system pipelines."""
    def __init__(self, check_interval: int = 10):
        self.schedules: Dict[str, ScheduleConfig] = {}
        self.running: Dict[str, threading.Thread] = {}
        self.check_interval = check_interval  # seconds
        self.scheduler_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.state_file = os.path.join('logs', 'scheduler_state.json')
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}. Shutting down scheduler...")
        self.stop()
        sys.exit(0)
    
    def schedule(
        self,
        pipeline: Pipeline,
        schedule_type: str = 'interval',
        interval: Optional[int] = None,
        cron_expression: Optional[str] = None,
        next_run: Optional[datetime] = None,
        max_runs: Optional[int] = None,
        initial_context: Optional[Dict[str, Any]] = None,
        enabled: bool = True
    ) -> str:
        """
        Schedule a pipeline for execution.
        
        Args:
            pipeline: The pipeline to schedule
            schedule_type: 'interval', 'cron', or 'once'
            interval: Seconds between runs for 'interval' type
            cron_expression: Cron expression for 'cron' type
            next_run: Specific time for next run
            max_runs: Maximum number of runs (None for unlimited)
            initial_context: Initial context for pipeline execution
            enabled: Whether the schedule is active
            
        Returns:
            The ID of the schedule
        """
        config = ScheduleConfig(
            pipeline=pipeline,
            schedule_type=schedule_type,
            interval=interval,
            cron_expression=cron_expression,
            next_run=next_run,
            max_runs=max_runs,
            initial_context=initial_context,
            enabled=enabled
        )
        
        self.schedules[config.id] = config
        logger.info(f"Scheduled pipeline '{pipeline.name}' ({config.id})")
        
        if not self.scheduler_thread or not self.scheduler_thread.is_alive():
            self.start()
            
        self._save_state()
        return config.id
    
    def _check_schedules(self):
        """Check schedules and run pipelines that are due."""
        while not self.stop_event.is_set():
            now = datetime.now()
            
            try:
                for schedule_id, config in list(self.schedules.items()):
                    if config.should_run():
                        # Check if already running
                        if schedule_id in self.running and self.running[schedule_id].is_alive():
                            logger.warning(f"Pipeline '{config.pipeline.name}' is already running. Skipping this run.")
                            continue
                            
                        logger.info(f"Starting pipeline '{config.pipeline.name}' ({schedule_id})")
                        
                        # Create thread for pipeline execution
                        thread = threading.Thread(
                            target=self._run_pipeline,
                            args=(config,),
                            name=f"pipeline-{config.pipeline.name}"
                        )
                        
                        self.running[schedule_id] = thread
                        thread.start()
                        
                        # Update schedule
                        config.update_after_run()
                        self._save_state()
                        
                # Clean up completed threads
                for schedule_id in list(self.running.keys()):
                    if not self.running[schedule_id].is_alive():
                        del self.running[schedule_id]
                        
            except Exception as e:
                logger.error(f"Error in scheduler: {e}", exc_info=True)
                
            # Sleep until next check
            self.stop_event.wait(self.check_interval)
    
    def _run_pipeline(self, config: ScheduleConfig):
        """Run a pipeline in a separate thread."""
        try:
            # Reset the pipeline for a fresh run
            config.pipeline.reset()
            
            # Run the pipeline with initial context
            result = config.pipeline.run(initial_context=config.initial_context)
            
            # Log result summary
            logger.info(f"Pipeline '{config.pipeline.name}' completed with status: {config.pipeline.status}")
            
            # Save execution history
            history_dir = os.path.join('logs', 'pipeline_history')
            os.makedirs(history_dir, exist_ok=True)
            history_file = os.path.join(
                history_dir, 
                f"{config.pipeline.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            config.pipeline.save_history(history_file)
            
        except Exception as e:
            logger.error(f"Error running pipeline '{config.pipeline.name}': {e}", exc_info=True)
    
    def start(self):
        """Start the scheduler."""
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            logger.warning("Scheduler is already running.")
            return
            
        self.stop_event.clear()
        self.scheduler_thread = threading.Thread(
            target=self._check_schedules,
            name="scheduler-main"
        )
        self.scheduler_thread.daemon = True
        self.scheduler_thread.start()
        
        logger.info("Workflow scheduler started.")
        
    def stop(self):
        """Stop the scheduler."""
        logger.info("Stopping workflow scheduler...")
        self.stop_event.set()
        
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=30)
            
        logger.info("Workflow scheduler stopped.")
        
    def _save_state(self):
        """Save scheduler state to a file."""
        try:
            os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
            state = {
                'schedules': {id: config.to_dict() for id, config in self.schedules.items()},
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving scheduler state: {e}", exc_info=True)
            
    def get_schedule_status(self) -> List[Dict[str, Any]]:
        """Get status of all scheduled pipelines."""
        return [
            {
                'id': id,
                'pipeline_name': config.pipeline.name,
                'schedule_type': config.schedule_type,
                'last_run': config.last_run.isoformat() if config.last_run else None,
                'next_run': config.next_run.isoformat() if config.next_run else None,
                'run_count': config.run_count,
                'enabled': config.enabled,
                'status': 'running' if id in self.running and self.running[id].is_alive() else 'idle'
            }
            for id, config in self.schedules.items()
        ]
        
    def disable_schedule(self, schedule_id: str) -> bool:
        """Disable a scheduled pipeline."""
        if schedule_id in self.schedules:
            self.schedules[schedule_id].enabled = False
            self._save_state()
            return True
        return False
        
    def enable_schedule(self, schedule_id: str) -> bool:
        """Enable a scheduled pipeline."""
        if schedule_id in self.schedules:
            self.schedules[schedule_id].enabled = True
            self._save_state()
            return True
        return False
        
    def remove_schedule(self, schedule_id: str) -> bool:
        """Remove a scheduled pipeline."""
        if schedule_id in self.schedules:
            del self.schedules[schedule_id]
            self._save_state()
            return True
        return False

def main():
    """Run the scheduler as a standalone service."""
    logger.info("Starting workflow scheduler service...")
    
    # Create scheduler instance
    scheduler = WorkflowScheduler(check_interval=10)
    
    # TODO: Load pipelines and schedules from configuration
    
    # Start the scheduler
    scheduler.start()
    
    # Keep the main thread alive
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
    finally:
        scheduler.stop()
        
if __name__ == "__main__":
    main()

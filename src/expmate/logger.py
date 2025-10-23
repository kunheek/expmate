import json
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional


class ExperimentLogger:
    """Experiment logger with structured logging and metrics tracking.

    This logger provides two equivalent APIs for text logging:
    1. Generic: logger.log(message, level='INFO')
    2. Level-specific: logger.info(message), logger.warning(message), logger.error(message)

    For metrics tracking, use logger.log_metric(step, split, name, value)

    Features:
        - Rank-aware logging (DDP compatible)
        - Human-readable and machine-readable logs
        - Metrics tracking with best model tracking
        - Profiling context managers
        - Configurable log levels

    Args:
        run_dir: Directory for log files
        rank: Process rank (0 for main process)
        run_id: Unique run identifier
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        console_output: Whether to output to console (only rank 0)

    Examples:
        >>> # Basic usage
        >>> logger = ExperimentLogger(run_dir='runs/exp1')
        >>>
        >>> # Text logging (two equivalent ways)
        >>> logger.info("Training started")  # Preferred
        >>> logger.log("Training started", level="INFO")  # Alternative
        >>>
        >>> # Metrics logging
        >>> logger.log_metric(step=0, split='train', name='loss', value=0.5)
        >>>
        >>> # Profiling
        >>> with logger.profile('data_loading'):
        >>>     data = load_data()
    """

    def __init__(
        self,
        run_dir: Path | str,
        rank: int = 0,
        run_id: str = None,
        log_level: str = "INFO",
        console_output: bool = True,
    ):
        # Convert to Path if string
        if isinstance(run_dir, str):
            run_dir = Path(run_dir)

        self.run_dir = run_dir
        self.rank = rank
        self.run_id = run_id or "unknown"

        run_dir.mkdir(parents=True, exist_ok=True)

        # Human-readable log (per-rank)
        log_filename = "exp.log" if rank == 0 else f"exp_rank{rank}.log"
        self.log_file = run_dir / log_filename
        self.log_handler = logging.FileHandler(self.log_file)
        self.log_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        # JSONL log for machine reading (per-rank)
        jsonl_filename = "events.jsonl" if rank == 0 else f"events_rank{rank}.jsonl"
        self.jsonl_file = run_dir / jsonl_filename

        # Console handler (rank-0 only by default)
        self.console_handler = None
        if rank == 0 and console_output:
            self.console_handler = logging.StreamHandler(sys.stdout)
            self.console_handler.setFormatter(
                logging.Formatter("%(levelname)s: %(message)s")
            )

        # Metrics CSV (rank-0 only)
        self.metrics_file = run_dir / "metrics.csv"
        if rank == 0 and not self.metrics_file.exists():
            with open(self.metrics_file, "w") as f:
                f.write("step,split,name,value,wall_time\n")

        # Best metrics tracking (per metric)
        self.best_metrics: Dict[
            str, Dict
        ] = {}  # {metric_name: {'value': ..., 'step': ..., 'mode': ...}}
        self.best_file = run_dir / "best.json"
        if rank == 0 and self.best_file.exists():
            with open(self.best_file) as f:
                self.best_metrics = json.load(f)

        # Setup logger
        self.logger = logging.getLogger(f"expmate_rank_{rank}")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        self.logger.addHandler(self.log_handler)
        if self.console_handler:
            self.logger.addHandler(self.console_handler)

        # Prevent propagation to root logger
        self.logger.propagate = False

    def log(self, message: str, level: str = "INFO", **kwargs):
        """Log a message with context.

        Args:
            message: The log message
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            **kwargs: Additional context to include in log
        """
        # Check if this level should be logged
        log_level_num = getattr(logging, level.upper(), logging.INFO)
        logger_level_num = self.logger.level

        # Only write to JSONL and logger if level is high enough
        if log_level_num >= logger_level_num:
            # Write to JSONL
            log_entry = {
                "timestamp": time.time(),
                "level": level.upper(),
                "message": message,
                "run_id": self.run_id,
                "rank": self.rank,
                **kwargs,
            }

            with open(self.jsonl_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            # Write to logger
            self.logger.log(log_level_num, message)

    def info(self, message: str, **kwargs):
        self.log(message, "INFO", **kwargs)

    def warning(self, message: str, **kwargs):
        self.log(message, "WARNING", **kwargs)

    def error(self, message: str, **kwargs):
        self.log(message, "ERROR", **kwargs)

    def log_metric(
        self,
        step: int,
        split: str,
        name: str,
        value: float,
        track_best: bool = True,
        mode: Optional[str] = None,
    ):
        """Log a metric to CSV and update best tracking.

        Args:
            step: Training step
            split: Data split (train/val/test)
            name: Metric name
            value: Metric value
            track_best: Whether to track this as a best metric
            mode: 'min' or 'max' for best tracking (auto-detected if None)
        """
        if self.rank != 0:
            return  # Only rank 0 logs metrics

        wall_time = time.time()
        with open(self.metrics_file, "a") as f:
            f.write(f"{step},{split},{name},{value},{wall_time}\n")

        # Update best metrics tracking
        if track_best:
            # Auto-detect mode if not specified
            if mode is None:
                name_lower = name.lower()
                if any(x in name_lower for x in ["loss", "error"]):
                    mode = "min"
                elif any(x in name_lower for x in ["acc", "accuracy", "f1", "auc"]):
                    mode = "max"
                else:
                    mode = "min"  # Default to min

            metric_key = f"{split}/{name}"

            # Check if this is a new best
            is_best = False
            if metric_key not in self.best_metrics:
                is_best = True
            else:
                prev_best = self.best_metrics[metric_key]["value"]
                is_better = (mode == "min" and value < prev_best) or (
                    mode == "max" and value > prev_best
                )
                if is_better:
                    is_best = True

            # Update if best
            if is_best:
                self.best_metrics[metric_key] = {
                    "value": value,
                    "step": step,
                    "mode": mode,
                }

                # Save best metrics
                with open(self.best_file, "w") as f:
                    json.dump(self.best_metrics, f, indent=2)

    def get_best_metric(self, name: str, split: str = "val") -> Optional[Dict]:
        """Get the best value for a metric.

        Args:
            name: Metric name
            split: Data split

        Returns:
            dict: Best metric info with 'value', 'step', 'mode' or None
        """
        metric_key = f"{split}/{name}"
        return self.best_metrics.get(metric_key)

    @contextmanager
    def profile(self, name: str, log_result: bool = True):
        """Context manager for profiling code sections.

        Args:
            name: Name of the profiled section
            log_result: Whether to log the timing result

        Usage:
            with logger.profile("data_loading"):
                data = load_data()

        Yields:
            dict: Dictionary that will contain 'elapsed' time after completion
        """
        result = {"elapsed": 0.0}
        start_time = time.perf_counter()

        try:
            yield result
        finally:
            elapsed = time.perf_counter() - start_time
            result["elapsed"] = elapsed

            if log_result and self.rank == 0:
                self.info(
                    f"Profile [{name}]: {elapsed:.4f}s", section=name, elapsed=elapsed
                )

    def set_log_level(self, level: str):
        """Change the logging level.

        Args:
            level: One of DEBUG, INFO, WARNING, ERROR
        """
        self.logger.setLevel(getattr(logging, level.upper()))

    def close(self):
        """Close all log handlers."""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)

    def log_dict(self, data: Dict, level: str = "INFO"):
        """Log a dictionary as a single entry.

        Args:
            data: Dictionary to log
            level: Log level
        """
        log_entry = {
            "timestamp": time.time(),
            "level": level.upper(),
            "run_id": self.run_id,
            "rank": self.rank,
            **data,
        }

        with open(self.jsonl_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def save_metadata(self, metadata: Dict, filename: str = "metadata.json"):
        """Save metadata to a JSON file.

        Args:
            metadata: Dictionary containing metadata
            filename: Name of file to save to
        """
        filepath = self.run_dir / filename
        with open(filepath, "w") as f:
            json.dump(metadata, f, indent=2)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False

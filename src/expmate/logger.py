import inspect
import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Optional

import expmate


class ColorFormatter(logging.Formatter):
    """Custom formatter that adds colors to console output."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"

    # Special colors for specific message types
    STAGE_COLOR = "\033[34m"  # Blue
    TIMER_COLOR = "\033[36m"  # Cyan
    METRIC_COLOR = "\033[35m"  # Magenta

    def format(self, record):
        """Format log record with colors."""
        # Get the base message
        message = record.getMessage()

        # Determine color based on level
        level_color = self.COLORS.get(record.levelname, "")

        # Special coloring for specific message patterns
        if message.startswith("Stage ["):
            # Stage messages in blue with bold stage name
            parts = message.split("]", 1)
            if len(parts) == 2:
                stage_name = parts[0] + "]"
                rest = parts[1]
                colored_message = (
                    f"{self.STAGE_COLOR}{self.BOLD}{stage_name}{self.RESET}"
                    f"{self.STAGE_COLOR}{rest}{self.RESET}"
                )
            else:
                colored_message = f"{self.STAGE_COLOR}{message}{self.RESET}"
        elif message.startswith("Timer ["):
            # Timer messages in cyan with bold timer name
            parts = message.split("]", 1)
            if len(parts) == 2:
                timer_name = parts[0] + "]"
                rest = parts[1]
                colored_message = (
                    f"{self.TIMER_COLOR}{self.BOLD}{timer_name}{self.RESET}"
                    f"{self.TIMER_COLOR}{rest}{self.RESET}"
                )
            else:
                colored_message = f"{self.TIMER_COLOR}{message}{self.RESET}"
        else:
            # Regular messages: color based on level
            level_name = f"{level_color}{self.BOLD}{record.levelname}{self.RESET}"
            colored_message = f"{level_color}{message}{self.RESET}"
            return f"{level_name}: {colored_message}"

        # For special messages, just add level name without extra coloring
        level_name = f"{level_color}{self.BOLD}{record.levelname}{self.RESET}"
        return f"{level_name}: {colored_message}"


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
        - Profiling context managers (can be disabled globally)
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
        >>> # Profiling (enabled by default)
        >>> with logger.timer('data_loading'):
        >>>     data = load_data()
        >>>
        >>> # Disable profiling globally for production
        >>> import expmate
        >>> expmate.timer = False  # or set EM_TIMER=0 environment variable
    """

    def __init__(
        self,
        run_dir: Path | str,
        rank: int = 0,
        run_id: str = None,
        log_level: str = None,
        console_output: bool = None,
    ):
        # Convert to Path if string
        if isinstance(run_dir, str):
            run_dir = Path(run_dir)

        self.run_dir = run_dir
        self.rank = rank
        self.run_id = run_id or "unknown"

        run_dir.mkdir(parents=True, exist_ok=True)

        # Use global log level if not specified
        if log_level is None:
            log_level = expmate.log_level

        # Use global verbose flag if console_output not specified
        if console_output is None:
            console_output = expmate.verbose

        # Human-readable log (per-rank) - NO COLORS in file
        log_filename = "exp.log" if rank == 0 else f"exp_rank{rank}.log"
        self.log_file = run_dir / log_filename
        self.log_handler = logging.FileHandler(self.log_file)
        self.log_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        )

        # JSONL log for machine reading (per-rank)
        jsonl_filename = "events.jsonl" if rank == 0 else f"events_rank{rank}.jsonl"
        self.jsonl_file = run_dir / jsonl_filename

        # Console handler (rank-0 only by default) - WITH COLORS
        self.console_handler = None
        if rank == 0 and console_output:
            self.console_handler = logging.StreamHandler(sys.stdout)
            # Check if stdout supports colors (not piped/redirected)
            use_colors = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
            if use_colors:
                self.console_handler.setFormatter(ColorFormatter())
            else:
                # Fallback to plain format if output is redirected
                self.console_handler.setFormatter(
                    logging.Formatter("%(levelname)s: %(message)s")
                )

        # Metrics CSV (rank-0 only) - respect track_metrics flag
        self.metrics_file = run_dir / "metrics.csv"
        if rank == 0 and expmate.track_metrics and not self.metrics_file.exists():
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

        # Setup logger (unique name to avoid handler accumulation)
        logger_name = f"expmate_rank_{rank}_{uuid.uuid4().hex[:8]}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Clear any existing handlers (in case logger was reused)
        self.logger.handlers.clear()

        self.logger.addHandler(self.log_handler)
        if self.console_handler:
            self.logger.addHandler(self.console_handler)

        # Prevent propagation to root logger
        self.logger.propagate = False

        # Stage tracking (for hierarchical context)
        self._current_stage: Optional[str] = None
        self._stage_stack: list = []

        # Rate limiting tracking
        self._rate_limit_counters: Dict[str, int] = {}
        self._rate_limit_last_log: Dict[str, float] = {}

        # Time estimation tracking (for automatic ETA in log_metric)
        self._eta_tracking: Dict[
            str, Dict
        ] = {}  # {key: {'start_time': ..., 'start_step': ..., 'total_steps': ...}}

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

    def debug(self, message: str, **kwargs):
        self.log(message, "DEBUG", **kwargs)

    def info(self, message: str, **kwargs):
        self.log(message, "INFO", **kwargs)

    def warning(self, message: str, **kwargs):
        self.log(message, "WARNING", **kwargs)

    def error(self, message: str, **kwargs):
        self.log(message, "ERROR", **kwargs)

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds into human-readable time string.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted time string (e.g., "2h 15m 30s", "45s", "1d 3h 20m")
        """
        if seconds < 0:
            return "0s"

        days = int(seconds // 86400)
        seconds %= 86400
        hours = int(seconds // 3600)
        seconds %= 3600
        minutes = int(seconds // 60)
        secs = int(seconds % 60)

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if secs > 0 or not parts:
            parts.append(f"{secs}s")

        return " ".join(parts)

    def log_metric(
        self,
        step: int,
        split: str,
        name: str,
        value: float,
        track_best: bool = True,
        mode: Optional[str] = None,
        total_steps: Optional[int] = None,
        eta_key: Optional[str] = None,
    ):
        """Log a metric to CSV and update best tracking.

        Args:
            step: Training step
            split: Data split (train/val/test)
            name: Metric name
            value: Metric value
            track_best: Whether to track this as a best metric
            mode: 'min' or 'max' for best tracking (auto-detected if None)
            total_steps: Total number of steps (enables automatic ETA estimation)
            eta_key: Key for ETA tracking (auto-generated from split/name if None)
        """
        if self.rank != 0:
            return  # Only rank 0 logs metrics

        # Skip if metrics tracking is disabled
        if not expmate.track_metrics:
            return

        wall_time = time.time()
        with open(self.metrics_file, "a") as f:
            f.write(f"{step},{split},{name},{value},{wall_time}\n")

        # Automatic ETA estimation if total_steps is provided
        if total_steps is not None and step > 0:
            # Auto-generate key if not provided
            if eta_key is None:
                eta_key = f"{split}/{name}"

            # Initialize tracking for this metric if first time
            if eta_key not in self._eta_tracking:
                self._eta_tracking[eta_key] = {
                    "start_time": wall_time,
                    "start_step": step,
                    "total_steps": total_steps,
                }

            # Calculate ETA
            tracking = self._eta_tracking[eta_key]
            elapsed = wall_time - tracking["start_time"]
            steps_done = step - tracking["start_step"]

            if steps_done > 0:
                steps_remaining = total_steps - step
                time_per_step = elapsed / steps_done
                eta_seconds = time_per_step * steps_remaining

                # Log ETA info (only if significant progress made to avoid spam)
                if steps_done % max(1, total_steps // 20) == 0 or step == total_steps:
                    progress_pct = 100 * step / total_steps
                    eta_str = self._format_time(eta_seconds)
                    elapsed_str = self._format_time(elapsed)

                    self.info(
                        f"Progress [{split}/{name}]: {step}/{total_steps} ({progress_pct:.1f}%) | "
                        f"Elapsed: {elapsed_str} | ETA: {eta_str}",
                        step=step,
                        progress=progress_pct,
                        eta_seconds=eta_seconds,
                        elapsed_seconds=elapsed,
                    )

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
    def timer(self, name: str, log_result: bool = True):
        """Context manager for timing code sections.

        Provides simple wall-clock timing using time.perf_counter().
        For detailed GPU/CPU profiling, use torch.profiler directly.

        Timing can be globally disabled by setting expmate.timer = False
        or environment variable EM_TIMER=0.

        Args:
            name: Name of the timed section
            log_result: Whether to log the timing result

        Usage:
            # Basic timing
            with logger.timer("data_loading"):
                data = load_data()

            # Get elapsed time
            with logger.timer("training_step") as result:
                loss = model(batch)
            print(f"Step took {result['elapsed']:.4f}s")

            # Silent timing (no logging)
            with logger.timer("forward", log_result=False) as result:
                output = model(input)
            elapsed = result['elapsed']

        Yields:
            dict: Dictionary with 'elapsed' key (in seconds), updated when context exits

        Example with PyTorch Profiler:
            # Use torch.profiler for detailed profiling
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
            ) as prof:
                with logger.timer("training_step"):
                    loss = model(batch)
                    loss.backward()
            prof.export_chrome_trace("trace.json")
        """
        result = {"elapsed": 0.0}

        # Check if profiling is enabled globally
        if not expmate.timer:
            yield result
            return

        start_time = time.perf_counter()

        try:
            yield result
        finally:
            result["elapsed"] = time.perf_counter() - start_time

            if log_result and self.rank == 0:
                self.info(
                    f"Timer [{name}]: {result['elapsed']:.4f}s",
                    section=name,
                    elapsed=result["elapsed"],
                )

    @contextmanager
    def stage(self, name: str, **metadata):
        """Context manager for tracking training stages with hierarchical context.

        Automatically logs stage entry/exit, tracks duration, and provides context
        for metrics and logs. Stages can be nested to create hierarchies
        (e.g., epoch → train → batch).

        Args:
            name: Name of the stage (e.g., 'epoch', 'train', 'validation')
            **metadata: Additional metadata to log (e.g., epoch=5, batch=10)

        Usage:
            # Simple stage
            with logger.stage("training"):
                train_model()

            # With metadata
            with logger.stage("epoch", epoch=5, lr=0.001):
                train_epoch()

            # Nested stages
            with logger.stage("epoch", epoch=5):
                with logger.stage("train"):
                    train_loss = train_epoch()
                with logger.stage("validation"):
                    val_loss = validate()

        Yields:
            dict: Stage info with 'name', 'metadata', 'elapsed' keys
        """
        stage_info = {"name": name, "metadata": metadata, "elapsed": 0.0}

        # Build hierarchical stage path
        if self._stage_stack:
            parent = self._stage_stack[-1]["name"]
            full_name = f"{parent}/{name}"
        else:
            full_name = name

        # Log stage entry
        if self.rank == 0:
            metadata_str = ", ".join(f"{k}={v}" for k, v in metadata.items())
            msg = f"Stage [{full_name}]"
            if metadata_str:
                msg += f" ({metadata_str})"
            msg += " - START"
            self.info(msg, stage=full_name, stage_event="start", **metadata)

        # Push to stack
        self._stage_stack.append(stage_info)
        old_stage = self._current_stage
        self._current_stage = full_name

        start_time = time.perf_counter()

        try:
            yield stage_info
        finally:
            elapsed = time.perf_counter() - start_time
            stage_info["elapsed"] = elapsed

            # Log stage exit
            if self.rank == 0:
                msg = f"Stage [{full_name}] - END ({elapsed:.4f}s)"
                self.info(
                    msg, stage=full_name, stage_event="end", elapsed=elapsed, **metadata
                )

            # Pop from stack
            self._stage_stack.pop()
            self._current_stage = old_stage

    def should_run(
        self, every: int = None, seconds: float = None, key: str = None
    ) -> bool:
        """Check if code should run based on rate limiting.

        Returns a boolean indicating whether to execute. Use with 'if' for clean syntax.

        Args:
            every: Execute every N iterations (mutually exclusive with seconds)
            seconds: Execute every N seconds (mutually exclusive with every)
            key: Unique key for this rate limiter (auto-generated if None)

        Returns:
            bool: True if code should execute, False if skipped

        Usage:
            # Clean 'if' syntax
            for step in range(10000):
                if logger.should_run(every=100):
                    logger.info(f"Step {step}")
                    save_checkpoint()

            # Time-based rate limiting
            for batch in dataloader:
                if logger.should_run(seconds=5.0):
                    logger.info("Expensive logging")
        """
        if every is None and seconds is None:
            raise ValueError("Must specify either 'every' or 'seconds'")
        if every is not None and seconds is not None:
            raise ValueError("Cannot specify both 'every' and 'seconds'")

        # Auto-generate key from caller location if not provided
        if key is None:
            frame = inspect.currentframe()
            caller_frame = frame.f_back if frame else None
            if caller_frame:
                key = f"{caller_frame.f_code.co_filename}:{caller_frame.f_lineno}"
            else:
                key = "default"

        should_execute = False

        if every is not None:
            # Iteration-based rate limiting
            count = self._rate_limit_counters.get(key, 0)
            self._rate_limit_counters[key] = count + 1

            if count % every == 0:
                should_execute = True
        else:
            # Time-based rate limiting
            current_time = time.time()
            last_execute_time = self._rate_limit_last_log.get(key, 0.0)

            if current_time - last_execute_time >= seconds:
                should_execute = True
                self._rate_limit_last_log[key] = current_time

        return should_execute

    # Backward compatibility alias - wraps should_run as context manager
    @contextmanager
    def do_every(self, every: int = None, seconds: float = None, key: str = None):
        """Deprecated: Use should_run() instead.

        Context manager version for backward compatibility.
        """
        should_execute = self.should_run(every=every, seconds=seconds, key=key)
        yield should_execute

    @contextmanager
    def log_every(self, every: int = None, seconds: float = None, key: str = None):
        """Context manager for rate-limited logging.

        Logs only every N iterations or every N seconds, useful for reducing
        log spam in tight training loops while still capturing data in JSONL.

        Args:
            every: Log every N iterations (mutually exclusive with seconds)
            seconds: Log every N seconds (mutually exclusive with every)
            key: Unique key for this rate limiter (auto-generated if None)

        Usage:
            # Log every 100 iterations
            for step in range(10000):
                loss = train_step()
                with logger.log_every(every=100):
                    logger.info(f"Step {step}: loss={loss:.4f}")

            # Log every 5 seconds
            for batch in dataloader:
                with logger.log_every(seconds=5.0):
                    logger.info(f"Processing batch...")

            # Multiple rate limiters with different keys
            for step in range(1000):
                with logger.log_every(every=10, key="loss"):
                    logger.info(f"Loss: {loss:.4f}")
                with logger.log_every(every=100, key="detailed"):
                    logger.info(f"Detailed metrics: {metrics}")

        Yields:
            bool: True if logging should occur, False if suppressed
        """
        # Use should_run to determine if we should log
        should_log = self.should_run(every=every, seconds=seconds, key=key)

        # Temporarily suppress logging if not time to log
        if not should_log and self.console_handler:
            # Remove console handler temporarily
            if self.console_handler in self.logger.handlers:
                self.logger.removeHandler(self.console_handler)
                try:
                    yield should_log
                finally:
                    # Restore console handler
                    if self.console_handler not in self.logger.handlers:
                        self.logger.addHandler(self.console_handler)
            else:
                # Handler not present, just yield
                yield should_log
        else:
            yield should_log

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

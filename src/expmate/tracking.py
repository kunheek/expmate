import warnings
from pathlib import Path
from typing import Any, Dict, Optional


class WandbTracker:
    """Weights & Biases integration.

    Args:
        project: WandB project name
        name: Run name
        config: Configuration dictionary to log
        **kwargs: Additional arguments passed to wandb.init()
    """

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        **kwargs,
    ):
        try:
            import wandb

            self.wandb = wandb
        except ImportError as e:
            msg = "wandb is not installed. Install with: pip install wandb"
            raise ImportError(msg) from e

        self.run = self.wandb.init(project=project, name=name, config=config, **kwargs)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to WandB.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        self.wandb.log(metrics, step=step)

    def log_artifact(self, path: Path, name: str, artifact_type: str = "model"):
        """Log a file artifact to WandB.

        Args:
            path: Path to artifact file
            name: Artifact name
            artifact_type: Type of artifact (model, dataset, etc.)
        """
        artifact = self.wandb.Artifact(name, type=artifact_type)
        artifact.add_file(str(path))
        self.wandb.log_artifact(artifact)

    def finish(self):
        """Finish the WandB run."""
        self.wandb.finish()


class TensorBoardTracker:
    """TensorBoard integration.

    Args:
        log_dir: Directory for TensorBoard logs
    """

    def __init__(self, log_dir: Path):
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.SummaryWriter = SummaryWriter
        except ImportError as e:
            raise ImportError(
                "tensorboard is not installed. Install with: pip install tensorboard"
            ) from e

        self.writer = self.SummaryWriter(log_dir=str(log_dir))

    def log(self, metrics: Dict[str, Any], step: int):
        """Log metrics to TensorBoard.

        Args:
            metrics: Dictionary of metrics to log
            step: Step number
        """
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(key, value, step)

    def log_histogram(self, tag: str, values, step: int):
        """Log a histogram to TensorBoard.

        Args:
            tag: Tag for the histogram
            values: Values to create histogram from
            step: Step number
        """
        self.writer.add_histogram(tag, values, step)

    def log_image(self, tag: str, image, step: int):
        """Log an image to TensorBoard.

        Args:
            tag: Tag for the image
            image: Image tensor
            step: Step number
        """
        self.writer.add_image(tag, image, step)

    def log_text(self, tag: str, text: str, step: int):
        """Log text to TensorBoard.

        Args:
            tag: Tag for the text
            text: Text to log
            step: Step number
        """
        self.writer.add_text(tag, text, step)

    def flush(self):
        """Flush pending events to disk."""
        self.writer.flush()

    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()


class MultiTracker:
    """Track experiments with multiple tools simultaneously.

    Args:
        trackers: List of tracker instances (WandbTracker, TensorBoardTracker, etc.)

    Example:
        tracker = MultiTracker([
            WandbTracker(project='my_project', name='run1'),
            TensorBoardTracker(log_dir='runs/run1')
        ])

        tracker.log({'loss': 0.5, 'acc': 0.9}, step=100)
    """

    def __init__(self, trackers: list):
        self.trackers = trackers

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to all trackers.

        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number
        """
        for tracker in self.trackers:
            try:
                if hasattr(tracker, "log"):
                    if step is not None:
                        tracker.log(metrics, step=step)
                    else:
                        tracker.log(metrics)
            except Exception as e:
                msg = f"Failed to log to {type(tracker).__name__}: {e}"
                warnings.warn(msg, stacklevel=2)

    def finish(self):
        """Finish all trackers."""
        for tracker in self.trackers:
            try:
                if hasattr(tracker, "finish"):
                    tracker.finish()
                elif hasattr(tracker, "close"):
                    tracker.close()
            except Exception as e:
                msg = f"Failed to finish {type(tracker).__name__}: {e}"
                warnings.warn(msg, stacklevel=2)


def create_tracker(backend: str, **kwargs) -> Any:
    """Factory function to create a tracker.

    Args:
        backend: Tracker backend ('wandb', 'tensorboard', or 'multi')
        **kwargs: Arguments specific to the tracker

    Returns:
        Tracker instance

    Example:
        # WandB
        tracker = create_tracker('wandb', project='my_project', name='run1')

        # TensorBoard
        tracker = create_tracker('tensorboard', log_dir='runs/run1')

        # Multiple
        tracker = create_tracker('multi', trackers=[...])
    """
    if backend == "wandb":
        return WandbTracker(**kwargs)
    if backend == "tensorboard":
        return TensorBoardTracker(**kwargs)
    if backend == "multi":
        return MultiTracker(**kwargs)
    raise ValueError(f"Unknown tracker backend: {backend}")

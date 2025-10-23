import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch


class CheckpointManager:
    """Manager for saving and loading model checkpoints.

    Features:
        - Save/load checkpoints with metadata
        - Track best checkpoints based on metrics
        - Automatic cleanup of old checkpoints
        - Support for both model state_dict and full model

    Args:
        checkpoint_dir: Directory to save checkpoints
        keep_last: Number of last checkpoints to keep (None = keep all)
        keep_best: Number of best checkpoints to keep (None = keep all)
        metric_name: Name of metric to track for best checkpoints
        mode: 'min' or 'max' - whether lower or higher metric is better
    """

    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        keep_last: Optional[int] = 3,
        keep_best: Optional[int] = 3,
        metric_name: str = "loss",
        mode: str = "min",
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.keep_last = keep_last
        self.keep_best = keep_best
        self.metric_name = metric_name
        self.mode = mode

        # Track checkpoint history
        self.history_file = self.checkpoint_dir / "checkpoint_history.json"
        self.history = self._load_history()

    def _load_history(self) -> List[Dict[str, Any]]:
        """Load checkpoint history from file."""
        if self.history_file.exists():
            with open(self.history_file) as f:
                return json.load(f)
        return []

    def _save_history(self):
        """Save checkpoint history to file."""
        with open(self.history_file, "w") as f:
            json.dump(self.history, f, indent=2)

    def save(
        self,
        model: Any,  # torch.nn.Module
        optimizer: Optional[Any] = None,  # torch.optim.Optimizer
        scheduler: Optional[Any] = None,
        epoch: Optional[int] = None,
        step: Optional[int] = None,
        metrics: Optional[Dict[str, float]] = None,
        extra: Optional[Dict[str, Any]] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """Save a checkpoint.

        Args:
            model: PyTorch model to save
            optimizer: Optimizer state to save
            scheduler: LR scheduler state to save
            epoch: Current epoch number
            step: Current training step
            metrics: Dictionary of metrics (e.g., {'loss': 0.5, 'acc': 0.9})
            extra: Any extra data to save
            filename: Custom filename (default: checkpoint_step_{step}.pt
                     or checkpoint_epoch_{epoch}.pt)

        Returns:
            Path: Path to saved checkpoint file
        """
        if filename is None:
            if step is not None:
                filename = f"checkpoint_step_{step}.pt"
            elif epoch is not None:
                filename = f"checkpoint_epoch_{epoch}.pt"
            else:
                filename = f"checkpoint_{int(time.time())}.pt"

        checkpoint_path = self.checkpoint_dir / filename

        # Build checkpoint dictionary
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "step": step,
            "metrics": metrics or {},
        }

        if optimizer is not None:
            checkpoint["optimizer_state_dict"] = optimizer.state_dict()

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        if extra is not None:
            checkpoint["extra"] = extra

        # Save checkpoint
        torch.save(checkpoint, checkpoint_path)

        # Update history
        history_entry = {
            "filename": filename,
            "epoch": epoch,
            "step": step,
            "metrics": metrics or {},
        }
        self.history.append(history_entry)
        self._save_history()

        # Cleanup old checkpoints
        self._cleanup()

        return checkpoint_path

    def load(
        self,
        model: Any,  # torch.nn.Module
        optimizer: Optional[Any] = None,  # torch.optim.Optimizer
        scheduler: Optional[Any] = None,
        filename: Optional[str] = None,
        checkpoint_path: Optional[Union[str, Path]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        """Load a checkpoint.

        Args:
            model: Model to load state into
            optimizer: Optimizer to load state into
            scheduler: Scheduler to load state into
            filename: Checkpoint filename to load
            checkpoint_path: Full path to checkpoint (overrides filename)
            strict: Whether to strictly enforce state_dict keys match

        Returns:
            dict: Checkpoint metadata (epoch, step, metrics, extra)
        """
        if checkpoint_path is not None:
            path = Path(checkpoint_path)
        elif filename is not None:
            path = self.checkpoint_dir / filename
        else:
            # Load latest checkpoint
            path = self.get_latest_checkpoint()
            if path is None:
                raise ValueError("No checkpoints found")

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

        # Load optimizer state
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load scheduler state
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        # Return metadata
        return {
            "epoch": checkpoint.get("epoch"),
            "step": checkpoint.get("step"),
            "metrics": checkpoint.get("metrics", {}),
            "extra": checkpoint.get("extra", {}),
        }

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to the latest checkpoint."""
        if not self.history:
            return None

        latest = self.history[-1]
        path = self.checkpoint_dir / latest["filename"]

        if path.exists():
            return path
        return None

    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to the best checkpoint based on metric."""
        if not self.history:
            return None

        # Filter history entries that have the metric
        valid_entries = [
            entry
            for entry in self.history
            if self.metric_name in entry.get("metrics", {})
        ]

        if not valid_entries:
            return None

        # Find best based on mode
        if self.mode == "min":
            best = min(valid_entries, key=lambda x: x["metrics"][self.metric_name])
        else:
            best = max(valid_entries, key=lambda x: x["metrics"][self.metric_name])

        path = self.checkpoint_dir / best["filename"]

        if path.exists():
            return path
        return None

    def _cleanup(self):
        """Clean up old checkpoints based on keep_last and keep_best settings."""
        if not self.history:
            return

        # Determine which checkpoints to keep
        keep_files = set()

        # Keep last N checkpoints
        if self.keep_last is not None and self.keep_last > 0:
            for entry in self.history[-self.keep_last :]:
                keep_files.add(entry["filename"])

        # Keep best N checkpoints
        if self.keep_best is not None and self.keep_best > 0:
            # Filter entries with the metric
            valid_entries = [
                entry
                for entry in self.history
                if self.metric_name in entry.get("metrics", {})
            ]

            if valid_entries:
                # Sort by metric
                sorted_entries = sorted(
                    valid_entries,
                    key=lambda x: x["metrics"][self.metric_name],
                    reverse=(self.mode == "max"),
                )

                for entry in sorted_entries[: self.keep_best]:
                    keep_files.add(entry["filename"])

        # If no limits set, keep all
        if self.keep_last is None and self.keep_best is None:
            return

        # Delete checkpoints not in keep list
        for entry in self.history:
            filename = entry["filename"]
            if filename not in keep_files:
                checkpoint_path = self.checkpoint_dir / filename
                if checkpoint_path.exists():
                    checkpoint_path.unlink()

        # Update history to only include kept checkpoints
        self.history = [
            entry for entry in self.history if entry["filename"] in keep_files
        ]
        self._save_history()

    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints with their metadata."""
        return self.history.copy()

    def delete_all(self):
        """Delete all checkpoints and history."""
        if self.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history = []
        self._save_history()


def save_checkpoint(
    path: Union[str, Path],
    model: Any,  # torch.nn.Module
    optimizer: Optional[Any] = None,  # torch.optim.Optimizer
    **kwargs,
) -> None:
    """Simple function to save a checkpoint.

    Args:
        path: Path to save checkpoint
        model: Model to save
        optimizer: Optimizer to save
        **kwargs: Additional data to save in checkpoint
    """
    checkpoint = {"model_state_dict": model.state_dict(), **kwargs}

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(
    path: Union[str, Path],
    model: Any,  # torch.nn.Module
    optimizer: Optional[Any] = None,  # torch.optim.Optimizer
    strict: bool = True,
) -> Dict[str, Any]:
    """Simple function to load a checkpoint.

    Args:
        path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        strict: Whether to strictly enforce state_dict keys match

    Returns:
        dict: Checkpoint contents
    """
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint

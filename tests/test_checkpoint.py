"""Tests for checkpoint module."""

import json

import pytest

try:
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)

    def forward(self, x):
        return self.fc(x)


@pytest.fixture
def model():
    """Create a simple model for testing."""
    return SimpleModel()


@pytest.fixture
def optimizer(model):
    """Create an optimizer for testing."""
    return torch.optim.Adam(model.parameters(), lr=0.001)


class TestCheckpointManager:
    """Test CheckpointManager class."""

    def test_init(self, temp_dir):
        from expmate.torch.checkpoint import CheckpointManager

        manager = CheckpointManager(checkpoint_dir=str(temp_dir))
        assert manager.checkpoint_dir == temp_dir
        assert temp_dir.exists()

    def test_save_checkpoint(self, temp_dir, model, optimizer):
        from expmate.torch.checkpoint import CheckpointManager

        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        checkpoint_path = manager.save(
            epoch=0,
            model=model,
            optimizer=optimizer,
            metrics={"loss": 0.5, "accuracy": 0.8},
        )

        assert checkpoint_path.exists()
        assert "checkpoint" in checkpoint_path.name

    def test_load_checkpoint(self, temp_dir, model, optimizer):
        from expmate.torch.checkpoint import CheckpointManager

        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        # Save checkpoint
        checkpoint_path = manager.save(
            epoch=5, model=model, optimizer=optimizer, metrics={"loss": 0.3}
        )

        # Create new model and optimizer to load into
        new_model = SimpleModel()
        new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)

        # Load checkpoint - it loads state directly into model/optimizer
        metadata = manager.load(
            new_model, new_optimizer, checkpoint_path=checkpoint_path
        )

        assert metadata["epoch"] == 5
        assert metadata["metrics"]["loss"] == 0.3

        # Verify model state was loaded
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)

    def test_keep_last_n_checkpoints(self, temp_dir, model, optimizer):
        from expmate.torch.checkpoint import CheckpointManager

        manager = CheckpointManager(checkpoint_dir=str(temp_dir), keep_last=2)

        # Save 5 checkpoints
        for epoch in range(5):
            manager.save(epoch=epoch, model=model, optimizer=optimizer)

        # Should only keep last 2
        checkpoints = list(temp_dir.glob("checkpoint_*.pt"))
        assert len(checkpoints) == 2

        # Verify it's the last 2 epochs
        epochs = [int(cp.stem.split("_")[-1]) for cp in checkpoints]
        assert sorted(epochs) == [3, 4]

    def test_track_best_checkpoints(self, temp_dir, model, optimizer):
        from expmate.torch.checkpoint import CheckpointManager

        manager = CheckpointManager(
            checkpoint_dir=str(temp_dir),
            keep_last=None,  # Don't keep by recency
            keep_best=2,
            metric_name="loss",
            mode="min",
        )

        # Save checkpoints with different losses
        manager.save(epoch=0, model=model, optimizer=optimizer, metrics={"loss": 1.0})
        manager.save(epoch=1, model=model, optimizer=optimizer, metrics={"loss": 0.5})
        manager.save(epoch=2, model=model, optimizer=optimizer, metrics={"loss": 0.8})
        manager.save(epoch=3, model=model, optimizer=optimizer, metrics={"loss": 0.3})
        manager.save(epoch=4, model=model, optimizer=optimizer, metrics={"loss": 0.6})

        # Should keep best 2 (epochs 3 and 1 with losses 0.3 and 0.5)
        checkpoints = list(temp_dir.glob("checkpoint_*.pt"))
        assert len(checkpoints) == 2

    def test_checkpoint_history(self, temp_dir, model, optimizer):
        from expmate.torch.checkpoint import CheckpointManager

        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        # Save a few checkpoints
        for epoch in range(3):
            manager.save(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                metrics={"loss": 1.0 - epoch * 0.2},
            )

        # Check history file
        history_file = temp_dir / "checkpoint_history.json"
        assert history_file.exists()

        with open(history_file) as f:
            history = json.load(f)
            assert len(history) == 3

    def test_get_latest_checkpoint(self, temp_dir, model, optimizer):
        from expmate.torch.checkpoint import CheckpointManager

        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        # No checkpoints yet
        assert manager.get_latest_checkpoint() is None

        # Save checkpoints
        for epoch in range(3):
            manager.save(epoch=epoch, model=model, optimizer=optimizer)

        # Get latest
        latest = manager.get_latest_checkpoint()
        assert latest is not None
        assert "epoch_2" in str(latest)

    def test_get_best_checkpoint(self, temp_dir, model, optimizer):
        from expmate.torch.checkpoint import CheckpointManager

        manager = CheckpointManager(
            checkpoint_dir=str(temp_dir), metric_name="accuracy", mode="max"
        )

        # Save checkpoints with different accuracies
        manager.save(
            epoch=0, model=model, optimizer=optimizer, metrics={"accuracy": 0.5}
        )
        manager.save(
            epoch=1, model=model, optimizer=optimizer, metrics={"accuracy": 0.8}
        )
        manager.save(
            epoch=2, model=model, optimizer=optimizer, metrics={"accuracy": 0.6}
        )

        # Get best
        best = manager.get_best_checkpoint()
        assert best is not None

        # Load and verify it's epoch 1
        new_model = SimpleModel()
        metadata = manager.load(new_model, checkpoint_path=best)
        assert metadata["epoch"] == 1

    def test_resume_training(self, temp_dir, model, optimizer):
        from expmate.torch.checkpoint import CheckpointManager

        manager = CheckpointManager(checkpoint_dir=str(temp_dir))

        # Save checkpoint
        original_state = {k: v.clone() for k, v in model.state_dict().items()}
        manager.save(epoch=5, model=model, optimizer=optimizer)

        # Modify model
        with torch.no_grad():
            model.fc.weight.fill_(0)

        # Resume from checkpoint
        latest = manager.get_latest_checkpoint()
        manager.load(model, optimizer, checkpoint_path=latest)

        # Verify model was restored
        restored_state = model.state_dict()
        for key in original_state:
            assert torch.allclose(original_state[key], restored_state[key])

import json

import pytest

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="PyTorch required for integration tests"
)


class SimpleNet(nn.Module):
    """Simple neural network for testing."""

    def __init__(self, input_size=10, hidden_size=20, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)


@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace with config file."""
    # Create config file
    config_file = tmp_path / "config.yaml"
    config_content = """
project_name: integration_test
run_name: test_run

model:
  input_size: 10
  hidden_size: 20
  output_size: 2

training:
  epochs: 3
  batch_size: 4
  learning_rate: 0.001
  checkpoint_every: 1

logging:
  log_every: 1
  track_best: true
  best_metric: val/loss
  best_mode: min
"""
    config_file.write_text(config_content)

    return tmp_path, config_file


class TestFullTrainingWorkflow:
    """Test complete training workflow with all ExpMate features."""

    def test_config_loading_and_interpolation(self, temp_workspace):
        """Test config loading with interpolation."""
        from expmate import Config

        tmp_path, config_file = temp_workspace

        # Test basic loading
        config = Config(str(config_file))
        assert config.project_name == "integration_test"
        assert config.model.input_size == 10
        assert config.training.epochs == 3

        # Test dot notation access
        assert config.get("training.learning_rate") == 0.001

        # Test dict-style access
        assert config["model"]["hidden_size"] == 20

    def test_config_with_overrides(self, temp_workspace):
        """Test config loading with CLI-style overrides."""
        from expmate import Config

        tmp_path, config_file = temp_workspace

        # Test with overrides
        overrides = [
            "training.epochs=5",
            "training.learning_rate=0.01",
            "model.hidden_size=32",
        ]
        config = Config(str(config_file), overrides=overrides)

        assert config.training.epochs == 5
        assert config.training.learning_rate == 0.01
        assert config.model.hidden_size == 32
        # Original values should be overridden
        assert config.model.input_size == 10  # Not overridden

    def test_experiment_logger_lifecycle(self, temp_workspace):
        """Test experiment logger creation and usage."""
        from expmate.logger import ExperimentLogger

        tmp_path, config_file = temp_workspace
        runs_dir = tmp_path / "runs" / "test_run_001"

        # Create logger
        logger = ExperimentLogger(
            run_dir=str(runs_dir),
            rank=0,
        )

        # Test logging
        logger.log("Starting experiment", level="INFO")
        logger.log_metric(step=1, split="train", name="loss", value=0.5)
        logger.log_metric(step=1, split="val", name="loss", value=0.4)

        # Test best tracking
        logger.log_metric(
            step=2, split="val", name="loss", value=0.3, track_best=True, mode="min"
        )
        logger.log_metric(
            step=3, split="val", name="loss", value=0.35, track_best=True, mode="min"
        )

        # Check best metric was tracked
        assert "val/loss" in logger.best_metrics
        assert logger.best_metrics["val/loss"]["value"] == 0.3
        assert logger.best_metrics["val/loss"]["step"] == 2

        # Close logger
        logger.close()

        # Verify logger created files
        assert (logger.run_dir / "events.jsonl").exists()
        assert (logger.run_dir / "metrics.csv").exists()
        assert (logger.run_dir / "exp.log").exists()

        # Verify best metrics file
        with open(logger.run_dir / "best.json") as f:
            best = json.load(f)
            assert "val/loss" in best
            assert best["val/loss"]["value"] == 0.3

    def test_checkpoint_manager_workflow(self, temp_workspace):
        """Test checkpoint manager save/load cycle."""
        from expmate.torch.checkpoint import CheckpointManager

        tmp_path, _ = temp_workspace
        checkpoint_dir = tmp_path / "checkpoints"

        # Create model and optimizer
        model = SimpleNet()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Create checkpoint manager
        manager = CheckpointManager(
            checkpoint_dir=str(checkpoint_dir),
            keep_last=2,
            keep_best=2,
            metric_name="val_loss",
            mode="min",
        )

        # Save checkpoints
        manager.save(
            epoch=0, model=model, optimizer=optimizer, metrics={"val_loss": 1.0}
        )
        manager.save(
            epoch=1, model=model, optimizer=optimizer, metrics={"val_loss": 0.8}
        )
        manager.save(
            epoch=2, model=model, optimizer=optimizer, metrics={"val_loss": 0.9}
        )

        # Verify files exist
        assert checkpoint_dir.exists()
        checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
        assert len(checkpoints) >= 2  # At least 2 kept

        # Test loading latest
        latest = manager.get_latest_checkpoint()
        assert latest is not None
        assert "epoch_2" in str(latest)

        # Test loading best
        best = manager.get_best_checkpoint()
        assert best is not None

        # Test loading into new model
        new_model = SimpleNet()
        new_optimizer = optim.Adam(new_model.parameters(), lr=0.001)
        metadata = manager.load(new_model, new_optimizer, checkpoint_path=best)

        assert "epoch" in metadata
        assert "metrics" in metadata

        # Verify weights were loaded
        for p1, p2 in zip(model.parameters(), new_model.parameters()):
            assert torch.allclose(p1, p2)

    def test_full_training_loop_integration(self, temp_workspace):
        """Test complete training loop with all components."""
        from expmate import Config
        from expmate.logger import ExperimentLogger
        from expmate.torch.checkpoint import CheckpointManager
        from expmate.utils import set_seed

        tmp_path, config_file = temp_workspace

        # Load config
        config = Config(str(config_file))

        # Set seed for reproducibility
        set_seed(42)

        # Create logger
        logger = ExperimentLogger(
            run_dir=str(tmp_path / "runs" / "integration_test"),
            rank=0,
        )

        # Create checkpoint manager
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=str(logger.run_dir / "checkpoints"),
            keep_last=1,
            keep_best=2,
            metric_name="val_loss",
            mode="min",
        )

        # Create model and optimizer
        model = SimpleNet(
            input_size=config.model.input_size,
            hidden_size=config.model.hidden_size,
            output_size=config.model.output_size,
        )
        optimizer = optim.Adam(
            model.parameters(), lr=float(config.training.learning_rate)
        )
        criterion = nn.CrossEntropyLoss()

        # Create synthetic data
        train_data = [
            (torch.randn(config.model.input_size), torch.randint(0, 2, (1,)).item())
            for _ in range(10)
        ]
        val_data = [
            (torch.randn(config.model.input_size), torch.randint(0, 2, (1,)).item())
            for _ in range(5)
        ]

        logger.log("Starting training")

        # Training loop
        for epoch in range(config.training.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            for x, y in train_data:
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output.unsqueeze(0), torch.tensor([y]))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_data)

            # Validation phase
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in val_data:
                    output = model(x)
                    loss = criterion(output.unsqueeze(0), torch.tensor([y]))
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_data)

            # Log metrics
            logger.log_metric(
                step=epoch, split="train", name="loss", value=avg_train_loss
            )
            logger.log_metric(
                step=epoch,
                split="val",
                name="loss",
                value=avg_val_loss,
                track_best=True,
                mode="min",
            )

            logger.log(
                f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}"
            )

            # Save checkpoint
            checkpoint_manager.save(
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                metrics={"val_loss": avg_val_loss},
            )

        # Close logger
        logger.close()

        # Verify everything was created
        assert logger.run_dir.exists()
        assert (logger.run_dir / "events.jsonl").exists()
        assert (logger.run_dir / "exp.log").exists()
        assert (logger.run_dir / "best.json").exists()
        assert (logger.run_dir / "checkpoints").exists()

        # Verify metrics were logged
        with open(logger.run_dir / "metrics.csv") as f:
            lines = f.readlines()
            assert len(lines) > 1  # Header + at least one metric
            assert "step,split,name,value,wall_time" in lines[0]

        # Verify checkpoints were saved
        checkpoints = list((logger.run_dir / "checkpoints").glob("checkpoint_*.pt"))
        assert len(checkpoints) > 0

        # Test loading and resuming
        latest_checkpoint = checkpoint_manager.get_latest_checkpoint()
        assert latest_checkpoint is not None

        new_model = SimpleNet(
            input_size=config.model.input_size,
            hidden_size=config.model.hidden_size,
            output_size=config.model.output_size,
        )
        new_optimizer = optim.Adam(
            new_model.parameters(), lr=float(config.training.learning_rate)
        )

        metadata = checkpoint_manager.load(
            new_model, new_optimizer, checkpoint_path=latest_checkpoint
        )
        assert metadata["epoch"] == config.training.epochs - 1

    def test_config_save_and_hash(self, temp_workspace):
        """Test config saving and hash generation."""
        from expmate import Config

        tmp_path, config_file = temp_workspace

        config = Config(str(config_file))

        # Save config
        save_path = tmp_path / "saved_config.yaml"
        config.save(str(save_path))

        assert save_path.exists()

        # Load saved config
        loaded_config = Config(str(save_path))
        assert loaded_config.project_name == config.project_name
        assert loaded_config.training.epochs == config.training.epochs

        # Test hash generation
        hash1 = config.hash()
        hash2 = config.hash()
        assert hash1 == hash2  # Same config should have same hash

        # Different config should have different hash
        config2 = Config(str(config_file), overrides=["training.epochs=10"])
        hash3 = config2.hash()
        assert hash1 != hash3

    def test_logger_with_profiling(self, temp_workspace):
        """Test logger profiling context manager."""
        from expmate.logger import ExperimentLogger
        import time

        tmp_path, _ = temp_workspace

        logger = ExperimentLogger(run_dir=str(tmp_path / "runs" / "profile_test"))

        # Test profiling
        with logger.profile("data_loading"):
            time.sleep(0.01)  # Simulate work

        with logger.profile("training_step"):
            time.sleep(0.02)  # Simulate work

        logger.close()

        # Check that timing events were logged
        with open(logger.run_dir / "events.jsonl") as f:
            events = [json.loads(line) for line in f]
            # Profile events have 'section' and 'elapsed' fields
            profile_events = [e for e in events if "section" in e and "elapsed" in e]
            assert len(profile_events) >= 2

    def test_checkpoint_cleanup(self, temp_workspace):
        """Test checkpoint manager cleanup functionality."""
        from expmate.torch.checkpoint import CheckpointManager

        tmp_path, _ = temp_workspace
        checkpoint_dir = tmp_path / "checkpoints"

        model = SimpleNet()
        optimizer = optim.Adam(model.parameters())

        # Test keep_last
        manager = CheckpointManager(
            checkpoint_dir=str(checkpoint_dir), keep_last=2, keep_best=None
        )

        # Save 5 checkpoints
        for i in range(5):
            manager.save(epoch=i, model=model, optimizer=optimizer)

        # Should only keep last 2
        checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
        assert len(checkpoints) == 2

        # Verify it's the last 2
        epochs = [int(cp.stem.split("_")[-1]) for cp in checkpoints]
        assert sorted(epochs) == [3, 4]


class TestMultiFileConfig:
    """Test configuration with multiple files and merging."""

    def test_config_merging(self, tmp_path):
        """Test loading and merging multiple config files."""
        from expmate import Config

        # Create base config
        base_config = tmp_path / "base.yaml"
        base_config.write_text(
            """
model:
  type: transformer
  hidden_size: 256
  num_layers: 6

training:
  epochs: 100
  batch_size: 32
"""
        )

        # Create experiment config
        exp_config = tmp_path / "experiment.yaml"
        exp_config.write_text(
            """
model:
  hidden_size: 512
  dropout: 0.1

training:
  epochs: 50
  learning_rate: 0.001
"""
        )

        # Load with multiple files
        config = Config([str(base_config), str(exp_config)])

        # Check merged values
        assert config.model.type == "transformer"  # From base
        assert config.model.hidden_size == 512  # Overridden by exp
        assert config.model.num_layers == 6  # From base
        assert config.model.dropout == 0.1  # From exp
        assert config.training.epochs == 50  # Overridden by exp
        assert config.training.batch_size == 32  # From base
        assert config.training.learning_rate == 0.001  # From exp


class TestErrorRecovery:
    """Test error handling and recovery scenarios."""

    def test_invalid_config_file(self, tmp_path):
        """Test handling of invalid config files."""
        from expmate import Config

        invalid_config = tmp_path / "invalid.yaml"
        invalid_config.write_text("invalid: yaml: content:")

        with pytest.raises(ValueError, match="Invalid YAML"):
            Config(str(invalid_config))

    def test_missing_config_file(self, tmp_path):
        """Test handling of missing config files."""
        from expmate import Config

        with pytest.raises(FileNotFoundError):
            Config(str(tmp_path / "nonexistent.yaml"))

    def test_checkpoint_load_nonexistent(self, tmp_path):
        """Test loading non-existent checkpoint."""
        from expmate.torch.checkpoint import CheckpointManager

        manager = CheckpointManager(checkpoint_dir=str(tmp_path / "checkpoints"))

        # Should return None for non-existent checkpoints
        assert manager.get_latest_checkpoint() is None
        assert manager.get_best_checkpoint() is None

"""Pytest configuration and shared fixtures."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Generator

import pytest


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for tests."""
    tmpdir = Path(tempfile.mkdtemp())
    yield tmpdir
    # Cleanup
    if tmpdir.exists():
        shutil.rmtree(tmpdir)


@pytest.fixture
def config_file(temp_dir: Path) -> Path:
    """Create a basic config file for testing."""
    config_path = temp_dir / "config.yaml"
    config_content = """
model:
  name: test_model
  hidden_dim: 128
  dropout: 0.1

training:
  lr: 0.001
  epochs: 10
  batch_size: 32

data:
  path: /data/test
  num_workers: 4

seed: 42
"""
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def advanced_config_file(temp_dir: Path) -> Path:
    """Create a config file with interpolation for testing."""
    config_path = temp_dir / "advanced_config.yaml"
    config_content = """
project:
  name: test_project
  version: "1.0"

paths:
  data_dir: /data
  output_dir: ${paths.data_dir}/output
  checkpoint_dir: ${paths.output_dir}/checkpoints

model:
  name: ${project.name}_model
  hidden_dim: 128

training:
  lr: 0.001
  epochs: 10
  save_dir: ${paths.checkpoint_dir}

experiment:
  timestamp: ${now:%Y%m%d_%H%M%S}
  hostname: ${hostname}
"""
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def sample_metrics_data():
    """Sample metrics data for testing visualization."""
    return [
        {"step": 0, "split": "train", "name": "loss", "value": 1.5},
        {"step": 0, "split": "val", "name": "loss", "value": 1.6},
        {"step": 1, "split": "train", "name": "loss", "value": 1.2},
        {"step": 1, "split": "val", "name": "loss", "value": 1.3},
        {"step": 0, "split": "train", "name": "accuracy", "value": 0.5},
        {"step": 0, "split": "val", "name": "accuracy", "value": 0.48},
        {"step": 1, "split": "train", "name": "accuracy", "value": 0.6},
        {"step": 1, "split": "val", "name": "accuracy", "value": 0.55},
    ]


@pytest.fixture
def mock_git_repo(temp_dir: Path) -> Path:
    """Create a mock git repository for testing."""
    import subprocess

    repo_dir = temp_dir / "git_repo"
    repo_dir.mkdir()

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=repo_dir, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=repo_dir,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"], cwd=repo_dir, capture_output=True
    )

    # Create a test file and commit
    test_file = repo_dir / "test.txt"
    test_file.write_text("test content")
    subprocess.run(["git", "add", "."], cwd=repo_dir, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"], cwd=repo_dir, capture_output=True
    )

    return repo_dir


@pytest.fixture(autouse=True)
def reset_env_vars():
    """Reset environment variables after each test."""
    original_env = os.environ.copy()
    yield
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)

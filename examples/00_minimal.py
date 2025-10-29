#!/usr/bin/env python3
"""Minimal ExpMate example - Quick start guide.

This example shows the simplest way to use ExpMate for experiment tracking.

Run with:
    python examples/00_minimal.py
    python examples/00_minimal.py conf/example.yaml
    python examples/00_minimal.py conf/example.yaml +training.lr=0.01
"""

import argparse
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from expmate import Config, ExperimentLogger, set_seed


@dataclass
class ModelConfig(Config):
    """Model configuration"""

    input_dim: int = 10
    output_dim: int = 3


@dataclass
class TrainingConfig(Config):
    """Training configuration"""

    epochs: int = 5
    lr: float = 0.001


@dataclass
class ExperimentConfig(Config):
    """Main experiment configuration"""

    run_id: str = "minimal_${now:%Y%m%d_%H%M%S}"
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Minimal ExpMate example")
    parser.add_argument(
        "--config", default="examples/conf/minimal.yaml", help="Config file path"
    )
    args, overrides = parser.parse_known_args()

    # Load config with optional overrides
    config = ExperimentConfig.from_file(args.config, overrides=overrides)

    # Set random seed for reproducibility
    set_seed(config.seed)

    # Create experiment logger
    logger = ExperimentLogger(run_dir=f"runs/{config.run_id}")
    logger.info(f"Starting experiment: {config.run_id}")
    logger.info(f"Config: seed={config.seed}, lr={config.training.lr}")

    # Create dummy model and data
    model = nn.Linear(config.model.input_dim, config.model.output_dim)
    dataset = TensorDataset(
        torch.randn(100, config.model.input_dim),
        torch.randint(0, config.model.output_dim, (100,)),
    )
    dataloader = DataLoader(dataset, batch_size=32)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(config.training.epochs):
        total_loss = 0

        for batch_x, batch_y in dataloader:
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        # Log metrics
        logger.log_metric(step=epoch, split="train", name="loss", value=avg_loss)
        logger.info(f"Epoch {epoch}/{config.training.epochs}: loss={avg_loss:.4f}")

        time.sleep(0.1)  # Simulate training time

    logger.info("Training complete!")
    logger.info(f"Logs saved to: {logger.run_dir}")


if __name__ == "__main__":
    main()

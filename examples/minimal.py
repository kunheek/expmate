#!/usr/bin/env python3
"""Minimal ExpMate example - Quick start guide.

This example shows the simplest way to use ExpMate for experiment tracking.

Run with:
    python examples/minimal.py conf/example.yaml
    python examples/minimal.py conf/example.yaml training.lr=0.01
"""

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from expmate import ExperimentLogger, parse_config, set_seed


def main():
    # Parse config from command line (first argument + overrides)
    config = parse_config()

    # Set random seed for reproducibility
    set_seed(int(config.seed))

    # Create experiment logger
    logger = ExperimentLogger(run_dir=f"runs/{config.run_id}")
    logger.info(f"Starting experiment: {config.run_id}")
    logger.info(f"Config: seed={config.seed}, lr={config.training.lr}")

    # Create dummy model and data
    model = nn.Linear(int(config.model.input_dim), int(config.model.output_dim))
    dataset = TensorDataset(
        torch.randn(100, int(config.model.input_dim)),
        torch.randint(0, int(config.model.output_dim), (100,)),
    )
    dataloader = DataLoader(dataset, batch_size=32)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.training.lr))
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(int(config.training.epochs)):
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

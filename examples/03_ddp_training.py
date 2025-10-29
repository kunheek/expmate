#!/usr/bin/env python3
"""
Distributed Training with PyTorch DDP
======================================

This example demonstrates:
- Multi-GPU distributed training with DDP
- Rank-aware logging
- DDP-safe run directory creation
- Metrics aggregation across processes
- Checkpoint management in DDP

Run with torchrun:
    torchrun --nproc_per_node=2 examples/03_ddp_training.py examples/conf/ddp.yaml
    torchrun --nproc_per_node=4 examples/03_ddp_training.py examples/conf/ddp.yaml +model.hidden_dim=256

For single GPU/CPU:
    python examples/03_ddp_training.py examples/conf/ddp.yaml
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from expmate import Config, ExperimentLogger, set_seed
from expmate.torch import CheckpointManager, mp


# Configuration dataclasses
@dataclass
class DataConfig(Config):
    """Data configuration"""

    n_samples: int = 10000
    input_dim: int = 100
    output_dim: int = 10
    batch_size: int = 64


@dataclass
class ModelConfig(Config):
    """Model configuration"""

    hidden_dim: int = 128


@dataclass
class TrainingConfig(Config):
    """Training configuration"""

    epochs: int = 5
    lr: float = 0.001


@dataclass
class CheckpointConfig(Config):
    """Checkpoint configuration"""

    keep_last: int = 2
    keep_best: int = 3


@dataclass
class ExperimentConfig(Config):
    """Main experiment configuration"""

    run_id: str = "ddp_${now:%Y%m%d_%H%M%S}"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


# Simple model
class SimpleModel(nn.Module):
    """Simple feedforward model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def create_dataset(n_samples, input_dim, output_dim, seed=42):
    """Create dummy dataset"""
    torch.manual_seed(seed)
    X = torch.randn(n_samples, input_dim)
    y = torch.randint(0, output_dim, (n_samples,))
    return TensorDataset(X, y)


def train_epoch(model, loader, optimizer, criterion, device, rank, world_size):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        total_correct += pred.eq(target).sum().item()
        total_samples += data.size(0)

    # Aggregate metrics across all processes
    if world_size > 1:
        metrics = torch.tensor(
            [total_loss, total_correct, total_samples], device=device
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss, total_correct, total_samples = metrics.tolist()

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, loader, criterion, device, world_size):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for data, target in loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)

        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        total_correct += pred.eq(target).sum().item()
        total_samples += data.size(0)

    # Aggregate metrics
    if world_size > 1:
        metrics = torch.tensor(
            [total_loss, total_correct, total_samples], device=device
        )
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        total_loss, total_correct, total_samples = metrics.tolist()

    avg_loss = total_loss / total_samples
    accuracy = 100.0 * total_correct / total_samples
    return avg_loss, accuracy


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Distributed training with ExpMate")
    parser.add_argument(
        "--config", default="examples/conf/ddp.yaml", help="Config file path"
    )
    args, overrides = parser.parse_known_args()

    # Load config with optional overrides
    config = ExperimentConfig.from_file(args.config, overrides=overrides)

    # Setup DDP
    rank, local_rank, world_size = mp.setup_ddp()
    is_main = rank == 0

    # Create shared run directory (DDP-safe)
    run_dir = mp.create_shared_run_dir(base_dir="runs", run_id=config.run_id)

    # Setup logging (rank-aware)
    logger = ExperimentLogger(
        run_dir=run_dir,
        rank=rank,
        log_level="INFO",
        console_output=is_main,  # Only main process prints to console
    )

    if is_main:
        logger.info(f"🚀 Starting DDP training: {config.run_id}")
        logger.info(f"World size: {world_size}")
        logger.info(f"Configuration:\n{config.to_yaml()}")

    # Set seed (with rank offset for different data loading)
    set_seed(config.seed + rank)

    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    logger.info(
        f"Process initialized: rank={rank}, local_rank={local_rank}, device={device}"
    )

    # Create datasets
    with logger.stage("data_preparation", rank=rank):
        train_dataset = create_dataset(
            config.data.n_samples,
            config.data.input_dim,
            config.data.output_dim,
            seed=config.seed,
        )
        val_dataset = create_dataset(
            config.data.n_samples // 5,
            config.data.input_dim,
            config.data.output_dim,
            seed=config.seed + 1000,
        )

        # Create distributed samplers
        train_sampler = (
            DistributedSampler(
                train_dataset, num_replicas=world_size, rank=rank, shuffle=True
            )
            if world_size > 1
            else None
        )

        val_sampler = (
            DistributedSampler(
                val_dataset, num_replicas=world_size, rank=rank, shuffle=False
            )
            if world_size > 1
            else None
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.data.batch_size,
            sampler=val_sampler,
            shuffle=False,
        )

        if is_main:
            logger.info(f"Train samples: {len(train_dataset)}")
            logger.info(f"Val samples: {len(val_dataset)}")
            logger.info(f"Samples per GPU: {len(train_dataset) // world_size}")

    # Create model
    with logger.stage("model_initialization", rank=rank):
        model = SimpleModel(
            input_dim=config.data.input_dim,
            hidden_dim=config.model.hidden_dim,
            output_dim=config.data.output_dim,
        ).to(device)

        # Wrap with DDP
        if world_size > 1:
            model = DDP(
                model, device_ids=[local_rank] if torch.cuda.is_available() else None
            )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)

        if is_main:
            total_params = sum(p.numel() for p in model.parameters())
            logger.info(f"Model: SimpleModel with {total_params:,} parameters")
            logger.info(f"Optimizer: Adam(lr={config.training.lr})")

    # Setup checkpoint manager (only on main process)
    checkpoint_manager = None
    if is_main:
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=Path(run_dir) / "checkpoints",
            keep_last=config.checkpoint.keep_last,
            keep_best=config.checkpoint.keep_best,
            metric_name="val_loss",
            mode="min",
        )

    # Training loop
    if is_main:
        logger.info(f"\n{'=' * 60}")
        logger.info(
            f"Training for {config.training.epochs} epochs on {world_size} GPU(s)"
        )
        logger.info(f"{'=' * 60}\n")

    for epoch in range(config.training.epochs):
        with logger.stage("epoch", epoch=epoch, rank=rank):
            # Set epoch for sampler (for proper shuffling)
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            # Train
            with logger.stage("train", epoch=epoch, rank=rank):
                train_loss, train_acc = train_epoch(
                    model, train_loader, optimizer, criterion, device, rank, world_size
                )

            # Validate
            with logger.stage("validation", epoch=epoch, rank=rank):
                val_loss, val_acc = validate(
                    model, val_loader, criterion, device, world_size
                )

            # Log metrics (only main process)
            if is_main:
                logger.log_metric(
                    step=epoch, split="train", name="loss", value=train_loss
                )
                logger.log_metric(
                    step=epoch, split="train", name="accuracy", value=train_acc
                )
                logger.log_metric(step=epoch, split="val", name="loss", value=val_loss)
                logger.log_metric(
                    step=epoch, split="val", name="accuracy", value=val_acc
                )

                logger.info(
                    f"Epoch {epoch:2d}/{config.training.epochs}: "
                    f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, "
                    f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%"
                )

                # Save checkpoint
                if checkpoint_manager:
                    # Get the actual model (unwrap DDP if needed)
                    model_to_save = model.module if isinstance(model, DDP) else model
                    checkpoint_manager.save(
                        model=model_to_save,
                        optimizer=optimizer,
                        epoch=epoch,
                        metrics={"val_loss": val_loss, "val_acc": val_acc},
                    )

    # Cleanup
    if world_size > 1:
        dist.barrier()  # Ensure all processes finish
        mp.cleanup_ddp()

    if is_main:
        best_val = logger.get_best_metric("loss", split="val")
        logger.info(
            f"\n✨ Best validation loss: {best_val['value']:.4f} at epoch {best_val['step']}"
        )
        logger.info(f"📁 Results saved to: {run_dir}")
        logger.info("✅ Training completed!")


if __name__ == "__main__":
    main()

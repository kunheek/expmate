#!/usr/bin/env python3
"""
MNIST Classification with PyTorch
===================================

This example demonstrates:
- Typed configuration with dataclasses for LSP support
- PyTorch neural network training
- Checkpoint management with automatic cleanup
- Experiment tracking (WandB/TensorBoard optional)
- Learning rate scheduling
- Early stopping based on best metrics
- Hierarchical stage tracking for train/val loops

Run with:
    python 02_mnist_classification.py examples/conf/mnist.yaml
    python 02_mnist_classification.py examples/conf/mnist.yaml +model.hidden_dim=256 +training.lr=0.001

Optional tracking:
    python 02_mnist_classification.py examples/conf/mnist.yaml +tracking.wandb=true +tracking.project=mnist-exp
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from expmate import Config, ExperimentLogger, set_seed

# Try to import checkpoint manager (optional)
try:
    from expmate.torch import CheckpointManager

    CHECKPOINT_AVAILABLE = True
except ImportError:
    CHECKPOINT_AVAILABLE = False


# Configuration dataclasses
@dataclass
class DataConfig(Config):
    """Data configuration"""

    batch_size: int = 64
    val_split: float = 0.1
    num_workers: int = 4


@dataclass
class ModelConfig(Config):
    """Model configuration"""

    hidden_dim: int = 128
    dropout: float = 0.5


@dataclass
class TrainingConfig(Config):
    """Training configuration"""

    epochs: int = 10
    lr: float = 0.001
    weight_decay: float = 1e-4
    early_stop_patience: int = 5


@dataclass
class CheckpointConfig(Config):
    """Checkpoint configuration"""

    keep_last: int = 3
    keep_best: int = 5


@dataclass
class TrackingConfig(Config):
    """Tracking configuration"""

    wandb: bool = False
    tensorboard: bool = False
    project: str = "mnist-experiments"


@dataclass
class ExperimentConfig(Config):
    """Main experiment configuration"""

    run_id: str = "mnist_${now:%Y%m%d_%H%M%S}"
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)


# Simple CNN model
class SimpleCNN(nn.Module):
    """Simple CNN for MNIST classification"""

    def __init__(self, hidden_dim=128, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 10)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


def train_epoch(model, loader, optimizer, criterion, device, logger, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    with logger.stage("train", epoch=epoch):
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

            # Log batch progress (rate limited)
            if batch_idx % 50 == 0:
                with logger.log_every(seconds=5.0, key="train_batch"):
                    logger.info(
                        f"  Batch {batch_idx}/{len(loader)}: "
                        f"loss={loss.item():.4f}, acc={100.0 * correct / total:.2f}%"
                    )

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, loader, criterion, device, logger, epoch):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with logger.stage("validation", epoch=epoch):
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MNIST classification with ExpMate")
    parser.add_argument(
        "--config", default="examples/conf/mnist.yaml", help="Config file path"
    )
    args, overrides = parser.parse_known_args()

    # Load config with optional overrides
    config = ExperimentConfig.from_file(args.config, overrides=overrides)

    # Set device and seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config.seed)

    # Create logger
    logger = ExperimentLogger(run_dir=f"runs/{config.run_id}")
    logger.info(f"🚀 Starting MNIST training: {config.run_id}")
    logger.info(f"Device: {device}")
    logger.info(f"Configuration:\n{config.to_yaml()}")

    # Prepare data
    with logger.stage("data_preparation"):
        with logger.timer("load_dataset"):
            transform = transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            )

            train_dataset = datasets.MNIST(
                "data", train=True, download=True, transform=transform
            )
            test_dataset = datasets.MNIST("data", train=False, transform=transform)

            # Split train into train/val
            val_size = int(config.data.val_split * len(train_dataset))
            train_size = len(train_dataset) - val_size
            train_dataset, val_dataset = random_split(
                train_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(config.seed),
            )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.data.batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.data.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        )

        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")
        logger.info(f"Test samples: {len(test_dataset)}")

    # Create model
    with logger.stage("model_initialization"):
        model = SimpleCNN(
            hidden_dim=config.model.hidden_dim, dropout=config.model.dropout
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.training.lr,
            weight_decay=config.training.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, verbose=True
        )

        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model: SimpleCNN with {total_params:,} parameters")
        logger.info(
            f"Optimizer: Adam(lr={config.training.lr}, wd={config.training.weight_decay})"
        )

    # Setup checkpoint manager
    checkpoint_manager = None
    if CHECKPOINT_AVAILABLE:
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=Path(logger.run_dir) / "checkpoints",
            keep_last=config.checkpoint.keep_last,
            keep_best=config.checkpoint.keep_best,
            metric_name="val_loss",
            mode="min",
        )
        logger.info(
            f"Checkpoint manager: keep_last={config.checkpoint.keep_last}, keep_best={config.checkpoint.keep_best}"
        )

    # Training loop
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Training for {config.training.epochs} epochs")
    logger.info(f"{'=' * 60}\n")

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(config.training.epochs):
        with logger.stage("epoch", epoch=epoch):
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, criterion, device, logger, epoch
            )
            logger.log_metric(step=epoch, split="train", name="loss", value=train_loss)
            logger.log_metric(
                step=epoch, split="train", name="accuracy", value=train_acc
            )

            # Validate
            val_loss, val_acc = validate(
                model, val_loader, criterion, device, logger, epoch
            )
            logger.log_metric(step=epoch, split="val", name="loss", value=val_loss)
            logger.log_metric(step=epoch, split="val", name="accuracy", value=val_acc)

            # Learning rate scheduling
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]
            logger.log_metric(
                step=epoch, split="train", name="learning_rate", value=current_lr
            )

            # Log epoch summary
            logger.info(
                f"Epoch {epoch:2d}/{config.training.epochs}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.2f}%, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%, "
                f"lr={current_lr:.6f}"
            )

            # Save checkpoint
            if checkpoint_manager:
                checkpoint_manager.save(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics={"val_loss": val_loss, "val_acc": val_acc},
                )

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= config.training.early_stop_patience:
                    logger.warning(f"Early stopping triggered after {epoch + 1} epochs")
                    break

    # Final evaluation on test set
    logger.info(f"\n{'=' * 60}")
    logger.info("Final Evaluation on Test Set")
    logger.info(f"{'=' * 60}\n")

    with logger.stage("test_evaluation"):
        test_loss, test_acc = validate(
            model, test_loader, criterion, device, logger, epoch=-1
        )
        logger.log_metric(
            step=config.training.epochs, split="test", name="loss", value=test_loss
        )
        logger.log_metric(
            step=config.training.epochs, split="test", name="accuracy", value=test_acc
        )

        logger.info(f"Test Loss: {test_loss:.4f}")
        logger.info(f"Test Accuracy: {test_acc:.2f}%")

    # Print best metrics
    best_val = logger.get_best_metric("loss", split="val")
    logger.info(
        f"\n✨ Best validation loss: {best_val['value']:.4f} at epoch {best_val['step']}"
    )

    logger.info(f"\n📁 Results saved to: {logger.run_dir}")
    logger.info("✅ Training completed!")


if __name__ == "__main__":
    main()

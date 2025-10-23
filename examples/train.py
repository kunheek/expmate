#!/usr/bin/env python3
"""Complete DDP training example with torchrun.

This example demonstrates:
- Distributed training setup with torchrun
- DDP-safe run directory creation
- Rank-aware logging
- Model checkpointing
- Metrics aggregation across processes
- Git integration for reproducibility

Run with:
    torchrun --nproc_per_node=2 examples/train.py conf/default.yaml
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

from expmate import ExperimentLogger, parse_config, set_seed
from expmate.torch.checkpoint import CheckpointManager
from expmate.git import get_git_info, save_git_info
from expmate.torch import mp


class SimpleModel(nn.Module):
    """Simple model for demonstration."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
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


def create_dummy_dataset(num_samples: int, input_dim: int, output_dim: int):
    """Create a dummy dataset for demonstration."""
    X = torch.randn(num_samples, input_dim)
    y = torch.randint(0, output_dim, (num_samples,))
    return TensorDataset(X, y)


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, logger):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    for batch_idx, (data, target) in enumerate(dataloader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1)
        total_correct += pred.eq(target).sum().item()
        total_samples += data.size(0)

    # Calculate average metrics
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    # Aggregate metrics across all processes
    if dist.is_initialized():
        # Convert to tensors for all_reduce
        metrics = torch.tensor([avg_loss, avg_acc, total_samples], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

        # Calculate weighted average
        world_size = dist.get_world_size()
        total_samples_all = metrics[2].item()
        avg_loss = metrics[0].item() / world_size
        avg_acc = metrics[1].item() / world_size

    return avg_loss, avg_acc


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            total_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1)
            total_correct += pred.eq(target).sum().item()
            total_samples += data.size(0)

    # Calculate average metrics
    avg_loss = total_loss / total_samples
    avg_acc = total_correct / total_samples

    # Aggregate metrics across all processes
    if dist.is_initialized():
        metrics = torch.tensor([avg_loss, avg_acc, total_samples], device=device)
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)

        world_size = dist.get_world_size()
        avg_loss = metrics[0].item() / world_size
        avg_acc = metrics[1].item() / world_size

    return avg_loss, avg_acc


def main():
    """Main training function."""
    # Setup DDP using expmate utility
    rank, local_rank, world_size = mp.setup_ddp()
    is_main = rank == 0

    # Parse configuration
    config = parse_config()

    # Print info from main process only
    if is_main:
        print(f"\n{'=' * 60}")
        print(f"PyTorch DDP Training with torchrun")
        print(f"{'=' * 60}")
        print(f"World Size: {world_size}")
        print(f"Backend: {'nccl' if torch.cuda.is_available() else 'gloo'}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        print(f"{'=' * 60}\n")

    # Create shared run directory (DDP-safe!)
    run_dir = mp.create_shared_run_dir(base_dir="runs", run_id=config.get("run_id"))

    # Setup logging (rank-aware)
    logger = ExperimentLogger(
        run_dir=run_dir,
        rank=rank,
        run_id=config["run_id"],
        log_level="INFO",
        console_output=is_main,
    )

    # Save git info for reproducibility (only rank 0)
    if is_main:
        git_info = get_git_info()
        save_git_info(run_dir, include_diff=True)
        logger.info(
            "Git info saved",
            sha=git_info["sha_short"],
            branch=git_info["branch"],
            dirty=git_info["dirty"],
        )

    # Set seed (with rank offset for different data loading)
    set_seed(int(config.seed) + rank)

    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    logger.info(
        "Process initialized",
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=str(device),
    )

    # Create model
    model = SimpleModel(
        input_dim=int(config.model.input_dim),
        hidden_dim=int(config.model.hidden_dim),
        output_dim=int(config.model.output_dim),
    ).to(device)

    # Wrap model with DDP
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
        )

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.training.lr))
    criterion = nn.CrossEntropyLoss()

    # Create datasets
    train_dataset = create_dummy_dataset(
        num_samples=int(config.data.num_train_samples),
        input_dim=int(config.model.input_dim),
        output_dim=int(config.model.output_dim),
    )
    val_dataset = create_dummy_dataset(
        num_samples=int(config.data.num_val_samples),
        input_dim=int(config.model.input_dim),
        output_dim=int(config.model.output_dim),
    )

    # Create data loaders with DistributedSampler
    train_sampler = (
        DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config.data.batch_size),  # Convert DotAccessor to int
        sampler=train_sampler,
        shuffle=(train_sampler is None),
    )

    val_sampler = (
        DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config.data.batch_size),  # Convert DotAccessor to int
        sampler=val_sampler,
        shuffle=False,
    )

    # Setup checkpoint manager (only rank 0 saves)
    ckpt_manager = CheckpointManager(
        checkpoint_dir=run_dir / "checkpoints",
        keep_last=3,
        keep_best=2,
    )

    logger.info("Starting training", epochs=int(config.training.epochs))

    # Training loop
    for epoch in range(int(config.training.epochs)):
        # Set epoch for sampler (ensures different shuffling each epoch)
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, logger
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Log metrics (only rank 0 writes to files)
        if is_main:
            logger.log_metric(step=epoch, split="train", name="loss", value=train_loss)
            logger.log_metric(
                step=epoch, split="train", name="accuracy", value=train_acc
            )
            logger.log_metric(step=epoch, split="val", name="loss", value=val_loss)
            logger.log_metric(step=epoch, split="val", name="accuracy", value=val_acc)

            print(
                f"Epoch {epoch:2d}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
            )

            # Save checkpoint (only rank 0)
            # Unwrap DDP model if necessary
            model_to_save = model.module if world_size > 1 else model
            ckpt_manager.save(
                model=model_to_save,
                optimizer=optimizer,
                epoch=epoch,
                metrics={
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                extra={"config": dict(config)},
            )

        # Synchronize all processes
        if world_size > 1:
            dist.barrier()

    # Final summary (only rank 0)
    if is_main:
        print(f"\n{'=' * 60}")
        print("Training completed!")
        print(f"{'=' * 60}")
        print(f"Results saved to: {run_dir}")
        print(f"Best checkpoint: {ckpt_manager.get_best_checkpoint()}")
        print(f"{'=' * 60}\n")

    # Cleanup using expmate utility
    mp.cleanup_ddp()


if __name__ == "__main__":
    main()

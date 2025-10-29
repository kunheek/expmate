#!/usr/bin/env python3
"""
Distributed Training with Progress Bars (mp_tqdm)
===================================================

This example demonstrates how to use mp_tqdm for clean progress bars
in distributed training. Only local_rank 0 on each node shows the progress bar,
avoiding cluttered output.

Features:
- Progress bars only on local_rank 0
- Works seamlessly with DDP
- Falls back gracefully if tqdm not installed
- Manual control via disable parameter

Run with:
    # Single GPU
    python examples/example_mp_tqdm.py

    # Multi-GPU (using torchrun)
    torchrun --nproc_per_node=4 examples/example_mp_tqdm.py
"""

import argparse
import time
from dataclasses import dataclass, field

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. This example requires PyTorch.")
    exit(1)

from expmate import Config, ExperimentLogger, set_seed
from expmate.torch import (
    get_local_rank,
    get_rank,
    mp_print,
    mp_tqdm,
    setup_ddp,
)


@dataclass
class TrainingConfig(Config):
    """Training configuration"""

    epochs: int = 5
    batch_size: int = 32
    num_batches: int = 100


@dataclass
class ExperimentConfig(Config):
    """Main experiment configuration"""

    run_id: str = "mp_tqdm_demo_${now:%Y%m%d_%H%M%S}"
    seed: int = 42
    training: TrainingConfig = field(default_factory=TrainingConfig)


def create_dummy_dataloader(batch_size: int, num_batches: int):
    """Create a dummy dataloader for demonstration."""
    # Create random data
    data = torch.randn(num_batches * batch_size, 10)
    targets = torch.randint(0, 2, (num_batches * batch_size,))
    dataset = TensorDataset(data, targets)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    rank: int,
    device: torch.device,
):
    """Train for one epoch with progress bar."""
    model.train()
    total_loss = 0.0

    # Progress bar only shows on local_rank 0
    for batch_idx, (data, target) in enumerate(
        mp_tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    ):
        # Move data to device
        data = data.to(device)
        target = target.to(device)

        # Simulate computation time
        time.sleep(0.01)

        optimizer.zero_grad()
        output = model(data)
        loss = nn.functional.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="mp_tqdm demonstration")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batches", type=int, default=100, help="Batches per epoch")
    args = parser.parse_args()

    # Setup DDP
    rank, local_rank, world_size = setup_ddp()

    # Create config
    config = ExperimentConfig(
        training=TrainingConfig(
            epochs=args.epochs,
            num_batches=args.batches,
        )
    )

    # Set seed
    set_seed(config.seed + rank)

    # Only print from rank 0
    mp_print(f"\n{'=' * 60}")
    mp_print(f"Distributed Training with mp_tqdm")
    mp_print(f"{'=' * 60}")
    mp_print(f"World size: {world_size}")
    mp_print(f"Local rank on this node: {local_rank}")
    mp_print(f"Configuration: {config.to_dict()}")
    mp_print(f"{'=' * 60}\n")

    # Create model and move to device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 2),
    ).to(device)

    # Wrap with DDP if multi-GPU
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP

        model = DDP(
            model,
            device_ids=[local_rank] if torch.cuda.is_available() else None,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Create dataloader
    dataloader = create_dummy_dataloader(
        config.training.batch_size,
        config.training.num_batches,
    )

    mp_print("Starting training...")
    mp_print("Note: Progress bars only show on local_rank 0\n")

    # Training loop with epoch-level progress bar
    for epoch in mp_tqdm(range(config.training.epochs), desc="Training"):
        avg_loss = train_epoch(model, dataloader, optimizer, epoch, rank, device)

        # Print only from rank 0
        mp_print(f"Rank {rank} | Epoch {epoch}: avg_loss={avg_loss:.4f}")

    mp_print("\n" + "=" * 60)
    mp_print("Training complete!")
    mp_print("=" * 60)

    # Cleanup
    if world_size > 1:
        from expmate.torch import cleanup_ddp

        cleanup_ddp()


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("This example requires PyTorch. Install with: pip install torch")
        exit(1)

    main()

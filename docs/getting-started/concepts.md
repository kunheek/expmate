# Basic Concepts

This guide explains the core concepts of ExpMate.

## Configuration Management

ExpMate uses a hierarchical configuration system based on YAML files with command-line overrides.

### Config Object

The `Config` object provides dictionary-like and attribute-like access:

```python
from expmate import Config

config = Config({
    'model': {'hidden_dim': 256},
    'training': {'lr': 0.001}
})

# Dictionary-style access
print(config['model']['hidden_dim'])  # 256

# Attribute-style access (preferred)
print(config.model.hidden_dim)  # 256
print(config.training.lr)  # 0.001
```

### Command-Line Overrides

Override any config value from the command line:

```bash
# Simple override
python train.py config.yaml training.lr=0.01

# Nested values
python train.py config.yaml model.hidden_dim=512

# Multiple overrides
python train.py config.yaml training.lr=0.01 training.epochs=50

# Add new keys with + prefix
python train.py config.yaml +optimizer.weight_decay=0.0001

# Type hints for ambiguous values
python train.py config.yaml training.lr:float=1e-3
```

### Variable Interpolation

Use variables in your config files:

```yaml
run_id: "exp_${now:%Y%m%d_%H%M%S}"  # Current timestamp
seed: 42
output_dir: "outputs/${run_id}"     # Reference other values
```

## Experiment Logging

ExpMate provides structured logging for both text and metrics.

### Text Logging

Use level-specific methods (preferred):

```python
logger.info("Training started")
logger.warning("Learning rate might be too high")
logger.error("NaN detected in loss")
logger.debug("Batch processing time: 0.5s")
```

Or the generic method:

```python
logger.log("Training started", level="INFO")
```

### Metrics Logging

Log structured metrics for tracking and visualization:

```python
logger.log_metric(
    step=epoch,
    split='train',  # 'train', 'val', 'test', etc.
    name='loss',
    value=0.5
)
```

All metrics are automatically saved to `metrics.csv` for easy analysis.

### Best Metrics Tracking

Track best values for important metrics:

```python
# Configure tracking (mode: 'min' or 'max')
logger.track_best('val_loss', mode='min')
logger.track_best('val_accuracy', mode='max')

# Log metrics normally
logger.log_metric(step=epoch, split='val', name='loss', value=0.5)

# Best values are automatically tracked and saved to best.json
```

## Distributed Training

ExpMate provides utilities for PyTorch distributed training.

### Setup DDP

```python
from expmate.torch import mp

# Initialize DDP
rank, local_rank, world_size = mp.setup_ddp()

# Check if main process
is_main = rank == 0
```

### Rank-Aware Logging

```python
from expmate import ExperimentLogger

# Pass rank to logger
logger = ExperimentLogger(run_dir=run_dir, rank=rank)

# Only rank 0 writes to main log files
logger.info("This message appears in all rank logs")
```

### Shared Run Directory

Create a single run directory shared across all processes:

```python
# DDP-safe directory creation
run_dir = mp.create_shared_run_dir(
    base_dir="runs",
    run_id=config.run_id
)
```

## Checkpoint Management

Manage model checkpoints with automatic cleanup and best model tracking.

### Basic Usage

```python
from expmate.torch import CheckpointManager

manager = CheckpointManager(
    checkpoint_dir='checkpoints',
    keep_last=3,        # Keep last 3 checkpoints
    keep_best=5,        # Keep top 5 checkpoints
    metric_name='val_loss',
    mode='min'          # Lower is better
)

# Save checkpoint
manager.save(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    metrics={'val_loss': 0.5, 'val_acc': 0.95}
)

# Load latest checkpoint
checkpoint = manager.load_latest()

# Load best checkpoint
best_checkpoint = manager.load_best()
```

## Experiment Tracking

Integrate with popular tracking tools.

### Weights & Biases

```python
from expmate.tracking import WandbTracker

tracker = WandbTracker(
    project="my-project",
    name=config.run_id,
    config=config.to_dict()
)

tracker.log({'train/loss': loss}, step=epoch)
tracker.finish()
```

### TensorBoard

```python
from expmate.tracking import TensorBoardTracker

tracker = TensorBoardTracker(log_dir=f'runs/{config.run_id}/tb')
tracker.log({'loss': loss}, step=epoch)
```

## Run Organization

ExpMate automatically organizes experiments:

```
runs/
├── exp_20250123_100000/
│   ├── run.yaml          # Config snapshot
│   ├── exp.log           # Human-readable log
│   ├── events.jsonl      # Machine-readable events
│   ├── metrics.csv       # All metrics
│   ├── best.json         # Best metric values
│   └── git_info.json     # Git commit info
├── exp_20250123_110000/
└── exp_20250123_120000/
```

## Next Steps

- [Configuration Management](../guide/configuration.md)
- [Experiment Logging](../guide/logging.md)
- [Checkpoint Management](../guide/checkpoints.md)
- [Distributed Training](../guide/distributed.md)

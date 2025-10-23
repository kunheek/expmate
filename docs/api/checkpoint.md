# API Reference: Checkpoint

## CheckpointManager

Manages model checkpoints with automatic cleanup and best model tracking.

```python
from expmate.torch import CheckpointManager

manager = CheckpointManager(
    checkpoint_dir='checkpoints',
    keep_last=3,
    keep_best=5,
    metric_name='val_loss',
    mode='min'
)
```

### Methods

#### `__init__(checkpoint_dir, keep_last=3, keep_best=3, metric_name='loss', mode='min')`

Initialize checkpoint manager.

**Parameters:**
- `checkpoint_dir` (str): Directory to save checkpoints
- `keep_last` (int): Number of last checkpoints to keep
- `keep_best` (int): Number of best checkpoints to keep
- `metric_name` (str): Metric name for tracking best checkpoints
- `mode` (str): 'min' or 'max' - whether lower or higher is better

#### `save(model, optimizer=None, scheduler=None, epoch=None, step=None, metrics=None, extra=None, filename=None)`

Save a checkpoint.

**Returns:** Path to saved checkpoint file

#### `load_latest()`

Load the most recent checkpoint.

**Returns:** Checkpoint dictionary

#### `load_best()`

Load the best checkpoint based on tracked metric.

**Returns:** Checkpoint dictionary

See the [source code](https://github.com/kunheek/expmate/blob/main/src/expmate/torch/checkpoint.py) for full API details.

# Experiment Logging

ExpMate provides a structured logging system for tracking experiments.

## ExperimentLogger

The `ExperimentLogger` class is the main interface for logging.

### Basic Setup

```python
from expmate import ExperimentLogger

logger = ExperimentLogger(
    run_dir='runs/exp1',
    rank=0,              # Process rank (for distributed training)
    run_id='exp1',       # Unique run identifier
    log_level='INFO',    # Logging level
    console_output=True  # Print to console
)
```

## Text Logging

### Level-Specific Methods (Preferred)

```python
logger.info("Training started")
logger.debug("Batch processing time: 0.5s")
logger.warning("Learning rate might be too high")
logger.error("NaN detected in loss")
```

### Generic Method

```python
logger.log("Training started", level="INFO")
logger.log("Debug message", level="DEBUG")
```

### Log Levels

- **DEBUG**: Detailed diagnostic information
- **INFO**: General information about progress
- **WARNING**: Warning messages for potential issues
- **ERROR**: Error messages

## Metrics Logging

### Log Individual Metrics

```python
logger.log_metric(
    step=epoch,         # Training step/epoch
    split='train',      # 'train', 'val', 'test', etc.
    name='loss',        # Metric name
    value=0.5          # Metric value
)

logger.log_metric(step=epoch, split='train', name='accuracy', value=0.95)
logger.log_metric(step=epoch, split='val', name='loss', value=0.4)
```

### Log Multiple Metrics

```python
# Log several metrics at once
for name, value in metrics.items():
    logger.log_metric(step=epoch, split='train', name=name, value=value)
```

## Best Metrics Tracking

Track and save the best values for important metrics:

```python
# Configure tracking
logger.track_best('val_loss', mode='min')      # Lower is better
logger.track_best('val_accuracy', mode='max')  # Higher is better

# Log metrics normally
logger.log_metric(step=epoch, split='val', name='loss', value=0.5)
logger.log_metric(step=epoch, split='val', name='accuracy', value=0.95)

# Best values are automatically tracked and saved to best.json
```

Access best metrics:

```python
best_loss = logger.best_metrics['val_loss']
print(f"Best loss: {best_loss['value']} at step {best_loss['step']}")
```

## Profiling

Profile code sections to measure execution time:

```python
# Context manager
with logger.profile('data_loading'):
    data = load_data()

with logger.profile('forward_pass'):
    output = model(input)

# Profiling results are logged automatically
```

## Log Files

ExpMate creates several log files automatically:

### Human-Readable Log (`exp.log`)

```
2025-01-23 14:30:22 - INFO - Starting experiment: exp_20250123_143022
2025-01-23 14:30:23 - INFO - Epoch 0/10: loss=0.5234
2025-01-23 14:30:24 - INFO - Epoch 1/10: loss=0.4567
```

### Machine-Readable Events (`events.jsonl`)

```json
{"timestamp": "2025-01-23T14:30:22", "level": "INFO", "message": "Starting experiment"}
{"timestamp": "2025-01-23T14:30:23", "level": "INFO", "message": "Epoch 0/10: loss=0.5234"}
```

### Metrics CSV (`metrics.csv`)

```csv
step,split,name,value,wall_time
0,train,loss,0.5234,1706017822
1,train,loss,0.4567,1706017823
0,val,loss,0.4123,1706017822
```

### Best Metrics (`best.json`)

```json
{
  "val_loss": {
    "value": 0.4123,
    "step": 5,
    "mode": "min"
  },
  "val_accuracy": {
    "value": 0.9567,
    "step": 8,
    "mode": "max"
  }
}
```

## Distributed Training

ExpMate supports rank-aware logging for distributed training:

```python
from expmate.torch import mp

# Setup DDP
rank, local_rank, world_size = mp.setup_ddp()

# Create logger with rank
logger = ExperimentLogger(
    run_dir=run_dir,
    rank=rank  # Each process has its own rank
)

# Rank 0 writes to main log files
# Other ranks write to separate files (exp_rank1.log, etc.)
if rank == 0:
    logger.info(f"Training on {world_size} GPUs")

# All ranks can log
logger.info(f"Rank {rank}: Processing batch")
```

### Log Files in DDP

```
runs/exp1/
├── exp.log                 # Rank 0 log
├── exp_rank1.log          # Rank 1 log
├── exp_rank2.log          # Rank 2 log
├── events.jsonl           # Rank 0 events
├── events_rank1.jsonl     # Rank 1 events
├── metrics.csv            # Rank 0 metrics (aggregated)
└── best.json              # Rank 0 best metrics
```

## Complete Example

```python
from expmate import ExperimentLogger, parse_config

# Parse config
config = parse_config()

# Create logger
logger = ExperimentLogger(run_dir=f"runs/{config.run_id}")
logger.info(f"Starting experiment: {config.run_id}")

# Configure best metric tracking
logger.track_best('val_loss', mode='min')
logger.track_best('val_accuracy', mode='max')

# Training loop
for epoch in range(config.training.epochs):
    # Training phase
    with logger.profile('train_epoch'):
        train_loss, train_acc = train_epoch(model, train_loader)
    
    logger.log_metric(step=epoch, split='train', name='loss', value=train_loss)
    logger.log_metric(step=epoch, split='train', name='accuracy', value=train_acc)
    
    # Validation phase
    with logger.profile('val_epoch'):
        val_loss, val_acc = validate(model, val_loader)
    
    logger.log_metric(step=epoch, split='val', name='loss', value=val_loss)
    logger.log_metric(step=epoch, split='val', name='accuracy', value=val_acc)
    
    # Log epoch summary
    logger.info(
        f"Epoch {epoch}/{config.training.epochs}: "
        f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
        f"val_acc={val_acc:.4f}"
    )

# Print best results
logger.info(f"Best validation loss: {logger.best_metrics['val_loss']['value']:.4f}")
logger.info(f"Best validation accuracy: {logger.best_metrics['val_accuracy']['value']:.4f}")
logger.info(f"Logs saved to: {logger.run_dir}")
```

## Best Practices

### 1. Use Descriptive Metric Names

```python
# Good
logger.log_metric(step=epoch, split='val', name='top1_accuracy', value=acc1)
logger.log_metric(step=epoch, split='val', name='top5_accuracy', value=acc5)

# Avoid
logger.log_metric(step=epoch, split='val', name='acc', value=acc)
```

### 2. Log at Appropriate Frequency

```python
# Log every epoch for validation metrics
logger.log_metric(step=epoch, split='val', name='loss', value=val_loss)

# Log every N steps for training metrics
if step % config.log_interval == 0:
    logger.log_metric(step=step, split='train', name='loss', value=loss)
```

### 3. Use Consistent Split Names

```python
# Standard splits
logger.log_metric(step=epoch, split='train', name='loss', value=loss)
logger.log_metric(step=epoch, split='val', name='loss', value=loss)
logger.log_metric(step=epoch, split='test', name='loss', value=loss)
```

### 4. Track Important Metrics

```python
# Track metrics you care about
logger.track_best('val_loss', mode='min')
logger.track_best('val_f1_score', mode='max')
logger.track_best('val_perplexity', mode='min')
```

### 5. Profile Critical Sections

```python
# Profile time-consuming operations
with logger.profile('data_loading'):
    batch = next(dataloader)

with logger.profile('forward_backward'):
    loss = model(batch)
    loss.backward()

with logger.profile('optimizer_step'):
    optimizer.step()
```

## Advanced Features

### Custom Log Formatting

```python
import logging

# Create custom handler with formatter
handler = logging.FileHandler('custom.log')
handler.setFormatter(
    logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
)
logger.log_handler = handler
```

### Reading Logs Programmatically

```python
import json
import pandas as pd

# Read metrics CSV
metrics_df = pd.read_csv('runs/exp1/metrics.csv')
train_loss = metrics_df[
    (metrics_df['split'] == 'train') & 
    (metrics_df['name'] == 'loss')
]

# Read events JSONL
events = []
with open('runs/exp1/events.jsonl') as f:
    for line in f:
        events.append(json.loads(line))

# Read best metrics
with open('runs/exp1/best.json') as f:
    best_metrics = json.load(f)
```

## See Also

- [Configuration Management](configuration.md)
- [Checkpoint Management](checkpoints.md)
- [CLI Tools](cli.md)
- [API Reference: Logger](../api/logger.md)

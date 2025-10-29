# Experiment Logging

ExpMate provides a structured logging system for tracking experiments with colorful console output, hierarchical stage tracking, and rate-limited logging.

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

### Key Features

- 🎨 **Colorful Console Output**: Automatic color-coding by log level (disabled in file logs)
- ⏱️ **Simple Timing**: `timer()` context manager for wall-clock timing
- 📂 **Hierarchical Stages**: `stage()` for tracking nested training phases
- 🔇 **Rate Limiting**: `log_every()` to reduce console spam
- 📊 **Best Metrics**: Automatic tracking of best model performance
- 🔄 **DDP Support**: Rank-aware logging for distributed training

## Text Logging

### Level-Specific Methods (Preferred)

```python
logger.info("Training started")        # Green in console
logger.debug("Batch processing time")  # Cyan in console
logger.warning("LR might be too high") # Yellow in console
logger.error("NaN detected in loss")   # Red in console
```

### Color Output

Console output is automatically colorized when connected to a terminal:
- **DEBUG**: Cyan
- **INFO**: Green
- **WARNING**: Yellow
- **ERROR**: Red
- **Stage messages**: Blue with bold stage names
- **Timer messages**: Cyan

Colors are automatically disabled when:
- Output is redirected to a file
- Output is piped to another command
- Running in non-TTY environment

File logs (`.log` files) and JSONL always use plain text without color codes.

### Generic Method

```python
logger.log("Training started", level="INFO")
logger.log("Debug message", level="DEBUG")
```

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
# Metrics are automatically tracked when logging
logger.log_metric(step=epoch, split='val', name='loss', value=0.5, track_best=True, mode='min')
logger.log_metric(step=epoch, split='val', name='accuracy', value=0.95, track_best=True, mode='max')

# Mode is auto-detected for common metric names
logger.log_metric(step=epoch, split='val', name='loss', value=0.5)  # Auto: mode='min'
logger.log_metric(step=epoch, split='val', name='accuracy', value=0.95)  # Auto: mode='max'
```

Access best metrics:

```python
best_loss = logger.get_best_metric('loss', split='val')
print(f"Best loss: {best_loss['value']} at step {best_loss['step']}")
```

## Timer (Simple Profiling)

Use `timer()` for simple wall-clock timing:

```python
# Basic timing
with logger.timer('data_loading'):
    data = load_data()
# Logs: "Timer [data_loading]: 0.1234s"

# Get elapsed time
with logger.timer('training_step') as result:
    loss = model(batch)
    loss.backward()
print(f"Step took {result['elapsed']:.4f}s")

# Silent timing (no logging)
with logger.timer('forward', log_result=False) as result:
    output = model(input)
elapsed = result['elapsed']
```

**Note**: For detailed GPU/CPU profiling, use `torch.profiler` directly. The `timer()` method provides simple wall-clock timing only.

**Global Control**: Timing can be disabled globally:
```python
import expmate
expmate.timer = False  # Disable all timing
```

## Hierarchical Stage Tracking

Track training stages with nested hierarchies:

```python
# Simple stage
with logger.stage('training'):
    train_model()
# Logs: "Stage [training] - START" and "Stage [training] - END (10.5s)"

# Stage with metadata
with logger.stage('epoch', epoch=5, lr=0.001):
    train_epoch()
# Logs: "Stage [epoch] (epoch=5, lr=0.001) - START"

# Nested stages create hierarchies
with logger.stage('epoch', epoch=5):
    with logger.stage('train'):
        train_loss = train_epoch()
    with logger.stage('validation'):
        val_loss = validate()
# Logs: "Stage [epoch/train] - START"
#       "Stage [epoch/validation] - START"
```

Stage context is preserved in JSONL logs for analysis:
```json
{"timestamp": 1234567890, "level": "INFO", "message": "Stage [epoch/train] - START", 
 "stage": "epoch/train", "stage_event": "start", "epoch": 5}
```

## Rate-Limited Logging

Reduce console spam in tight training loops:

```python
# Log every 100 iterations
for step in range(10000):
    loss = train_step()
    with logger.log_every(every=100):
        logger.info(f"Step {step}: loss={loss:.4f}")
# Logs to console every 100 steps, but ALL logs go to JSONL

# Time-based rate limiting (every 5 seconds)
for batch in dataloader:
    with logger.log_every(seconds=5.0):
        logger.info(f"Processing batch...")

# Multiple independent rate limiters
for step in range(1000):
    with logger.log_every(every=10, key='loss'):
        logger.info(f"Loss: {loss:.4f}")
    with logger.log_every(every=100, key='detailed'):
        logger.info(f"Detailed metrics: {metrics}")
```

**Key Points**:
- Console output is suppressed when not logging
- JSONL always captures all events
- Auto-generated keys based on call location
- Use custom `key` for multiple rate limiters

## Profiling (Deprecated)

**Note**: The `profile()` method is deprecated. Use `timer()` instead for simple timing, or `torch.profiler` for detailed profiling.

```python
# NEW (recommended)
with logger.timer('data_loading'):
    data = load_data()
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

# Training loop with hierarchical stages
for epoch in range(config.training.epochs):
    with logger.stage('epoch', epoch=epoch):
        # Training phase
        with logger.stage('train'):
            with logger.timer('train_epoch'):
                train_loss, train_acc = train_epoch(model, train_loader)
        
        logger.log_metric(step=epoch, split='train', name='loss', value=train_loss)
        logger.log_metric(step=epoch, split='train', name='accuracy', value=train_acc)
        
        # Validation phase
        with logger.stage('validation'):
            with logger.timer('val_epoch'):
                val_loss, val_acc = validate(model, val_loader)
        
        logger.log_metric(step=epoch, split='val', name='loss', value=val_loss)
        logger.log_metric(step=epoch, split='val', name='accuracy', value=val_acc)
        
        # Log epoch summary (with rate limiting)
        with logger.log_every(every=1):  # Log every epoch
            logger.info(
                f"Epoch {epoch}/{config.training.epochs}: "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"val_acc={val_acc:.4f}"
            )

# Print best results
best_loss = logger.get_best_metric('loss', split='val')
best_acc = logger.get_best_metric('accuracy', split='val')
logger.info(f"Best validation loss: {best_loss['value']:.4f} at epoch {best_loss['step']}")
logger.info(f"Best validation accuracy: {best_acc['value']:.4f} at epoch {best_acc['step']}")
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

# Use rate limiting for training metrics in tight loops
for step in range(10000):
    loss = train_step()
    with logger.log_every(every=100):
        logger.info(f"Step {step}: loss={loss:.4f}")
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
# Metrics are automatically tracked with auto-detection
logger.log_metric(step=epoch, split='val', name='loss', value=val_loss)  # mode='min'
logger.log_metric(step=epoch, split='val', name='f1_score', value=f1)    # mode='max'
logger.log_metric(step=epoch, split='val', name='perplexity', value=ppl) # mode='min'
```

### 5. Use Timer for Critical Sections

```python
# Time important operations
with logger.timer('data_loading'):
    batch = next(dataloader)

with logger.timer('forward_backward'):
    loss = model(batch)
    loss.backward()

with logger.timer('optimizer_step'):
    optimizer.step()
```

### 6. Use Hierarchical Stages

```python
# Organize training with stages
with logger.stage('epoch', epoch=epoch):
    with logger.stage('train'):
        train_loss = train_epoch()
    with logger.stage('validation'):
        val_loss = validate()
    with logger.stage('checkpoint'):
        save_checkpoint(model)
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

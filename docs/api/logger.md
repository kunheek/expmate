# API Reference: Logger

## ExperimentLogger

Structured logging and metrics tracking for experiments with colorful console output, timing, and hierarchical stage tracking.

```python
from expmate import ExperimentLogger

logger = ExperimentLogger(run_dir='runs/exp1')
logger.info("Training started")
logger.log_metric(step=0, split='train', name='loss', value=0.5)
```

### Methods

#### `__init__(run_dir, rank=0, run_id=None, log_level='INFO', console_output=True)`

Initialize experiment logger.

**Features:**
- Colorful console output (automatically disabled when output is redirected)
- File logging without colors (clean, parseable logs)
- JSONL event logging
- Metrics tracking with CSV storage

#### `info(message)`, `warning(message)`, `error(message)`, `debug(message)`

Log text messages at different levels with color-coded console output.

**Console Colors:**
- `DEBUG`: Cyan
- `INFO`: Green
- `WARNING`: Yellow
- `ERROR`: Red

#### `log_metric(step, split, name, value, track_best=True, mode=None)`

Log a metric value.

**Parameters:**
- `step` (int): Training step/epoch
- `split` (str): Data split ('train', 'val', 'test')
- `name` (str): Metric name
- `value` (float): Metric value
- `track_best` (bool): Whether to track best value
- `mode` (str): 'min' or 'max' for best tracking (auto-detected if None)

#### `timer(name, log_result=True)`

Context manager for timing code sections (simple wall-clock timing).

```python
# Basic timing
with logger.timer('data_loading'):
    data = load_data()

# Get elapsed time
with logger.timer('training_step') as result:
    loss = model(batch)
print(f"Step took {result['elapsed']:.4f}s")

# Silent timing (no logging)
with logger.timer('forward', log_result=False) as result:
    output = model(input)
```

**Parameters:**
- `name` (str): Name of the timed section
- `log_result` (bool): Whether to log the result

**Yields:**
- `dict`: Dictionary with `'elapsed'` key (in seconds)

**Note:** For detailed GPU/CPU profiling, use `torch.profiler` directly.

#### `stage(name, **metadata)`

Context manager for hierarchical stage tracking with duration logging.

```python
# Simple stage
with logger.stage('training'):
    train_model()

# With metadata
with logger.stage('epoch', epoch=5, lr=0.001):
    train_epoch()

# Nested stages
with logger.stage('epoch', epoch=5):
    with logger.stage('train'):
        train_loss = train_epoch()
    with logger.stage('validation'):
        val_loss = validate()
```

**Parameters:**
- `name` (str): Stage name
- `**metadata`: Additional metadata (e.g., epoch=5, batch=10)

**Yields:**
- `dict`: Stage info with `'name'`, `'metadata'`, `'elapsed'` keys

**Console Output:** Blue color with bold stage names

#### `log_every(every=None, seconds=None, key=None)`

Context manager for rate-limited logging to reduce console spam.

```python
# Log every 100 iterations
for step in range(10000):
    loss = train_step()
    with logger.log_every(every=100):
        logger.info(f"Step {step}: loss={loss:.4f}")

# Log every 5 seconds
for batch in dataloader:
    with logger.log_every(seconds=5.0):
        logger.info(f"Processing batch...")

# Multiple rate limiters
for step in range(1000):
    with logger.log_every(every=10, key='loss'):
        logger.info(f"Loss: {loss:.4f}")
    with logger.log_every(every=100, key='detailed'):
        logger.info(f"Detailed metrics: {metrics}")
```

**Parameters:**
- `every` (int): Log every N iterations
- `seconds` (float): Log every N seconds
- `key` (str): Unique key for this rate limiter (auto-generated if None)

**Yields:**
- `bool`: True if logging should occur, False if suppressed

**Note:** Must specify either `every` or `seconds`, not both.

#### `profile(name)` (Deprecated)

Deprecated alias for `timer()`. Use `timer()` instead.

#### `get_best_metric(name, split='val')`

Get the best value for a metric.

**Parameters:**
- `name` (str): Metric name
- `split` (str): Data split

**Returns:**
- `dict` or `None`: Best metric info with `'value'`, `'step'`, `'mode'` keys

### Color Output

Console output is automatically colorized when connected to a terminal (TTY). Colors are disabled when:
- Output is redirected to a file
- Output is piped to another command
- `sys.stdout.isatty()` returns False

File logs (`.log` files) and JSONL event logs always use plain text without color codes.

See the [source code](https://github.com/kunheek/expmate/blob/main/src/expmate/logger.py) for full API details.

# API Reference: Logger

## ExperimentLogger

Structured logging and metrics tracking for experiments.

```python
from expmate import ExperimentLogger

logger = ExperimentLogger(run_dir='runs/exp1')
logger.info("Training started")
logger.log_metric(step=0, split='train', name='loss', value=0.5)
```

### Methods

#### `__init__(run_dir, rank=0, run_id=None, log_level='INFO', console_output=True)`

Initialize experiment logger.

#### `info(message)`, `warning(message)`, `error(message)`, `debug(message)`

Log text messages at different levels.

#### `log_metric(step, split, name, value)`

Log a metric value.

**Parameters:**
- `step` (int): Training step/epoch
- `split` (str): Data split ('train', 'val', 'test')
- `name` (str): Metric name
- `value` (float): Metric value

#### `track_best(metric_name, mode='min')`

Track best value for a metric.

**Parameters:**
- `metric_name` (str): Name of metric to track
- `mode` (str): 'min' or 'max'

#### `profile(name)`

Context manager for profiling code sections.

```python
with logger.profile('data_loading'):
    data = load_data()
```

See the [source code](https://github.com/kunheek/expmate/blob/main/src/expmate/logger.py) for full API details.

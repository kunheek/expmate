# API Reference: Tracking

## WandB Tracker

Integration with Weights & Biases.

```python
from expmate.tracking import WandbTracker

tracker = WandbTracker(
    project="my-project",
    name="exp1",
    config=config.to_dict()
)

tracker.log({'train/loss': loss}, step=epoch)
tracker.finish()
```

### Methods

#### `__init__(project, name=None, config=None, **kwargs)`

Initialize WandB tracker.

#### `log(metrics, step=None)`

Log metrics to WandB.

#### `log_artifact(path, name, artifact_type='model')`

Log a file artifact.

#### `finish()`

Finish the WandB run.

## TensorBoard Tracker

Integration with TensorBoard.

```python
from expmate.tracking import TensorBoardTracker

tracker = TensorBoardTracker(log_dir='runs/exp1/tb')
tracker.log({'loss': loss}, step=epoch)
```

### Methods

#### `__init__(log_dir)`

Initialize TensorBoard tracker.

#### `log(metrics, step)`

Log metrics to TensorBoard.

#### `log_histogram(tag, values, step)`

Log histogram to TensorBoard.

See the [source code](https://github.com/kunheek/expmate/blob/main/src/expmate/tracking.py) for full API details.

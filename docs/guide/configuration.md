# Configuration Management

ExpMate provides a powerful configuration system for managing experiment parameters.

## Loading Configurations

### From YAML File

```python
from expmate.config import load_config

config = load_config('config.yaml')
```

### With Overrides

```python
config = load_config(
    'config.yaml',
    overrides=['training.lr=0.01', 'model.hidden_dim=512']
)
```

### From Dictionary

```python
config = load_config({
    'model': {'hidden_dim': 256},
    'training': {'lr': 0.001}
})
```

### Multiple Config Files

Merge multiple config files (later files override earlier ones):

```python
config = load_config(['base.yaml', 'experiment.yaml'])
```

## Command-Line Parsing

### Using parse_config

The simplest way to parse configs with CLI overrides:

```python
from expmate import parse_config

# Automatically parses first argument as config file
# and remaining arguments as overrides
config = parse_config()
```

Usage:

```bash
# Config file is the first positional argument
python train.py config.yaml training.lr=0.01
```

### Advanced Parser Usage

If you need additional command-line arguments:

```python
from expmate import ConfigArgumentParser

parser = ConfigArgumentParser()
# Config file is still the first positional argument
# You can add additional arguments if needed
args, config = parser.parse_args(), parser.parse_args()
```

Usage:

```bash
# Config file first, then overrides
python train.py config.yaml training.lr=0.01
```

## Override Syntax

### Basic Override

```bash
python train.py config.yaml key=value
```

### Nested Values

Use dot notation for nested values:

```bash
python train.py config.yaml model.hidden_dim=512
python train.py config.yaml training.optimizer.lr=0.01
```

### Adding New Keys

Add new keys with the `+` prefix:

```bash
python train.py config.yaml +optimizer.weight_decay=0.0001
```

Without `+`, overriding a non-existent key will raise an error.

### Type Hints

Use type hints for ambiguous values:

```bash
# Force float type (useful for scientific notation)
python train.py config.yaml training.lr:float=1e-3

# Force int type
python train.py config.yaml model.layers:int=10

# Force string type
python train.py config.yaml model.name:str=resnet
```

### Lists and Complex Values

#### JSON Format

Use JSON syntax for lists and dictionaries:

```bash
# Lists (JSON format)
python train.py config.yaml training.layers=[64,128,256]

# Dictionaries
python train.py config.yaml optimizer='{"name":"adam","lr":0.001}'
```

#### Space-Separated Lists

Provide lists as space-separated values with automatic type detection:

```bash
# Auto-detect type (integers in this case)
python train.py config.yaml training.gpus=0 1 2 3

# Auto-detect type (floats)
python train.py config.yaml model.dropout=0.1 0.2 0.3
```

#### Typed Lists

Use type hints to explicitly specify list element types:

```bash
# Force integer type
python train.py config.yaml training.gpu_ids:int=0 1 2 3

# Force float type
python train.py config.yaml model.scales:float=0.5 1.0 2.0

# Force string type
python train.py config.yaml data.splits:str=train val test

# Force boolean type
python train.py config.yaml flags:bool=true false true
```

## Configuration Introspection

### View Current Configuration

Display the current configuration structure:

```bash
python train.py config.yaml --show-config
```

This shows the complete configuration after loading the YAML file and applying any overrides.

### View Configuration Schema

Display configuration with type information:

```bash
python train.py config.yaml --config-help
```

Example output:

```
================================================================
Configuration Schema
================================================================

Available configuration keys and their current values:

  training.lr:
    type: float
    value: 0.001
    override: training.lr=<value>

  training.gpus:
    type: list[int]
    value: [0, 1]
    override: training.gpus=[...]  or  training.gpus:int=0 1 2

  model.hidden_dim:
    type: int
    value: 256
    override: model.hidden_dim=<value>

================================================================
```

## Variable Interpolation

### Timestamp Variables

```yaml
run_id: "exp_${now:%Y%m%d_%H%M%S}"  # exp_20250123_143022
log_dir: "logs_${now:%Y%m%d}"        # logs_20250123
```

Format codes follow Python's strftime:
- `%Y`: 4-digit year (2025)
- `%m`: Month (01-12)
- `%d`: Day (01-31)
- `%H`: Hour (00-23)
- `%M`: Minute (00-59)
- `%S`: Second (00-59)

### Environment Variables

```yaml
data_dir: "${DATA_ROOT}/train"
cache_dir: "${HOME}/.cache/expmate"
```

### Config References

Reference other config values:

```yaml
model:
  hidden_dim: 256
  
training:
  batch_size: 32

# Reference other values
model_name: "model_h${model.hidden_dim}"  # model_h256
output_dir: "outputs/${run_id}"
```

### Hostname Variables

```yaml
run_id: "exp_${hostname}_${now:%Y%m%d_%H%M%S}"
```

## Config Object

### Access Patterns

The `Config` object supports multiple access patterns:

```python
from expmate import Config

config = Config({
    'model': {'hidden_dim': 256},
    'training': {'lr': 0.001}
})

# Dictionary-style (supports any key)
print(config['model']['hidden_dim'])  # 256

# Attribute-style (preferred, cleaner syntax)
print(config.model.hidden_dim)  # 256
print(config.training.lr)  # 0.001

# Get with default
print(config.get('missing_key', 'default'))  # 'default'
```

### Conversion Methods

```python
# Convert to dictionary
config_dict = config.to_dict()

# Convert to flat dictionary
flat_dict = config.to_flat_dict()
# {'model.hidden_dim': 256, 'training.lr': 0.001}

# Convert to YAML string
yaml_str = config.to_yaml()
```

### Updating Config

```python
# Update single value
config.model.hidden_dim = 512

# Update multiple values
config.update({'training': {'lr': 0.01, 'epochs': 100}})

# Merge with another config
config.merge(other_config)
```

## Saving Configurations

### Save to YAML

```python
config.save('config.yaml')
```

### Save Run Config

ExpMate automatically saves the full config to the run directory:

```python
from expmate import ExperimentLogger

logger = ExperimentLogger(run_dir=f"runs/{config.run_id}")
# Automatically saves config to runs/{run_id}/run.yaml
```

## Advanced Features

### Config Validation

```python
from expmate.config import Config

config = Config(data)

# Check required keys
assert 'model' in config
assert 'training' in config

# Validate types
assert isinstance(config.training.lr, float)
assert isinstance(config.training.epochs, int)
```

### Deep Merging

When loading multiple config files, nested dictionaries are merged deeply:

```yaml
# base.yaml
model:
  type: resnet
  hidden_dim: 256
```

```yaml
# experiment.yaml
model:
  hidden_dim: 512
  dropout: 0.1
```

Result:
```python
config = load_config(['base.yaml', 'experiment.yaml'])
# config.model = {'type': 'resnet', 'hidden_dim': 512, 'dropout': 0.1}
```

### Type Preservation

ExpMate preserves types when overriding values:

```yaml
# config.yaml
training:
  lr: 0.001  # float
  epochs: 10  # int
```

```bash
# Type is preserved
python train.py config.yaml training.lr=0.01  # Stays float
python train.py config.yaml training.epochs=20  # Stays int

# Override with string (automatic type detection)
python train.py config.yaml training.lr=1e-3  # Parsed as float
```

## Best Practices

### 1. Use Hierarchical Organization

```yaml
# Good: organized by component
model:
  type: resnet
  hidden_dim: 256
  
data:
  path: /data
  batch_size: 32
  
training:
  epochs: 100
  lr: 0.001
```

### 2. Set Sensible Defaults

```yaml
# Include reasonable defaults for all parameters
seed: 42
device: cuda

model:
  type: resnet
  hidden_dim: 256
  dropout: 0.1
```

### 3. Use Descriptive Names

```yaml
# Good
training:
  learning_rate: 0.001
  num_epochs: 100
  
# Avoid
training:
  lr: 0.001
  n: 100
```

### 4. Document Your Config

```yaml
# Model configuration
model:
  type: resnet  # Options: resnet, vgg, densenet
  hidden_dim: 256  # Hidden layer dimension
  dropout: 0.1  # Dropout probability (0.0 - 1.0)
```

### 5. Use Variable Interpolation

```yaml
# Reference common values
run_id: "exp_${now:%Y%m%d_%H%M%S}"
output_dir: "outputs/${run_id}"
checkpoint_dir: "${output_dir}/checkpoints"
```

## Example: Complete Config File

```yaml
# Experiment identification
run_id: "exp_${now:%Y%m%d_%H%M%S}"
seed: 42
device: cuda

# Model architecture
model:
  type: resnet50
  pretrained: true
  num_classes: 10
  hidden_dim: 256
  dropout: 0.1

# Dataset configuration
data:
  root: ${DATA_ROOT}/imagenet
  train_split: train
  val_split: val
  batch_size: 32
  num_workers: 4
  augmentation:
    horizontal_flip: true
    random_crop: true
    normalize: true

# Training configuration
training:
  epochs: 100
  lr: 0.001
  weight_decay: 0.0001
  momentum: 0.9
  scheduler:
    type: cosine
    warmup_epochs: 10
  gradient_clip: 1.0

# Logging configuration
logging:
  interval: 100
  checkpoint_interval: 5
  track_best:
    - name: val_loss
      mode: min
    - name: val_accuracy
      mode: max

# Experiment tracking
wandb:
  enabled: true
  project: my-project
  entity: my-team
```

## See Also

- [Quick Start](../getting-started/quickstart.md)
- [Basic Concepts](../getting-started/concepts.md)
- [API Reference: Config](../api/config.md)

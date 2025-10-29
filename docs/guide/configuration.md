# Configuration Management

ExpMate provides a powerful typed configuration system using Python dataclasses with full IDE support.

## Quick Start

### 1. Define Your Config

Use dataclasses with type hints for IDE autocomplete and validation:

```python
from dataclasses import dataclass, field
from expmate import Config

@dataclass
class ModelConfig(Config):
    """Model configuration"""
    hidden_dim: int = 256
    dropout: float = 0.1
    layers: list[int] = field(default_factory=lambda: [128, 256, 512])

@dataclass
class TrainingConfig(Config):
    """Training configuration"""
    lr: float = 0.001
    epochs: int = 100
    batch_size: int = 32

@dataclass
class ExperimentConfig(Config):
    """Main experiment configuration"""
    run_id: str = "exp_${now:%Y%m%d_%H%M%S}"
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
```

### 2. Create YAML Config

```yaml
# config.yaml
seed: 42
model:
  hidden_dim: 256
  dropout: 0.1
training:
  lr: 0.001
  epochs: 100
  batch_size: 32
```

### 3. Load Config with Overrides

```python
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("config", help="Path to config file")
args, overrides = parser.parse_known_args()

# Load config with CLI overrides
config = ExperimentConfig.from_file(args.config, overrides=overrides)
```

### 4. Access Config Values

```python
# Typed access with IDE autocomplete
print(config.model.hidden_dim)  # 256
print(config.training.lr)       # 0.001

# Dictionary-style access also works
print(config['model']['hidden_dim'])  # 256
```

## Loading Configurations

### From YAML File

```python
# Load with defaults
config = ExperimentConfig.from_file('config.yaml')

# Load with CLI overrides
config = ExperimentConfig.from_file('config.yaml', overrides=overrides)
```

### From Dictionary

```python
config_dict = {
    'seed': 42,
    'model': {'hidden_dim': 256},
    'training': {'lr': 0.001}
}
config = ExperimentConfig.from_dict(config_dict)
```

### With Typed Validation

The dataclass automatically validates types:

```python
@dataclass
class Config(Config):
    lr: float = 0.001
    epochs: int = 100

# This will validate types
config = Config.from_dict({'lr': 0.01, 'epochs': 50})  # ✓
config = Config.from_dict({'lr': 'invalid'})  # ✗ Type error
```

## Command-Line Integration

### Recommended Pattern with argparse

```python
import argparse
from dataclasses import dataclass, field
from expmate import Config

@dataclass
class ExperimentConfig(Config):
    run_id: str = "exp_${now:%Y%m%d_%H%M%S}"
    seed: int = 42
    # ... other fields

def main():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("config", help="Path to config file")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    # parse_known_args returns (args, unknown_args)
    args, overrides = parser.parse_known_args()
    
    # Load config with overrides
    config = ExperimentConfig.from_file(args.config, overrides=overrides)
    
    # Use config
    print(f"Running experiment: {config.run_id}")
    print(f"Model: {config.model.hidden_dim} hidden dims")

if __name__ == "__main__":
    main()
```

Usage:

```bash
# Basic usage
python train.py config.yaml

# With overrides
python train.py config.yaml +training.lr=0.01 +model.hidden_dim=512

# With argparse flags
python train.py config.yaml --debug +training.epochs=50
```


## Override Syntax

### Basic Override

Override existing values:

```bash
python train.py config.yaml +training.lr=0.01
python train.py config.yaml +model.hidden_dim=512
```

### Nested Values

Use dot notation for deeply nested values:

```bash
python train.py config.yaml +model.hidden_dim=512
python train.py config.yaml +training.optimizer.lr=0.01
python train.py config.yaml +data.augmentation.strength=0.5
```

### Adding New Keys

Add new keys that don't exist in the config:

```bash
# Add new key
python train.py config.yaml +optimizer.weight_decay:float=0.0001

# Add nested new key
python train.py config.yaml +training.warmup_steps:int=1000
```

Without `+`, overriding a non-existent key will raise an error for safety.

### Type Hints

Use type hints to explicitly specify the type:

```bash
# Force float type (useful for scientific notation)
python train.py config.yaml +training.lr:float=1e-3

# Force int type
python train.py config.yaml +model.num_layers:int=12

# Force string type
python train.py config.yaml +model.name:str=resnet50

# Force boolean type
python train.py config.yaml +training.use_amp:bool=true
```

### Lists

ExpMate supports multiple ways to specify lists:

#### Space-Separated Lists (Recommended)

The most natural way - just separate values with spaces:

```bash
# Auto-detect type (integers)
python train.py config.yaml +training.gpu_ids:int 0 1 2 3

# Auto-detect type (floats)
python train.py config.yaml +model.dropout_rates:float 0.1 0.2 0.3

# Strings
python train.py config.yaml +data.splits:str train val test

# Mixed with other overrides
python train.py config.yaml +model.layers:int 128 256 512 training.lr=0.001
```

#### JSON Lists

Use JSON syntax for lists:

```bash
# Integers
python train.py config.yaml +model.layers=[128,256,512]

# Floats  
python train.py config.yaml +model.dropout=[0.1,0.2,0.3]

# Strings (needs quotes)
python train.py config.yaml '+data.splits=["train","val","test"]'
```

**Note:** JSON lists with strings need shell quotes to prevent parsing issues.

### Dictionaries

Use JSON syntax for dictionaries:

```bash
python train.py config.yaml '+optimizer={"name":"adam","lr":0.001,"betas":[0.9,0.999]}'
```

### Boolean Values

Multiple formats accepted:

```bash
# Standard boolean
python train.py config.yaml +training.use_amp=true
python train.py config.yaml +training.use_amp=false

# Also accepts
python train.py config.yaml +training.use_amp=True
python train.py config.yaml +training.use_amp=False
python train.py config.yaml +training.use_amp=1
python train.py config.yaml +training.use_amp=0
```

### Examples

```bash
# Single override
python train.py config.yaml +training.lr=0.01

# Multiple overrides
python train.py config.yaml +training.lr=0.01 +model.hidden_dim=512 +seed=123

# List as space-separated values
python train.py config.yaml +model.layers:int 128 256 512

# List as JSON
python train.py config.yaml +model.layers=[128,256,512]

# Type hints
python train.py config.yaml +training.lr:float=1e-3 +model.dropout:float=0.1

# Complex combination
python train.py config.yaml \
    +model.layers:int 128 256 512 \
    training.lr=0.001 \
    +training.warmup_steps:int=1000 \
    +seed=42
```


## Variable Interpolation

ExpMate supports dynamic variable interpolation in config files.

### Timestamp Variables

Use `${now:format}` for timestamps:

```yaml
run_id: "exp_${now:%Y%m%d_%H%M%S}"  # exp_20250128_143022
log_dir: "logs/${now:%Y%m%d}"        # logs/20250128
checkpoint: "ckpt_${now:%H%M%S}.pt"  # ckpt_143022.pt
```

Format codes follow Python's strftime:
- `%Y`: 4-digit year (2025)
- `%y`: 2-digit year (25)
- `%m`: Month (01-12)
- `%d`: Day (01-31)
- `%H`: Hour (00-23)
- `%M`: Minute (00-59)
- `%S`: Second (00-59)

### Environment Variables

Reference environment variables with `${VAR_NAME}`:

```yaml
data_dir: "${DATA_ROOT}/train"
cache_dir: "${HOME}/.cache/expmate"
output_dir: "${SCRATCH}/experiments"
api_key: "${WANDB_API_KEY}"
```

### Config References

Reference other config values:

```yaml
model:
  hidden_dim: 256
  
training:
  batch_size: 32

# Reference other values
model_name: "model_h${model.hidden_dim}"      # model_h256
output_dir: "outputs/${run_id}"
cache_path: "${data.root}/.cache"
```

### Hostname Variables

Use `${hostname}` for machine-specific configs:

```yaml
run_id: "exp_${hostname}_${now:%Y%m%d_%H%M%S}"
cache_dir: "/tmp/${hostname}/cache"
```

### Combined Example

```yaml
# Dynamic run identification
run_id: "exp_${now:%Y%m%d_%H%M%S}"
hostname: "${hostname}"

# Paths with interpolation
data_root: "${DATA_ROOT}/datasets"
output_dir: "outputs/${run_id}"
checkpoint_dir: "${output_dir}/checkpoints"
log_file: "${output_dir}/${run_id}.log"

model:
  name: "resnet_${model.depth}"
  depth: 50
```

## Config Access Patterns

### Attribute Access (Recommended)

Clean, type-safe access with IDE support:

```python
@dataclass
class ModelConfig(Config):
    hidden_dim: int = 256
    dropout: float = 0.1

@dataclass
class ExperimentConfig(Config):
    model: ModelConfig = field(default_factory=ModelConfig)

config = ExperimentConfig.from_file('config.yaml')

# Attribute access with autocomplete
print(config.model.hidden_dim)  # 256
print(config.model.dropout)     # 0.1

# Type-safe: your IDE knows the types!
config.model.hidden_dim = 512   # ✓
config.model.hidden_dim = "invalid"  # ✗ Type error
```

### Dictionary Access

Also supports dictionary-style access:

```python
# Dictionary-style
print(config['model']['hidden_dim'])  # 256

# Mixed access
print(config.model['dropout'])        # 0.1
print(config['model'].dropout)        # 0.1
```

### Iteration

```python
# Iterate over config keys
for key in config:
    print(f"{key}: {config[key]}")

# Check if key exists
if 'model' in config:
    print("Model config found")
```

### Conversion Methods

```python
# Convert to dictionary
config_dict = config.to_dict()
# {'model': {'hidden_dim': 256, 'dropout': 0.1}, ...}

# Convert to flat dictionary
flat_dict = config.to_flat_dict()
# {'model.hidden_dim': 256, 'model.dropout': 0.1, ...}

# Convert to YAML string
yaml_str = config.to_yaml()

# Save to file
config.save('output.yaml')
```

## Saving Configurations

### Save to YAML

```python
# Save current config
config.save('experiment.yaml')

# Save to specific directory
config.save('runs/exp_001/config.yaml')
```

### Automatic Saving with Logger

ExpMate automatically saves configs when using ExperimentLogger:

```python
from expmate import ExperimentLogger

logger = ExperimentLogger(run_dir=f"runs/{config.run_id}")
# Automatically saves config to runs/{run_id}/run.yaml
```

The saved config includes all applied overrides, making experiments fully reproducible.


## Advanced Patterns

### Nested Dataclasses

Build complex hierarchical configs:

```python
from dataclasses import dataclass, field
from expmate import Config

@dataclass
class OptimizerConfig(Config):
    name: str = "adam"
    lr: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0

@dataclass
class SchedulerConfig(Config):
    type: str = "cosine"
    warmup_epochs: int = 10
    min_lr: float = 1e-6

@dataclass
class TrainingConfig(Config):
    epochs: int = 100
    batch_size: int = 32
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

@dataclass
class ExperimentConfig(Config):
    run_id: str = "exp_${now:%Y%m%d_%H%M%S}"
    training: TrainingConfig = field(default_factory=TrainingConfig)
```

YAML:

```yaml
training:
  epochs: 100
  batch_size: 32
  optimizer:
    name: adamw
    lr: 0.001
    weight_decay: 0.01
  scheduler:
    type: cosine
    warmup_epochs: 10
```

Access:

```python
config = ExperimentConfig.from_file('config.yaml')
print(config.training.optimizer.lr)           # 0.001
print(config.training.scheduler.warmup_epochs) # 10
```

### Optional Fields

Use `Optional[]` for nullable fields:

```python
from typing import Optional

@dataclass
class ModelConfig(Config):
    name: str = "resnet50"
    pretrained_path: Optional[str] = None  # Can be None or str
    checkpoint: Optional[str] = None

# Usage
config = ModelConfig.from_dict({
    'name': 'resnet50',
    'checkpoint': 'model.pt'  # pretrained_path stays None
})
```

### Default Factories

Use `field(default_factory=...)` for mutable defaults:

```python
from typing import List

@dataclass
class Config(Config):
    # ✓ Correct: use default_factory for lists/dicts
    gpu_ids: List[int] = field(default_factory=lambda: [0, 1, 2, 3])
    layers: List[int] = field(default_factory=lambda: [128, 256, 512])
    metadata: dict = field(default_factory=dict)
    
    # ✗ Wrong: don't use mutable defaults directly
    # gpu_ids: List[int] = [0, 1, 2, 3]  # All instances share same list!
```

### Config Validation

Add validation in `__post_init__`:

```python
@dataclass
class TrainingConfig(Config):
    lr: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    
    def __post_init__(self):
        # Validation
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        
        # Derived values
        self.total_steps = self.epochs * 1000  # Assume 1000 steps/epoch
```

### Multiple Config Sources

Merge configs from multiple files:

```python
# Load base config
config = ExperimentConfig.from_file('base.yaml')

# Override with experiment-specific config
exp_config = ExperimentConfig.from_file('experiment.yaml')
config.update(exp_config.to_dict())

# Apply CLI overrides
config = ExperimentConfig.from_dict(config.to_dict(), overrides=overrides)
```

Or use YAML anchors/references:

```yaml
# base.yaml
defaults: &defaults
  seed: 42
  device: cuda

experiment1:
  <<: *defaults
  lr: 0.001

experiment2:
  <<: *defaults
  lr: 0.01
```


## Best Practices

### 1. Use Typed Dataclasses

Always use typed dataclasses for IDE support and validation:

```python
# ✓ Good: Typed with IDE autocomplete
@dataclass
class ModelConfig(Config):
    hidden_dim: int = 256
    dropout: float = 0.1

# ✗ Avoid: Plain dictionary (no type checking)
config = Config({'hidden_dim': 256, 'dropout': 0.1})
```

### 2. Hierarchical Organization

Organize configs by component:

```python
@dataclass
class DataConfig(Config):
    """Data loading configuration"""
    batch_size: int = 32
    num_workers: int = 4

@dataclass
class ModelConfig(Config):
    """Model architecture configuration"""
    hidden_dim: int = 256
    num_layers: int = 12

@dataclass
class TrainingConfig(Config):
    """Training configuration"""
    lr: float = 0.001
    epochs: int = 100

@dataclass
class ExperimentConfig(Config):
    """Main experiment configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
```

### 3. Document Your Config

Use docstrings and comments:

```python
@dataclass
class TrainingConfig(Config):
    """Training hyperparameters"""
    
    lr: float = 0.001
    """Learning rate (default: 0.001)"""
    
    epochs: int = 100
    """Number of training epochs"""
    
    warmup_steps: int = 1000
    """Number of warmup steps for learning rate scheduler"""
```

### 4. Sensible Defaults

Provide working defaults for all parameters:

```python
@dataclass
class Config(Config):
    # ✓ Good: sensible defaults
    seed: int = 42
    device: str = "cuda"
    lr: float = 0.001
    
    # ✗ Avoid: requiring user to set everything
    # seed: int  # No default!
```

### 5. Use Variable Interpolation

Avoid repetition with interpolation:

```yaml
# ✓ Good: DRY principle
run_id: "exp_${now:%Y%m%d_%H%M%S}"
output_dir: "outputs/${run_id}"
checkpoint_dir: "${output_dir}/checkpoints"
log_dir: "${output_dir}/logs"

# ✗ Avoid: repetition
run_id: "exp_20250128_143022"
output_dir: "outputs/exp_20250128_143022"
checkpoint_dir: "outputs/exp_20250128_143022/checkpoints"
```

### 6. Validation

Add validation for critical parameters:

```python
@dataclass
class TrainingConfig(Config):
    lr: float = 0.001
    batch_size: int = 32
    
    def __post_init__(self):
        if self.lr <= 0:
            raise ValueError(f"Learning rate must be positive, got {self.lr}")
        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {self.batch_size}")
```

### 7. Version Your Configs

Include version info for reproducibility:

```yaml
config_version: "1.0"
created: "${now:%Y-%m-%d %H:%M:%S}"
```

## Complete Example

```python
# config.py
from dataclasses import dataclass, field
from typing import Optional
from expmate import Config

@dataclass
class DataConfig(Config):
    """Data loading configuration"""
    root: str = "${DATA_ROOT}/imagenet"
    batch_size: int = 32
    num_workers: int = 4
    train_split: str = "train"
    val_split: str = "val"

@dataclass
class ModelConfig(Config):
    """Model architecture configuration"""
    name: str = "resnet50"
    pretrained: bool = True
    num_classes: int = 1000
    hidden_dim: int = 256
    dropout: float = 0.1

@dataclass
class OptimizerConfig(Config):
    """Optimizer configuration"""
    name: str = "adamw"
    lr: float = 0.001
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)

@dataclass
class TrainingConfig(Config):
    """Training configuration"""
    epochs: int = 100
    gradient_clip: float = 1.0
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    
    def __post_init__(self):
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")

@dataclass
class ExperimentConfig(Config):
    """Main experiment configuration"""
    run_id: str = "exp_${now:%Y%m%d_%H%M%S}"
    seed: int = 42
    device: str = "cuda"
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# train.py
import argparse
from expmate import ExperimentLogger, set_seed

def main():
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument("config", help="Path to config file")
    args, overrides = parser.parse_known_args()
    
    # Load config with overrides
    config = ExperimentConfig.from_file(args.config, overrides=overrides)
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Initialize logger (automatically saves config)
    logger = ExperimentLogger(run_dir=f"runs/{config.run_id}")
    logger.info(f"Starting experiment: {config.run_id}")
    logger.info(f"Config: {config.to_yaml()}")
    
    # Train model
    train(config, logger)

if __name__ == "__main__":
    main()
```

YAML config:

```yaml
# config.yaml
seed: 42
device: cuda

data:
  root: /data/imagenet
  batch_size: 64
  num_workers: 8

model:
  name: resnet50
  pretrained: true
  hidden_dim: 256
  dropout: 0.1

training:
  epochs: 100
  gradient_clip: 1.0
  optimizer:
    name: adamw
    lr: 0.001
    weight_decay: 0.01
```

Usage:

```bash
# Basic usage
python train.py config.yaml

# With overrides
python train.py config.yaml \
    +model.hidden_dim=512 \
    +training.epochs=200 \
    +training.optimizer.lr=0.0001

# With list overrides
python train.py config.yaml \
    +data.gpu_ids:int 0 1 2 3 \
    +training.epochs=50
```

## See Also

- [Quick Start](../getting-started/quickstart.md)
- [Basic Concepts](../getting-started/concepts.md)
- [API Reference: Config](../api/config.md)
- [Examples: Linear Regression](../examples/training.md)

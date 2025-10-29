# Typed Configuration with LSP Support

The `Config` class now supports Python dataclasses for full LSP (Language Server Protocol) support in your IDE!

## Features

- ✅ **Full IDE autocomplete** - Get suggestions for all config fields
- ✅ **Type checking** - Catch type errors before runtime
- ✅ **Inline documentation** - Hover to see field descriptions
- ✅ **Refactoring support** - Safely rename fields across your codebase
- ✅ **Go-to-definition** - Jump to config schema definitions
- ✅ **Proper type inference** - LSP knows the exact type returned from `from_file()`/`from_dict()`
- ✅ **Backward compatible** - Existing code continues to work

## Quick Start

### Method 1: Dynamic Config (Original)

Works like before - no type hints, but very flexible:

```python
from expmate.config import Config

# Load config dynamically
config = Config.from_file("config.yaml")

# Access with dot notation (no autocomplete)
lr = config.training.lr
epochs = config.training.epochs
```

### Method 2: Typed Config (NEW - Recommended!)

Define your config schema for full LSP support:

```python
from dataclasses import dataclass
from expmate.config import Config

# Define typed config schema
@dataclass
class TrainingConfig(Config):
    """Training hyperparameters."""
    learning_rate: float
    batch_size: int
    epochs: int
    optimizer: str = "adam"  # with default

# Load config with types!
config = TrainingConfig.from_file("config.yaml")

# Full LSP autocomplete and type checking!
lr: float = config.learning_rate  # IDE knows this is float
batch_size: int = config.batch_size  # IDE knows this is int
```

## Usage Examples

### Basic Typed Config

```python
from dataclasses import dataclass
from expmate.config import Config

@dataclass
class MyConfig(Config):
    name: str
    seed: int
    learning_rate: float
    description: str = "My experiment"

# Load from YAML
config = MyConfig.from_file("config.yaml")

# Or from dict
config = MyConfig.from_dict({
    "name": "exp1",
    "seed": 42,
    "learning_rate": 0.001
})

# Access with full LSP support
print(f"Experiment: {config.name}")
print(f"LR: {config.learning_rate}")
```

### Nested Typed Config

```python
from dataclasses import dataclass
from typing import Optional
from expmate.config import Config

@dataclass
class ModelConfig(Config):
    """Model architecture."""
    hidden_dim: int
    num_layers: int
    dropout: float = 0.1

@dataclass
class ExperimentConfig(Config):
    """Main experiment config."""
    name: str
    seed: int
    # Nested configs can be dict or typed
    model: dict  # For now, use dict for nested configs

config = ExperimentConfig.from_file("config.yaml")

# Top-level fields have full LSP support
name: str = config.name
seed: int = config.seed

# Nested dicts accessed normally
model_hidden = config.model["hidden_dim"]
```

### Mixed Dynamic and Typed Access

```python
# Even with typed configs, you can still use dynamic access
config = MyConfig.from_file("config.yaml")

# Typed access (preferred)
name: str = config.name

# Dict-style access (still works)
name = config["name"]

# Convert to dict
config_dict = config.to_dict()

# Save to file
config.save("saved_config.yaml")

# Generate hash
hash_val = config.hash()
```

## Migration Guide

### Existing Code (No Changes Needed)

```python
# This still works exactly as before!
config = Config("config.yaml")
lr = config.training.lr
```

### Upgrade to Typed Config

1. **Define your schema:**

```python
@dataclass
class MyConfig(Config):
    learning_rate: float
    batch_size: int
    epochs: int
```

2. **Update initialization:**

```python
# Old
config = Config("config.yaml")

# New (both work!)
config = Config.from_file("config.yaml")  # Dynamic
config = MyConfig.from_file("config.yaml")  # Typed
```

3. **Enjoy LSP support!**

Your IDE will now provide autocomplete, type checking, and documentation for all config fields.

## Best Practices

1. **Define schemas for important configs** - Use dataclasses for configs you use frequently
2. **Use type hints** - Annotate variables: `lr: float = config.learning_rate`
3. **Add docstrings** - Document your config classes and fields
4. **Use defaults** - Provide sensible defaults: `optimizer: str = "adam"`
5. **Keep flexibility** - Use dynamic `Config` for one-off experiments

## YAML Config Example

```yaml
# config.yaml
name: my_experiment
seed: 42
learning_rate: 0.001
batch_size: 32
epochs: 100
optimizer: adam
```

```python
# Python code
@dataclass
class MyConfig(Config):
    name: str
    seed: int
    learning_rate: float
    batch_size: int
    epochs: int
    optimizer: str

config = MyConfig.from_file("config.yaml")
# All fields autocomplete! ✨
```

## Advanced Features

### Optional Fields

```python
from typing import Optional

@dataclass
class MyConfig(Config):
    required_field: str
    optional_field: Optional[int] = None
```

### Overrides

```python
# Works with typed configs too!
config = MyConfig.from_file(
    "config.yaml",
    overrides=["learning_rate=0.01", "batch_size=64"]
)
```

### Run Directory Snapshots

```python
from pathlib import Path

config = MyConfig.from_file(
    "config.yaml",
    run_dir=Path("runs/exp_001")
)
# Automatically saves config snapshot to runs/exp_001/run.yaml
```

## Troubleshooting

### "Config has no attribute X"

Make sure your YAML keys match your dataclass field names:

```python
@dataclass
class MyConfig(Config):
    learning_rate: float  # ← Must match YAML key
```

```yaml
learning_rate: 0.001  # ← Must match dataclass field
```

### Type Mismatches

The config will try to load values as the specified type. Make sure your YAML values match:

```python
@dataclass
class MyConfig(Config):
    epochs: int  # YAML should have: epochs: 100 (not "100")
    lr: float    # YAML should have: lr: 0.001 (not 1e-3 notation might work)
```

## Summary

The new dataclass-based `Config` class gives you the best of both worlds:

- **Flexibility** - Dynamic configs for quick experiments
- **Type safety** - Typed configs for production code
- **LSP support** - Full IDE integration for typed configs
- **Backward compatible** - All existing code still works!

Start using typed configs today and enjoy a better development experience! 🚀

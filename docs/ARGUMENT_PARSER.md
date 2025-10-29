# Config Override Pattern with LSP Support

The recommended pattern for using argument parsing with Config provides full LSP support for type inference, autocomplete, and go-to-definition.

## Why This Pattern?

✅ **Full LSP Support** - Type inference works perfectly  
✅ **Simple** - Uses standard argparse, no custom wrappers  
✅ **Clear** - Explicit separation of CLI args and config  
✅ **Flexible** - Full control over both argparse and config  

## Recommended Pattern

```python
import argparse
from dataclasses import dataclass
from expmate import Config, override_config

@dataclass
class TrainingConfig(Config):
    lr: float = 0.001
    batch_size: int = 32
    epochs: int = 100

# Standard argparse
parser = argparse.ArgumentParser()
parser.add_argument("--config-file", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--device", type=str, default="cuda")

# Parse arguments
args, unknown = parser.parse_known_args()

# Load config - LSP knows exact type!
if args.config_file:
    config = TrainingConfig.from_file(args.config_file)  # ✨ LSP: TrainingConfig
else:
    config = TrainingConfig.from_dict({})  # ✨ LSP: TrainingConfig

# Apply + prefix overrides - Type is preserved!
config = override_config(config, unknown)  # ✨ LSP still knows: TrainingConfig

# Use with full autocomplete!
print(config.lr)  # ✨ LSP autocompletes .lr, .batch_size, .epochs
```

## LSP Benefits

The type information is preserved throughout the entire workflow:

1. **From file/dict** - `config = TrainingConfig.from_file(...)` → LSP knows it's `TrainingConfig`
2. **After overrides** - `config = override_config(config, unknown)` → LSP still knows it's `TrainingConfig`!
3. **Autocomplete** - IDE suggests `config.lr`, `config.batch_size`, `config.epochs`
4. **Go-to-Definition** - Jump directly to the dataclass field
5. **Type Checking** - Static type checkers know the exact type
6. **Refactoring** - Rename fields across entire codebase

## Command Line Usage

```bash
# Use defaults
python train.py --seed 123

# Load config from file
python train.py --config-file config.yaml --seed 123

# Override config with + prefix
python train.py --seed 123 +lr=0.01 +batch_size=64

# Combine all
python train.py --config-file config.yaml --seed 123 +lr=0.01 +epochs=200

# Add new dynamic config
python train.py +experiment.name="test" +debug=true

# Pass sequences/lists naturally
python train.py +layers 64 128 256 512
python train.py +ids:int 1 2 3 4 5
python train.py +learning_rates:float 0.1 0.01 0.001

# Mix sequences with other overrides
python train.py --config-file config.yaml +lr=0.01 +layers 128 256 512 +epochs=200
```

## Config Override Formats

The `override_config()` function supports these formats with `+` prefix:

| Format | Example | Description |
|--------|---------|-------------|
| `+key=value` | `+lr=0.01` | Basic key-value |
| `+key value` | `+lr 0.01` | Space-separated |
| `+key val1 val2 val3` | `+layers 64 128 256` | Sequence/list (auto-detect types) |
| `+nested.key=value` | `+model.hidden_dim=512` | Nested config |
| `+key:type=value` | `+lr:float=0.01` | With type hint |
| `+key:type value` | `+lr:float 0.01` | Type hint, space-separated |
| `+key:type v1 v2 v3` | `+ids:int 1 2 3` | Typed sequence/list |

## Warning System

If you accidentally pass `-` or `--` prefixed args to `override_config()`, you'll get a warning:

```python
args, unknown = parser.parse_known_args(["--ignored", "+lr=0.01"])
config = override_config(config, unknown)
# ⚠️  Warning: Found arguments with '-' or '--' prefix: ['--ignored'].
#     These will be ignored by override_config().
#     Use '+' prefix for config overrides (e.g., +key=value).
```

## Complete Example

```python
import argparse
from dataclasses import dataclass
from expmate import Config, override_config

@dataclass
class TrainingConfig(Config):
    """Training configuration with full LSP support."""
    lr: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    optimizer: str = "adam"

def main():
    parser = argparse.ArgumentParser(description="Training script")
    parser.add_argument("--config-file", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--verbose", action="store_true")
    
    args, unknown = parser.parse_known_args()
    
    # Load config with LSP support
    if args.config_file:
        config = TrainingConfig.from_file(args.config_file)
    else:
        config = TrainingConfig.from_dict({})
    
    # Override with + prefix args
    config = override_config(config, unknown)
    
    # Use with full IDE support!
    print(f"Training with lr={config.lr}, batch_size={config.batch_size}")
    print(f"Seed: {args.seed}, Device: {args.device}")

if __name__ == "__main__":
    main()
```

Run with:
```bash
python train.py --config-file config.yaml --seed 999 +lr=0.005 +batch_size=128
```

## Why Not a Custom Parser?

We previously had `ArgumentParser` and `ConfigArgumentParser` wrapper classes, but removed them because:

1. **LSP doesn't work** - Return type `config` is not explicitly typed
2. **Too complex** - Custom wrappers add unnecessary complexity  
3. **Less flexible** - Can't use full argparse features
4. **Standard is better** - Everyone knows argparse

The recommended pattern is simpler, more explicit, and gives perfect LSP support! ✨

## Nested Configs

You can define config classes with nested structure for better organization:

```python
from dataclasses import dataclass, field
from expmate import Config, override_config

@dataclass
class ModelConfig(Config):
    """Model architecture configuration."""
    dim: int = 128
    layers: int = 4
    dropout: float = 0.1

@dataclass
class OptimizerConfig(Config):
    """Optimizer configuration."""
    name: str = "adam"
    lr: float = 0.001
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])

@dataclass
class TrainingConfig(Config):
    """Main config with nested configs."""
    seed: int = 42
    epochs: int = 100
    
    # Nested configs - use field(default_factory=...)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

# Load and use
config = TrainingConfig.from_dict({})
config = override_config(config, unknown)

# LSP autocompletes everything!
print(config.model.dim)  # ✨ Autocomplete: .dim, .layers, .dropout
print(config.optimizer.lr)  # ✨ Autocomplete: .name, .lr, .betas
```

Override nested configs from command line:

```bash
# Override nested fields with dot notation
python train.py +model.dim=512 +model.layers=12
python train.py +optimizer.lr=0.0001 +optimizer.name=adamw

# Override sequences in nested configs
python train.py +optimizer.betas:float 0.95 0.999

# Mix top-level and nested overrides
python train.py +seed=999 +model.dim=256 +optimizer.lr=0.01
```

See `examples/example_nested_config.py` for a complete example.

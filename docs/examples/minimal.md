# Minimal Example

This example demonstrates the simplest way to use ExpMate with typed configuration, experiment logging, and metrics tracking.

## Source Code

The complete example is available at [`examples/00_minimal.py`](https://github.com/kunheek/expmate/blob/main/examples/00_minimal.py).

## Quick Start

```bash
# Run with defaults
python examples/00_minimal.py

# Run with config file
python examples/00_minimal.py examples/conf/example.yaml

# Run with config file and overrides
python examples/00_minimal.py examples/conf/example.yaml +training.lr=0.01 +training.epochs=10
```

## Configuration

### Using Typed Config (Recommended)

```python
from dataclasses import dataclass, field
from expmate import Config

@dataclass
class ModelConfig(Config):
    """Model configuration"""
    input_dim: int = 10
    output_dim: int = 3

@dataclass
class TrainingConfig(Config):
    """Training configuration"""
    epochs: int = 5
    lr: float = 0.001

@dataclass
class ExperimentConfig(Config):
    """Main experiment configuration"""
    run_id: str = "minimal_${now:%Y%m%d_%H%M%S}"
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
```

### Optional: YAML Config File

Create `examples/conf/example.yaml`:

```yaml
run_id: "exp_${now:%Y%m%d_%H%M%S}"
seed: 42

model:
  input_dim: 10
  output_dim: 3

training:
  epochs: 5
  lr: 0.001
```

## Training Script

```python
#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from expmate import ExperimentLogger, Config, set_seed


def main():
    # Parse config from file if provided, otherwise use defaults
    import sys
    if len(sys.argv) > 1 and not sys.argv[1].startswith('+'):
        config_file = sys.argv[1]
        overrides = sys.argv[2:]
    else:
        config_file = None
        overrides = sys.argv[1:]
    
    config = ExperimentConfig.from_args(config_file, overrides) if config_file else ExperimentConfig.from_args(overrides=overrides)

    # Set random seed for reproducibility
    set_seed(config.seed)

    # Create experiment logger
    logger = ExperimentLogger(run_dir=f"runs/{config.run_id}")
    logger.info(f"Starting experiment: {config.run_id}")

    # Create model and data
    model = nn.Linear(config.model.input_dim, config.model.output_dim)
    dataset = TensorDataset(
        torch.randn(100, config.model.input_dim),
        torch.randint(0, config.model.output_dim, (100,)),
    )
    dataloader = DataLoader(dataset, batch_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(config.training.epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logger.log_metric(step=epoch, split="train", name="loss", value=avg_loss)
        logger.info(f"Epoch {epoch}/{config.training.epochs}: loss={avg_loss:.4f}")

    logger.info(f"Logs saved to: {logger.run_dir}")


if __name__ == "__main__":
    main()
```

## Running the Example

### Basic Run (No Config File)

```bash
python examples/00_minimal.py
```

### With Config File

```bash
python examples/00_minimal.py examples/conf/example.yaml
```

### With Parameter Overrides

```bash
# Change learning rate
python examples/00_minimal.py examples/conf/example.yaml +training.lr=0.01

# Change multiple parameters
python examples/minimal.py examples/conf/example.yaml training.lr=0.01 training.epochs=10

# Add new parameters
python examples/minimal.py examples/conf/example.yaml +optimizer.weight_decay=0.0001
```

## Output

ExpMate creates a structured run directory:

```
runs/
└── exp_20250123_143022/
    ├── run.yaml          # Saved configuration
    ├── exp.log           # Training logs
    ├── events.jsonl      # Structured events
    ├── metrics.csv       # Training metrics
    └── best.json         # Best metric values
```

### Viewing Logs

```bash
# View training logs
cat runs/exp_20250123_143022/exp.log

# View metrics
cat runs/exp_20250123_143022/metrics.csv
```

### Comparing Runs

```bash
# Run multiple experiments
python examples/minimal.py examples/conf/example.yaml training.lr=0.001
python examples/minimal.py examples/conf/example.yaml training.lr=0.01
python examples/minimal.py examples/conf/example.yaml training.lr=0.1

# Compare results
expmate compare runs/exp_*
```

## Key Concepts Demonstrated

1. **Configuration Management**: Loading and overriding config values
2. **Experiment Logging**: Structured logging with automatic file management
3. **Metrics Tracking**: Logging and tracking training metrics
4. **Reproducibility**: Setting seeds and saving full configuration

## Next Steps

- [Complete Training Example](training.md)
- [Distributed Training Example](ddp.md)
- [Configuration Guide](../guide/configuration.md)

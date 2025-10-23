# Minimal Example

This example demonstrates the simplest way to use ExpMate.

## Source Code

The complete example is available at [`examples/minimal.py`](https://github.com/kunheek/expmate/blob/main/examples/minimal.py).

## Configuration File

Create `conf/example.yaml`:

```yaml
run_id: "exp_${now:%Y%m%d_%H%M%S}"
seed: 42

model:
  input_dim: 128
  output_dim: 10

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

from expmate import ExperimentLogger, parse_config, set_seed


def main():
    # Parse config from command line
    config = parse_config()

    # Set random seed for reproducibility
    set_seed(int(config.seed))

    # Create experiment logger
    logger = ExperimentLogger(run_dir=f"runs/{config.run_id}")
    logger.info(f"Starting experiment: {config.run_id}")
    logger.info(f"Config: seed={config.seed}, lr={config.training.lr}")

    # Create dummy model and data
    model = nn.Linear(int(config.model.input_dim), int(config.model.output_dim))
    dataset = TensorDataset(
        torch.randn(100, int(config.model.input_dim)),
        torch.randint(0, int(config.model.output_dim), (100,)),
    )
    dataloader = DataLoader(dataset, batch_size=32)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.training.lr))
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(int(config.training.epochs)):
        total_loss = 0

        for batch_x, batch_y in dataloader:
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        # Log metrics
        logger.log_metric(step=epoch, split="train", name="loss", value=avg_loss)
        logger.info(f"Epoch {epoch}/{config.training.epochs}: loss={avg_loss:.4f}")

    logger.info("Training complete!")
    logger.info(f"Logs saved to: {logger.run_dir}")


if __name__ == "__main__":
    main()
```

## Running the Example

### Basic Run

```bash
python examples/minimal.py examples/conf/example.yaml
```

### With Parameter Overrides

```bash
# Change learning rate
python examples/minimal.py examples/conf/example.yaml training.lr=0.01

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

# Quick Start

This guide will walk you through creating your first experiment with ExpMate.

## Step 1: Create a Configuration File

Create a YAML configuration file `config.yaml`:

```yaml
# Run identification
run_id: "exp_${now:%Y%m%d_%H%M%S}"
seed: 42

# Model architecture
model:
  input_dim: 128
  hidden_dim: 256
  output_dim: 10

# Training configuration
training:
  epochs: 10
  lr: 0.001
  batch_size: 32
```

## Step 2: Write Your Training Script

Create `train.py`:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from expmate import ExperimentLogger, parse_config, set_seed

def main():
    # Parse config from command line
    config = parse_config()
    
    # Set random seed for reproducibility
    set_seed(config.seed)
    
    # Create experiment logger
    logger = ExperimentLogger(run_dir=f"runs/{config.run_id}")
    logger.info(f"Starting experiment: {config.run_id}")
    logger.info(f"Config: {config.to_dict()}")
    
    # Create model
    model = nn.Sequential(
        nn.Linear(config.model.input_dim, config.model.hidden_dim),
        nn.ReLU(),
        nn.Linear(config.model.hidden_dim, config.model.output_dim)
    )
    
    # Create dummy dataset
    dataset = TensorDataset(
        torch.randn(1000, config.model.input_dim),
        torch.randint(0, config.model.output_dim, (1000,))
    )
    dataloader = DataLoader(dataset, batch_size=config.training.batch_size)
    
    # Setup training
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
        
        # Log metrics
        logger.log_metric(step=epoch, split='train', name='loss', value=avg_loss)
        logger.info(f"Epoch {epoch}/{config.training.epochs}: loss={avg_loss:.4f}")
    
    logger.info("Training complete!")
    logger.info(f"Logs saved to: {logger.run_dir}")

if __name__ == "__main__":
    main()
```

## Step 3: Run Your Experiment

Run with default config:

```bash
python train.py config.yaml
```

Run with parameter overrides:

```bash
python train.py config.yaml training.lr=0.01 training.epochs=20
```

Add new parameters:

```bash
python train.py config.yaml +optimizer.weight_decay=0.0001
```

## Step 4: Check Your Results

ExpMate automatically creates a run directory with the following structure:

```
runs/
└── exp_20250123_143022/
    ├── run.yaml              # Full config used for this run
    ├── exp.log               # Human-readable logs
    ├── events.jsonl          # Machine-readable event logs
    ├── metrics.csv           # All logged metrics
    └── best.json             # Best metric values
```

## Step 5: Compare Experiments

Run multiple experiments with different hyperparameters:

```bash
python train.py config.yaml training.lr=0.001
python train.py config.yaml training.lr=0.01
python train.py config.yaml training.lr=0.1
```

Compare them using the CLI:

```bash
expmate compare runs/exp_*
```

Visualize training curves:

```bash
expmate viz runs/exp_* --metrics loss
```

## What's Next?

- Learn about [Basic Concepts](concepts.md)
- Explore [Configuration Management](../guide/configuration.md)
- Check out [Complete Examples](../examples/minimal.md)

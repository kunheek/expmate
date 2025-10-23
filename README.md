# ExpMate

**ML Research Boilerplate — Config & Logging First**

[![PyPI version](https://img.shields.io/pypi/v/expmate.svg)](https://pypi.org/project/expmate/)
[![Python versions](https://img.shields.io/pypi/pyversions/expmate.svg)](https://pypi.org/project/expmate/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ExpMate is a lightweight experiment management toolkit designed for ML researchers who want to focus on their experiments, not on boilerplate code. It provides clean, reusable patterns for configuration management, logging, and experiment tracking—everything you need to run reproducible ML experiments.

## 🌟 Key Features

- **🔧 Configuration Management**: YAML-based configs with command-line overrides, type preservation, and nested value support
- **📊 Experiment Logging**: Structured logging with metrics tracking, best model monitoring, and rank-aware DDP support
- **🚀 PyTorch Integration**: Checkpoint management, DDP utilities, and distributed training helpers
- **📈 Experiment Tracking**: Built-in support for WandB and TensorBoard
- **🔍 CLI Tools**: Compare runs, visualize metrics, and manage hyperparameter sweeps
- **🔄 Git Integration**: Automatic git info tracking for reproducibility
- **⚡ Zero Configuration**: Works out of the box with sensible defaults

## 📦 Installation

### Basic Installation

Install ExpMate with all core features:

```bash
pip install expmate
```

**What's included:**
- ✅ **Configuration parser** - YAML configs with CLI overrides
- ✅ **Experiment logger** - Structured logging and metrics tracking
- ✅ **CLI tools** - `compare`, `visualize`, `sweep` commands
- ✅ **Visualization** - Plot metrics with matplotlib
- ✅ **Data analysis** - Fast data processing with polars
- ✅ **System monitoring** - Track CPU/memory usage with psutil

### Optional: Experiment Tracking

Add integration with popular tracking platforms:

```bash
# Weights & Biases only
pip install expmate[wandb]

# TensorBoard only
pip install expmate[tensorboard]

# Both tracking platforms
pip install expmate[tracking]
```

### Using with PyTorch

ExpMate works great with PyTorch! **Install PyTorch separately:**

```bash
pip install expmate
pip install torch torchvision  # Install PyTorch your way
```

## 🚀 Quick Start

### Minimal Example

```python
from expmate import ExperimentLogger, parse_config, set_seed

# Parse config from YAML + command-line overrides
config = parse_config()

# Set random seed for reproducibility
set_seed(config.seed)

# Create experiment logger
logger = ExperimentLogger(run_dir=f"runs/{config.run_id}")
logger.info(f"Starting experiment: {config.run_id}")

# Your training code here...
for epoch in range(config.training.epochs):
    # ... training logic ...

    # Log metrics
    logger.log_metric(step=epoch, split='train', name='loss', value=loss)
    logger.info(f"Epoch {epoch}: loss={loss:.4f}")

logger.info("Training complete!")
```

### Configuration File (YAML)

```yaml
# config.yaml
run_id: "exp_${now:%Y%m%d_%H%M%S}"  # Auto-generate with timestamp
seed: 42

model:
  input_dim: 128
  hidden_dim: 256
  output_dim: 10

training:
  epochs: 10
  lr: 0.001
  batch_size: 32
```

### Running with Overrides

```bash
# Basic run
python train.py config.yaml

# Override parameters
python train.py config.yaml training.lr=0.01 training.epochs=20

# Add new parameters
python train.py config.yaml +optimizer.weight_decay=0.0001

# Type hints for ambiguous values
python train.py config.yaml training.lr:float=1e-3
```

## 📚 Core Components

### Configuration Parser

ExpMate provides a powerful configuration system with automatic type inference and CLI overrides:

```python
from expmate import ConfigArgumentParser

parser = ConfigArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--model-type", type=str, default="resnet")
config = parser.parse_config()

# Access nested config values
print(config.model.hidden_dim)
print(config.training.lr)
```

### Experiment Logger

Structured logging with automatic file management:

```python
from expmate import ExperimentLogger

logger = ExperimentLogger(
    run_dir='runs/exp1',
    rank=0,  # For distributed training
    log_level='INFO'
)

# Text logging
logger.info("Training started")
logger.warning("Learning rate might be too high")
logger.error("NaN detected in loss")

# Metrics logging
logger.log_metric(step=100, split='train', name='loss', value=0.5)
logger.log_metric(step=100, split='train', name='accuracy', value=0.95)

# Track best metrics automatically
logger.track_best('val_loss', mode='min')
logger.track_best('val_accuracy', mode='max')

# Profiling
with logger.profile('data_loading'):
    data = load_data()
```

### PyTorch Checkpoint Manager

Intelligent checkpoint management with automatic cleanup:

```python
from expmate.torch import CheckpointManager

manager = CheckpointManager(
    checkpoint_dir='checkpoints',
    keep_last=3,      # Keep last 3 checkpoints
    keep_best=5,      # Keep top 5 checkpoints
    metric_name='val_loss',
    mode='min'
)

# Save checkpoint
manager.save(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=epoch,
    metrics={'val_loss': 0.5, 'val_acc': 0.95}
)

# Load checkpoint
checkpoint = manager.load_latest()
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Load best checkpoint
best_checkpoint = manager.load_best()
```

### Distributed Training (DDP) Support

Easy distributed training setup:

```python
from expmate.torch import mp
from expmate import ExperimentLogger

# Setup DDP
rank, local_rank, world_size = mp.setup_ddp()

# Create shared run directory (DDP-safe!)
run_dir = mp.create_shared_run_dir(base_dir="runs", run_id=config.run_id)

# Rank-aware logging
logger = ExperimentLogger(run_dir=run_dir, rank=rank)
if rank == 0:
    logger.info(f"Training on {world_size} GPUs")

# Your DDP training code...
```

Run with torchrun:

```bash
torchrun --nproc_per_node=4 train.py config.yaml
```

## 🛠️ CLI Tools

ExpMate includes powerful command-line tools for experiment management:

### Compare Runs

```bash
# Compare multiple experiments
expmate compare runs/exp1 runs/exp2 runs/exp3

# Compare specific metrics
expmate compare runs/exp* --metrics loss accuracy

# Export to CSV
expmate compare runs/exp* --output results.csv

# Show config differences
expmate compare runs/exp* --show-config
```

### Visualize Metrics

```bash
# Plot training curves
expmate viz runs/exp1 --metrics loss accuracy

# Compare multiple runs
expmate viz runs/exp1 runs/exp2 runs/exp3 --metrics loss

# Specify output file
expmate viz runs/exp1 --output metrics.png

# Different plot styles
expmate viz runs/exp1 --style seaborn
```

### Hyperparameter Sweeps

```bash
# Grid search with Python
expmate sweep "python train.py {config}" \
  --config config.yaml \
  --sweep "training.lr=[0.001,0.01,0.1]" \
          "model.hidden_dim=[128,256,512]"

# With torchrun for distributed training
expmate sweep "torchrun --nproc_per_node=4 train.py {config}" \
  --config config.yaml \
  --sweep "training.lr=[0.001,0.01,0.1]" \
          "model.hidden_dim=[128,256]"

# Dry run to preview commands
expmate sweep "python train.py {config}" \
  --config config.yaml \
  --sweep "training.lr=[0.001,0.01]" \
  --dry-run
```

## 🔌 Experiment Tracking Integration

### Weights & Biases

```python
from expmate.tracking import WandbTracker

tracker = WandbTracker(
    project="my-project",
    name=config.run_id,
    config=config.to_dict()
)

# Log metrics
tracker.log({'train/loss': loss, 'train/acc': acc}, step=epoch)

# Log artifacts
tracker.log_artifact(path='model.pt', name='final_model')
tracker.finish()
```

### TensorBoard

```python
from expmate.tracking import TensorBoardTracker

tracker = TensorBoardTracker(log_dir=f'runs/{config.run_id}/tensorboard')

# Log metrics
tracker.log({'loss': loss, 'accuracy': acc}, step=epoch)

# Log histograms
tracker.log_histogram('weights', model.fc.weight, step=epoch)
```

## 📖 Examples

Check out the [`examples/`](examples/) directory for complete examples:

- **[`minimal.py`](examples/minimal.py)**: Quick start guide with basic usage
- **[`train.py`](examples/train.py)**: Complete DDP training example with all features
- **[`train.sh`](examples/train.sh)**: Shell script for running experiments

## 🎯 Design Philosophy

ExpMate is built around three core principles:

1. **Configuration First**: All experiments start with a config file, making them reproducible and easy to modify
2. **Logging First**: Structured logging and metrics tracking from day one
3. **Zero Boilerplate**: Sensible defaults that work out of the box, with customization when needed

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

ExpMate was created to simplify ML research workflows. Special thanks to the open-source community for inspiration and tools.

## 📞 Contact

- **Author**: Kunhee Kim
- **Email**: kunhee.kim@kaist.ac.kr
- **GitHub**: [kunheek/expmate](https://github.com/kunheek/expmate)

## 🔗 Links

- [PyPI Package](https://pypi.org/project/expmate/)
- [GitHub Repository](https://github.com/kunheek/expmate)
- [Documentation](https://github.com/kunheek/expmate#readme)
- [Issue Tracker](https://github.com/kunheek/expmate/issues)

---

**Made with ❤️ for ML researchers**

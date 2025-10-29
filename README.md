# ExpMate

**Lightweight ML Experiment Management — Configuration & Logging Made Simple**

[![PyPI version](https://img.shields.io/pypi/v/expmate.svg)](https://pypi.org/project/expmate/)
[![Python versions](https://img.shields.io/pypi/pyversions/expmate.svg)](https://pypi.org/project/expmate/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ExpMate is a minimalist experiment management toolkit for ML researchers. It provides clean, type-safe configuration management and colorful structured logging—everything you need to run reproducible experiments without the boilerplate.

## 📑 Table of Contents

- [Why ExpMate?](#-why-expmate)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Key Features](#-key-features)
  - [Type-Safe Configuration](#type-safe-configuration)
  - [Colorful Logging](#colorful-logging-with-stages--timing)
  - [Metrics Tracking](#metrics-tracking--best-value-monitoring)
  - [PyTorch Checkpoints](#pytorch-checkpoint-management)
  - [Distributed Training](#distributed-training-ddp-support)
- [CLI Tools](#️-cli-tools)
- [Experiment Tracking](#-experiment-tracking)
- [Examples](#-examples)
- [Design Philosophy](#-design-philosophy)
- [Contributing](#-contributing)

## ✨ Why ExpMate?

- **🎯 Type-Safe Configs**: Dataclass-based configurations with automatic YAML loading and CLI overrides
- **🎨 Colorful Logging**: Beautiful console output with hierarchical stages, timing utilities, and rate limiting
- **📊 Smart Tracking**: Automatic metrics logging with best-value tracking and multiple output formats
- **🚀 PyTorch Ready**: Built-in checkpoint management and distributed training (DDP) support
- **🔧 Professional CLI**: Powerful argument parsing with `parse_known_args()` pattern
- **📈 Analysis Tools**: Compare experiments, visualize metrics, and run hyperparameter sweeps
- **⚡ Zero Dependencies**: Core features work with just Python stdlib (optional deps for extras)

## 📦 Installation

```bash
pip install expmate
```

### Optional Dependencies

```bash
# For experiment tracking
pip install expmate[wandb]        # Weights & Biases
pip install expmate[tensorboard]  # TensorBoard
pip install expmate[tracking]     # Both W&B and TensorBoard

# For PyTorch features (checkpoints, DDP)
pip install torch  # Install PyTorch separately
```

**Note**: ExpMate's core features (config, logging, CLI tools) work without any optional dependencies!

## 🚀 Quick Start

### 1. Define Your Configuration

Create a dataclass that inherits from `Config`:

```python
# train.py
from dataclasses import dataclass
from expmate import Config, ExperimentLogger, set_seed

@dataclass
class TrainingConfig:
    lr: float
    epochs: int

@dataclass
class ExperimentConfig(Config):
    run_id: str
    seed: int
    training: TrainingConfig
```

### 2. Create a Config File

```yaml
# config.yaml
run_id: my_experiment
seed: 42

training:
  lr: 0.001
  epochs: 10
```

### 3. Load Config and Run

```python
import argparse

def main():
    # Parse arguments - config file + overrides
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args, overrides = parser.parse_known_args()

    # Load config with type safety
    config = ExperimentConfig.from_file(args.config, overrides=overrides)

    # Set reproducibility
    set_seed(config.seed)

    # Create logger
    logger = ExperimentLogger(run_dir=f"runs/{config.run_id}")
    logger.info(f"🚀 Starting: {config.run_id}")

    # Your training loop
    for epoch in range(config.training.epochs):
        loss = train_epoch()  # Your training code

        logger.log_metric(step=epoch, split='train', name='loss', value=loss)
        logger.info(f"Epoch {epoch}/{config.training.epochs}: loss={loss:.4f}")

    logger.info("✅ Training complete!")

if __name__ == "__main__":
    main()
```

### 4. Run with CLI Overrides

```bash
# Use default config
python train.py

# Override parameters (note the + prefix for adding/modifying)
python train.py +training.lr=0.01 +training.epochs=20

# Use different config
python train.py --config custom.yaml +training.lr=0.001

# Type hints for explicit type conversion
python train.py +training.lr:float=1e-3 +training.batch_size:int=64

# List/sequence overrides with type hints
python train.py +model.layers:int 128 256 512  # List of integers
python train.py +data.splits:str train val test  # List of strings

# Or use JSON list format
python train.py '+model.layers=[128,256,512]' '+data.splits=["train","val","test"]'
```

## 🎨 Key Features

### Type-Safe Configuration

Define configs as dataclasses for full IDE support and type checking:

```python
from dataclasses import dataclass
from expmate import Config

@dataclass
class ModelConfig:
    hidden_dim: int
    dropout: float
    num_layers: int

@dataclass
class ExperimentConfig(Config):
    seed: int
    model: ModelConfig
```

Load from YAML with automatic type conversion:

```python
# Load config
config = ExperimentConfig.from_file("config.yaml")

# Type-safe access with autocomplete
print(config.model.hidden_dim)  # IDE knows this is an int!

# Override from CLI
config = ExperimentConfig.from_file(
    "config.yaml",
    overrides=["+model.hidden_dim=512", "+model.dropout=0.2"]
)
```

### Colorful Logging with Stages & Timing

ExpMate's logger provides beautiful, informative console output:

```python
from expmate import ExperimentLogger

logger = ExperimentLogger(run_dir="runs/exp1")

# Hierarchical stages for organized output
with logger.stage("data_preparation"):
    with logger.timer("load_data"):
        data = load_dataset()  # Automatically shows timing
    logger.info(f"Loaded {len(data)} samples")

with logger.stage("training"):
    for epoch in range(10):
        with logger.stage("epoch", epoch=epoch):
            loss = train_epoch()
            logger.log_metric(step=epoch, split='train', name='loss', value=loss)

# Rate limiting to reduce console spam
for step in range(10000):
    loss = train_step()
    with logger.log_every(every=100):  # Log every 100 steps
        logger.info(f"Step {step}: loss={loss:.4f}")
```

**Output:**
```
INFO: Stage [data_preparation] - START
INFO: Timer [load_data]: 0.234s
INFO: Loaded 50000 samples
INFO: Stage [data_preparation] - END (0.245s)
INFO: Stage [training] - START
INFO: Stage [epoch] (epoch=0) - START
INFO: Stage [epoch] (epoch=0) - END (2.1s)
```

### Metrics Tracking & Best Value Monitoring

Automatically track metrics and find best values:

```python
logger = ExperimentLogger(run_dir="runs/exp1")

# Log metrics - multiple formats supported
logger.log_metric(step=0, split='train', name='loss', value=0.5)
logger.log_metric(step=0, split='val', name='accuracy', value=0.85)

# Track best values automatically
logger.track_best(split='val', name='loss', value=0.3, mode='min', step=10)
logger.track_best(split='val', name='accuracy', value=0.92, mode='max', step=10)

# Retrieve best values
best = logger.get_best()
print(f"Best val/loss: {best['val/loss']}")  # Shows value and step
```

**Output formats:**
- `metrics.csv` - Easy to analyze in pandas/Excel
- `events.jsonl` - Structured JSON lines
- `best.json` - Best metric values with steps

### PyTorch Checkpoint Management

Smart checkpoint saving with automatic cleanup:

```python
from expmate.torch import CheckpointManager

manager = CheckpointManager(
    checkpoint_dir="checkpoints",
    keep_last=3,      # Keep 3 most recent
    keep_best=5,      # Keep 5 best checkpoints
    metric_name="val_loss",
    mode="min"
)

# Save checkpoint
manager.save(
    model=model,
    optimizer=optimizer,
    epoch=epoch,
    metrics={"val_loss": 0.25, "val_acc": 0.95}
)

# Load latest or best
checkpoint = manager.load_latest()
checkpoint = manager.load_best()

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

### Distributed Training (DDP) Support

Easy multi-GPU training with automatic rank handling:

```python
from expmate.torch import mp, mp_tqdm
from expmate import ExperimentLogger

# Setup DDP
rank, local_rank, world_size = mp.setup_ddp()

# Create shared run directory (DDP-safe!)
run_dir = mp.create_shared_run_dir(base_dir="runs", run_id=config.run_id)

# Rank-aware logging (only rank 0 logs by default)
logger = ExperimentLogger(run_dir=run_dir, rank=rank)

# Training loop with progress bars (only on local_rank 0)
if rank == 0:
    logger.info(f"🚀 Training on {world_size} GPUs")

for epoch in mp_tqdm(range(epochs), desc="Epochs"):
    for batch in mp_tqdm(train_loader, desc="Training", leave=False):
        loss = train_step(batch)  # Your training code
```

Run with torchrun:

```bash
torchrun --nproc_per_node=4 train.py --config config.yaml +training.lr=0.01
```## 🛠️ CLI Tools

ExpMate includes powerful command-line tools for experiment analysis:

### Compare Experiments

```bash
# Compare multiple runs
expmate compare runs/exp1 runs/exp2 runs/exp3

# Show specific metrics
expmate compare runs/exp* --metrics loss accuracy

# Export comparison to CSV
expmate compare runs/exp* --output comparison.csv

# Show configuration differences
expmate compare runs/exp* --show-config
```

### Visualize Metrics

```bash
# Plot training curves
expmate visualize runs/exp1

# Compare multiple experiments
expmate visualize runs/exp1 runs/exp2 runs/exp3

# Save to file
expmate visualize runs/exp* --output metrics.png

# Specify metrics to plot
expmate visualize runs/exp1 --metrics train/loss val/loss val/accuracy
```

### Hyperparameter Sweeps

```bash
# Run grid search
expmate sweep "python train.py --config {config}" \
  --base-config config.yaml \
  --params "training.lr=[0.001,0.01,0.1]" \
           "model.hidden_dim=[128,256,512]"

# With distributed training
expmate sweep "torchrun --nproc_per_node=4 train.py --config {config}" \
  --base-config config.yaml \
  --params "training.lr=[0.001,0.01,0.1]"

# Preview commands (dry run)
expmate sweep "python train.py --config {config}" \
  --base-config config.yaml \
  --params "training.lr=[0.001,0.01]" \
  --dry-run
```

## 🔌 Experiment Tracking

ExpMate integrates with popular tracking platforms:

### Weights & Biases

```python
from expmate.tracking import WandbTracker

tracker = WandbTracker(
    project="my-project",
    name=config.run_id,
    config=config.to_dict()
)

# Log metrics
tracker.log({"train/loss": loss, "val/accuracy": acc}, step=epoch)

# Log artifacts
tracker.log_artifact(path="model.pt", name="final_model")
tracker.finish()
```

### TensorBoard

```python
from expmate.tracking import TensorBoardTracker

tracker = TensorBoardTracker(log_dir=f"runs/{config.run_id}/tensorboard")

# Log metrics
tracker.log({"loss": loss, "accuracy": acc}, step=epoch)

# Log histograms
tracker.log_histogram("weights", model.fc.weight, step=epoch)
tracker.close()
```

## 📖 Examples

Check out the [`examples/`](examples/) directory for complete, runnable examples:

### Progressive Tutorial

1. **[`00_minimal.py`](examples/00_minimal.py)** - Simplest possible usage
   - Basic config loading and logging
   - 30 lines of code

2. **[`01_linear_regression.py`](examples/01_linear_regression.py)** - Core features
   - Typed configs with dataclasses
   - Hierarchical logging with stages
   - Timer utilities and metrics tracking
   - NumPy-based, no PyTorch required

3. **[`02_mnist_classification.py`](examples/02_mnist_classification.py)** - PyTorch training
   - PyTorch model training
   - Checkpoint management
   - Early stopping and best model tracking
   - Complete training pipeline

4. **[`03_ddp_training.py`](examples/03_ddp_training.py)** - Distributed training
   - Multi-GPU training with DDP
   - Rank-aware logging
   - Metric aggregation across processes
   - Production-ready distributed setup

### Helper Scripts

- **[`run_ddp.sh`](examples/run_ddp.sh)** - Launch distributed training
- **[`run_sweep.sh`](examples/run_sweep.sh)** - Run hyperparameter sweeps
- **[`run_compare.sh`](examples/run_compare.sh)** - Compare experiment results
- **[`run_visualize.sh`](examples/run_visualize.sh)** - Visualize metrics

### Running Examples

```bash
# Minimal example
python examples/00_minimal.py +training.lr=0.001

# Linear regression with overrides
python examples/01_linear_regression.py +training.epochs=100 +model.lr=0.02

# MNIST training
python examples/02_mnist_classification.py --config examples/conf/mnist.yaml

# Distributed training (4 GPUs)
torchrun --nproc_per_node=4 examples/03_ddp_training.py
```

## 💡 Design Philosophy

ExpMate is built on three core principles:

1. **Type Safety First**: Use dataclasses for configs to catch errors early and get full IDE support
2. **Minimal Boilerplate**: Sensible defaults that work out-of-the-box, customize only when needed
3. **Developer Experience**: Beautiful console output, clear error messages, and intuitive APIs

### Why Not Hydra/MLflow/Sacred?

- **Hydra**: Powerful but complex. ExpMate is simpler, lighter, and more Pythonic.
- **MLflow**: Great for model deployment. ExpMate focuses on the experiment phase.
- **Sacred**: Feature-rich but heavy. ExpMate gives you just what you need.

ExpMate is the **lightweight alternative** for researchers who want clean code without sacrificing functionality.

## 🗺️ Roadmap

- [ ] **Cloud storage support** - S3, GCS, Azure Blob integration
- [ ] **Experiment resumption** - Automatic checkpoint restoration
- [ ] **Advanced sweeps** - Bayesian optimization, hyperband
- [ ] **Web dashboard** - Real-time experiment monitoring
- [ ] **Profiling tools** - Memory, compute, and I/O profiling

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Report bugs** - Open an issue with details and reproduction steps
2. **Suggest features** - Share your ideas in GitHub discussions
3. **Submit PRs** - Fix bugs, add features, improve docs
4. **Share examples** - Contribute real-world usage examples

Please see our [Contributing Guide](docs/contributing.md) for details.

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

ExpMate was created to simplify ML research workflows. Special thanks to:

- The PyTorch team for inspiration on API design
- The Hydra project for configuration management ideas
- All contributors and users of ExpMate

## 📞 Contact & Links

- **Author**: Kunhee Kim (kunhee.kim@kaist.ac.kr)
- **PyPI**: [pypi.org/project/expmate](https://pypi.org/project/expmate/)
- **GitHub**: [github.com/kunheek/expmate](https://github.com/kunheek/expmate)
- **Documentation**: [kunheek.github.io/expmate](https://kunheek.github.io/expmate)
- **Issues**: [github.com/kunheek/expmate/issues](https://github.com/kunheek/expmate/issues)

---

<p align="center">
  <b>Empowering ML researchers to focus on science, not boilerplate</b><br>
  <sub>Star ⭐ the repo if you find ExpMate useful!</sub>
</p>

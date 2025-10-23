# ExpMate

**ML Research Boilerplate — Config & Logging First**

Welcome to ExpMate, a lightweight experiment management toolkit designed for ML researchers who want to focus on their experiments, not on boilerplate code.

## Overview

ExpMate provides clean, reusable patterns for configuration management, logging, and experiment tracking—everything you need to run reproducible ML experiments.

## Key Features

- **🔧 Configuration Management**: YAML-based configs with command-line overrides
- **📊 Experiment Logging**: Structured logging with metrics tracking
- **🚀 PyTorch Integration**: Checkpoint management and DDP utilities
- **📈 Experiment Tracking**: Built-in support for WandB and TensorBoard
- **🔍 CLI Tools**: Compare runs, visualize metrics, and manage sweeps
- **🔄 Git Integration**: Automatic git info tracking for reproducibility

## Quick Example

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
```

## Installation

```bash
pip install expmate
```

For additional features:

```bash
# PyTorch support
pip install expmate[torch]

# Experiment tracking (WandB, TensorBoard)
pip install expmate[tracking]

# Visualization tools
pip install expmate[viz]

# Everything
pip install expmate[all]
```

## Next Steps

- [Installation Guide](getting-started/installation.md)
- [Quick Start Tutorial](getting-started/quickstart.md)
- [Basic Concepts](getting-started/concepts.md)
- [API Reference](api/config.md)

## Support

- **GitHub**: [kunheek/expmate](https://github.com/kunheek/expmate)
- **Issues**: [Issue Tracker](https://github.com/kunheek/expmate/issues)
- **Email**: kunhee.kim@kaist.ac.kr

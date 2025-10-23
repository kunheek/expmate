# Installation

## Requirements

ExpMate requires Python 3.8 or later.

## Basic Installation

Install ExpMate using pip:

```bash
pip install expmate
```

This installs the core package with all essential dependencies:
- ✅ **Configuration parser** - YAML configs with CLI overrides (pyyaml)
- ✅ **Experiment logger** - Structured logging and metrics tracking (numpy)
- ✅ **CLI tools** - `compare`, `visualize`, `sweep` commands
- ✅ **Visualization** - Plot metrics with matplotlib
- ✅ **Data analysis** - Fast data processing with polars
- ✅ **System monitoring** - Track CPU/memory usage with psutil

## Optional Dependencies

### Experiment Tracking

For integration with Weights & Biases and TensorBoard:

```bash
# Weights & Biases only
pip install expmate[wandb]

# TensorBoard only
pip install expmate[tensorboard]

# Both tracking platforms
pip install expmate[tracking]
```

### Development Tools

For contributing to ExpMate:

```bash
pip install expmate[dev]
```

This includes:
- `pytest>=7.0.0`
- `pytest-cov>=4.0.0`
- `black>=23.0.0`
- `ruff>=0.1.0`
- `mypy>=1.0.0`
- `pre-commit>=3.0.0`

## Install from Source

To install from source (for development or latest features):

```bash
git clone https://github.com/kunheek/expmate.git
cd expmate
pip install -e .
```

## Verify Installation

Verify the installation by checking the version:

```bash
python -c "import expmate; print(expmate.__version__)"
```

Or use the CLI:

```bash
expmate --version
```

## Next Steps

- [Quick Start Guide](quickstart.md)
- [Basic Concepts](concepts.md)

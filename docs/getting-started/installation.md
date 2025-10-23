# Installation

## Requirements

ExpMate requires Python 3.8 or later.

## Basic Installation

Install ExpMate using pip:

```bash
pip install expmate
```

This installs the core package with minimal dependencies (PyYAML and NumPy).

## Optional Dependencies

ExpMate provides several optional dependency groups for different use cases:

### PyTorch Support

For PyTorch-specific features like checkpoint management and DDP utilities:

```bash
pip install expmate[torch]
```

This includes:
- `torch>=1.12.0`

### Experiment Tracking

For integration with Weights & Biases and TensorBoard:

```bash
pip install expmate[tracking]
```

This includes:
- `wandb>=0.15.0`
- `tensorboard>=2.13.0`

### Visualization Tools

For CLI visualization and comparison tools:

```bash
pip install expmate[viz]
```

This includes:
- `matplotlib>=3.5.0`
- `polars>=0.20.0`

### System Monitoring

For tracking system resources during training:

```bash
pip install expmate[monitor]
```

This includes:
- `psutil>=5.9.0`

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

### All Dependencies

To install everything:

```bash
pip install expmate[all]
```

## Install from Source

To install from source (for development or latest features):

```bash
git clone https://github.com/kunheek/expmate.git
cd expmate
pip install -e .
```

With optional dependencies:

```bash
pip install -e ".[all]"
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

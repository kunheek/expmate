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

### Experiment Tracking

For integration with Weights & Biases and TensorBoard:

```bash
pip install expmate[tracking]
```

This includes:
- `wandb>=0.15.0`
- `tensorboard>=2.13.0`

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

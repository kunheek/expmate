# Contributing to ExpMate

Thank you for your interest in contributing to ExpMate! This guide will help you get started.

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/YOUR_USERNAME/expmate.git
cd expmate
```

### 2. Install Development Dependencies

```bash
pip install -e ".[dev]"
```

This installs ExpMate in editable mode along with development tools:
- pytest: Testing framework
- pytest-cov: Coverage reporting
- black: Code formatting
- ruff: Linting
- mypy: Type checking
- pre-commit: Git hooks

### 3. Install Pre-commit Hooks

```bash
pre-commit install
```

This sets up automatic code formatting and linting on commit.

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/add-new-parser`
- `fix/checkpoint-bug`
- `docs/update-readme`

### 2. Make Your Changes

Write clean, well-documented code following the existing style.

### 3. Write Tests

Add tests for your changes in the `tests/` directory:

```python
# tests/test_your_feature.py
import pytest
from expmate import YourFeature

def test_your_feature():
    feature = YourFeature()
    assert feature.works() == True
```

### 4. Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=expmate

# Run specific test file
pytest tests/test_config.py

# Run specific test
pytest tests/test_config.py::test_load_config
```

### 5. Format Code

```bash
# Format with black
black src/expmate tests

# Lint with ruff
ruff check src/expmate tests

# Type check with mypy
mypy src/expmate
```

### 6. Commit Changes

```bash
git add .
git commit -m "Add feature: descriptive message"
```

Pre-commit hooks will automatically format and lint your code.

### 7. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub.

## Code Style

### Python Style Guide

- Follow PEP 8
- Use black for formatting (line length: 88)
- Use type hints for function signatures
- Write docstrings in Google style

Example:

```python
def load_config(
    config_input: Union[str, List[str], Dict[str, Any]], 
    overrides: List[str] = None
) -> Dict[str, Any]:
    """Load and merge configuration from YAML files or dict.

    Args:
        config_input: Config file path(s) or dict
        overrides: List of key=value overrides

    Returns:
        Loaded and merged configuration

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config format is invalid
    """
    pass
```

### Documentation

- Add docstrings to all public functions and classes
- Include examples in docstrings when helpful
- Update relevant documentation in `docs/`

## Testing Guidelines

### Test Coverage

- Aim for >80% test coverage
- Test both success and failure cases
- Include edge cases

### Test Organization

```
tests/
├── test_config.py      # Config management tests
├── test_logger.py      # Logging tests
├── test_checkpoint.py  # Checkpoint tests
└── test_integration.py # Integration tests
```

### Test Naming

Use descriptive test names:

```python
def test_load_config_from_yaml_file():
    """Test loading config from YAML file."""
    pass

def test_load_config_with_overrides():
    """Test loading config with CLI overrides."""
    pass

def test_load_config_raises_on_missing_file():
    """Test that load_config raises FileNotFoundError."""
    pass
```

## Documentation

### Building Docs Locally

```bash
# Install docs dependencies
pip install mkdocs mkdocs-material mkdocstrings[python]

# Serve docs locally
mkdocs serve

# Build docs
mkdocs build
```

Visit http://localhost:8000 to view documentation.

### Documentation Structure

```
docs/
├── index.md                    # Home page
├── getting-started/
│   ├── installation.md
│   ├── quickstart.md
│   └── concepts.md
├── guide/
│   ├── configuration.md
│   ├── logging.md
│   └── ...
├── api/
│   ├── config.md
│   ├── logger.md
│   └── ...
└── examples/
    └── ...
```

## Pull Request Guidelines

### PR Checklist

- [ ] Tests pass (`pytest`)
- [ ] Code is formatted (`black`)
- [ ] Code is linted (`ruff`)
- [ ] Type hints are correct (`mypy`)
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated
- [ ] Commit messages are descriptive

### PR Description

Include in your PR description:
- What changes you made
- Why you made them
- How to test them
- Any breaking changes

Example:

```markdown
## Description
Add support for remote config files via HTTP/HTTPS URLs.

## Motivation
Users requested the ability to load configs from remote locations.

## Changes
- Add `load_config_from_url()` function
- Update `load_config()` to detect and handle URLs
- Add tests for URL loading
- Update documentation

## Testing
```bash
pytest tests/test_config.py::test_load_config_from_url
```

## Breaking Changes
None
```

## Release Process

Releases are managed by maintainers:

1. Update version in `pyproject.toml` and `src/expmate/__init__.py`
2. Update `CHANGELOG.md`
3. Create git tag: `git tag v0.2.0`
4. Push tag: `git push origin v0.2.0`
5. Build and publish to PyPI

## Getting Help

- **Questions**: Open a [GitHub Discussion](https://github.com/kunheek/expmate/discussions)
- **Bugs**: Open a [GitHub Issue](https://github.com/kunheek/expmate/issues)
- **Chat**: Join our community (link TBD)

## Code of Conduct

Be respectful and inclusive. We follow the [Contributor Covenant](https://www.contributor-covenant.org/).

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

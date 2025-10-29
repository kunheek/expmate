# Test Suite for ExpMate

This directory contains the test suite for ExpMate.

## Running Tests

### Install Test Dependencies

```bash
pip install -e ".[dev]"
```

### Run All Tests

```bash
pytest
```

### Run Specific Test Files

```bash
pytest tests/test_config.py
pytest tests/test_log.py
```

### Run Specific Test Classes or Functions

```bash
pytest tests/test_config.py::TestConfig
pytest tests/test_config.py::TestConfig::test_dot_notation_access
```

### Run with Coverage

```bash
pytest --cov=src/expmate --cov-report=html
```

This will generate an HTML coverage report in `htmlcov/index.html`.

### Run Fast Tests Only (Skip Slow Tests)

```bash
pytest -m "not slow"
```

### Run in Verbose Mode

```bash
pytest -v
```

## Test Structure

- `conftest.py` - Shared fixtures and pytest configuration
- `test_config.py` - Tests for configuration management
- `test_parser.py` - Tests for CLI argument parsing
- `test_log.py` - Tests for experiment logging
- `test_utils.py` - Tests for utility functions
- `test_checkpoint.py` - Tests for checkpoint management (requires PyTorch)
- `test_git.py` - Tests for Git integration
- `test_torch_mp.py` - Tests for PyTorch multiprocessing utilities
- `test_sweep.py` - Tests for hyperparameter sweep functionality

## Fixtures

Common fixtures are defined in `conftest.py`:

- `temp_dir` - Temporary directory for test files
- `config_file` - Basic YAML config file
- `advanced_config_file` - Config with interpolation
- `sample_metrics_data` - Sample metrics for visualization tests
- `mock_git_repo` - Mock Git repository

## Coverage Goals

We aim for:
- **>80% overall code coverage**
- **100% coverage for core modules** (config, parser, log)
- **>70% coverage for optional modules** (checkpoint, git, torch utilities)

## Writing New Tests

1. Create test file: `test_<module_name>.py`
2. Import the module to test
3. Create test class: `class Test<ClassName>`
4. Write test methods: `def test_<functionality>(self)`
5. Use fixtures from `conftest.py` or create new ones

Example:

```python
import pytest
from expmate.my_module import MyClass

class TestMyClass:
    def test_basic_functionality(self, temp_dir):
        obj = MyClass(dir=temp_dir)
        assert obj.process() == expected_result
```

## Continuous Integration

Tests are automatically run on:
- Every push to main branch
- Every pull request
- Using GitHub Actions (see `.github/workflows/test.yml`)

## Troubleshooting

### PyTorch tests failing

Some tests require PyTorch. Install with:
```bash
pip install -e ".[torch]"
```

### Git tests failing

Git tests require git to be installed and available in PATH.

### Import errors

Make sure ExpMate is installed in development mode:
```bash
pip install -e .
```

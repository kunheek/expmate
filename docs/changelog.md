# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.3] - 2025-10-29

### Added
- **Automatic ETA estimation in `log_metric()`**: Simply pass `total_steps` parameter to automatically track progress and estimate remaining time
- `_format_time()` helper method for human-readable time formatting (e.g., "2h 15m 30s")
- New example `examples/example_auto_eta.py` demonstrating automatic time estimation
- Comprehensive test suite for automatic ETA feature (`tests/test_auto_eta.py`)

### Changed
- Renamed global flag `profile` to `timer` for better consistency with method name
- Renamed environment variable `EM_PROFILE` to `EM_TIMER`
- Updated all documentation and examples to reflect the `profile` → `timer` rename
- Enhanced `log_metric()` with optional `total_steps` and `eta_key` parameters

### Fixed
- All tests updated to use new `timer` flag naming

## [0.1.2] - 2025-10-XX

### Added
- Documentation site with MkDocs
- Comprehensive README for PyPI

## [0.1.0] - 2025-01-XX

### Added
- Initial release
- Configuration management with YAML and CLI overrides
- Experiment logging with structured metrics
- Checkpoint management for PyTorch models
- Distributed training utilities (DDP support)
- WandB and TensorBoard integration
- CLI tools for comparing runs and visualizing metrics
- Git integration for reproducibility
- Hyperparameter sweep utilities
- Examples and documentation

### Features
- Hierarchical configuration with dot notation
- Variable interpolation in configs
- Rank-aware logging for distributed training
- Best model tracking
- Automatic checkpoint cleanup
- DDP-safe run directory creation
- Metrics visualization
- Run comparison tools

[Unreleased]: https://github.com/kunheek/expmate/compare/v0.1.3...HEAD
[0.1.3]: https://github.com/kunheek/expmate/releases/tag/v0.1.3
[0.1.2]: https://github.com/kunheek/expmate/releases/tag/v0.1.2
[0.1.0]: https://github.com/kunheek/expmate/releases/tag/v0.1.0

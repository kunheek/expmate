# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/kunheek/expmate/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/kunheek/expmate/releases/tag/v0.1.0

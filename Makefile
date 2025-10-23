# Makefile for ExpMate development

.PHONY: help install install-dev test test-cov test-fast lint format clean

help:
	@echo "ExpMate Development Commands"
	@echo "============================"
	@echo "install         - Install package in development mode"
	@echo "install-dev     - Install with all development dependencies"
	@echo "test            - Run all tests"
	@echo "test-cov        - Run tests with coverage report"
	@echo "test-fast       - Run fast tests only (skip slow tests)"
	@echo "lint            - Run linters (ruff, mypy)"
	@echo "format          - Format code with black and ruff"
	@echo "clean           - Remove build artifacts and cache files"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev,torch,viz,monitor,tracking]"

test:
	pytest

test-cov:
	pytest --cov=src/expmate --cov-report=html --cov-report=term

test-fast:
	pytest -m "not slow"

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/ examples/
	ruff check --fix src/ tests/

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

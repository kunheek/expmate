"""Tests for utilities module."""

import random

import numpy as np
import pytest

from expmate.utils import get_gpu_devices, set_seed, str2bool


class TestStr2Bool:
    """Test string to boolean conversion."""

    def test_true_values(self):
        assert str2bool("true") is True
        assert str2bool("True") is True
        assert str2bool("yes") is True
        assert str2bool("y") is True
        assert str2bool("1") is True

    def test_false_values(self):
        assert str2bool("false") is False
        assert str2bool("False") is False
        assert str2bool("no") is False
        assert str2bool("n") is False
        assert str2bool("0") is False

    def test_invalid_value(self):
        with pytest.raises(ValueError):
            str2bool("invalid")

    def test_whitespace_handling(self):
        assert str2bool("  true  ") is True
        assert str2bool("  false  ") is False


class TestSetSeed:
    """Test seed setting functionality."""

    def test_set_seed_random(self):
        set_seed(42)
        val1 = random.random()

        set_seed(42)
        val2 = random.random()

        assert val1 == val2

    def test_set_seed_numpy(self):
        set_seed(42)
        val1 = np.random.random()

        set_seed(42)
        val2 = np.random.random()

        assert val1 == val2

    def test_set_seed_torch(self):
        try:
            import torch

            set_seed(42)
            val1 = torch.rand(1).item()

            set_seed(42)
            val2 = torch.rand(1).item()

            assert val1 == val2
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_different_seeds_produce_different_values(self):
        set_seed(42)
        val1 = random.random()

        set_seed(123)
        val2 = random.random()

        assert val1 != val2


class TestGetGPUDevices:
    """Test GPU device detection."""

    def test_get_gpu_devices_returns_list(self):
        devices = get_gpu_devices()
        assert isinstance(devices, list)

    def test_gpu_devices_format(self):
        devices = get_gpu_devices()
        # If GPUs are available, they should be integers
        if devices:
            assert all(isinstance(d, int) for d in devices)

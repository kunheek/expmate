"""Tests for utilities module."""

import random

import numpy as np
import pytest

from expmate.utils import set_seed, get_gpu_devices


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

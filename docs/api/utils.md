# API Reference: Utils

## set_seed()

Set random seed for reproducibility.

```python
from expmate import set_seed

set_seed(42)  # Sets seed for Python, NumPy, and PyTorch
```

**Parameters:**
- `seed` (int): Random seed value

## get_gpu_devices()

Get available GPU devices.

```python
from expmate import get_gpu_devices

devices = get_gpu_devices()
print(f"Available GPUs: {devices}")
```

**Returns:** List of GPU device IDs

See the [source code](https://github.com/kunheek/expmate/blob/main/src/expmate/utils.py) for full API details.

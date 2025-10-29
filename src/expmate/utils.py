import os
import random
from typing import List, Optional

import numpy as np


def str2bool(v: str) -> bool:
    """Convert string to boolean.

    Args:
        v: String value to convert

    Returns:
        Boolean value

    Raises:
        ValueError: If value cannot be converted to boolean

    Examples:
        >>> str2bool("true")
        True
        >>> str2bool("false")
        False
        >>> str2bool("1")
        True
    """
    v = v.strip().lower()
    if isinstance(v, bool):
        return v
    if v in ("yes", "true", "t", "y", "1"):
        return True
    if v in ("no", "false", "f", "n", "0"):
        return False
    raise ValueError(f"Boolean value expected, got: {v}")


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch if available
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_gpu_devices(requested: Optional[str] = None) -> List[int]:
    """Get list of GPU device IDs, respecting CUDA_VISIBLE_DEVICES."""
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible:
        available = [int(x) for x in cuda_visible.split(",") if x.strip()]
    else:
        try:
            import torch

            available = list(range(torch.cuda.device_count()))
        except ImportError:
            available = []

    if requested:
        # Parse requested, e.g., "0,1" or "all"
        if requested.lower() == "all":
            return available
        req_ids = [int(x.strip()) for x in requested.split(",")]
        return [d for d in req_ids if d in available]
    return available

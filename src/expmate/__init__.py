import os

__version__ = "0.1.1"

# Import submodules to make them available
from . import config, git, logger, monitor, parser, tracking, utils

# Import optional modules
try:
    from . import torch
except ImportError:
    torch = None  # type: ignore

# Import commonly used classes and functions
from .config import Config
from .logger import ExperimentLogger
from .parser import ConfigArgumentParser, parse_config
from .utils import get_gpu_devices, set_seed

# Export public API
__all__ = [
    "__version__",
    "Config",
    "ConfigArgumentParser",
    "ExperimentLogger",
    "config",
    "get_gpu_devices",
    "git",
    "logger",
    "monitor",
    "parse_config",
    "parser",
    "set_seed",
    "torch",
    "tracking",
    "utils",
]

# Debug mode flag
debug = parser.str2bool(os.environ.get("EM_DEBUG", "0"))

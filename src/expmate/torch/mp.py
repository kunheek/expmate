import functools
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import torch
from torch import distributed as dist


def is_dist_available() -> bool:
    """Check if distributed training is available."""
    return dist.is_available()


def is_dist_initialized() -> bool:
    """Check if distributed training is initialized."""
    return is_dist_available() and dist.is_initialized()


def get_rank() -> int:
    """Get the rank of the current process.

    Returns:
        int: Rank of current process (0 if not in distributed mode)
    """
    if not is_dist_initialized():
        return 0
    return dist.get_rank()


def get_world_size() -> int:
    """Get the total number of processes.

    Returns:
        int: Total number of processes (1 if not in distributed mode)
    """
    if not is_dist_initialized():
        return 1
    return dist.get_world_size()


def is_main_process() -> bool:
    """Check if current process is the main process (rank 0).

    Returns:
        bool: True if rank 0 or not in distributed mode
    """
    return get_rank() == 0


def get_local_rank() -> int:
    """Get the local rank of the current process on the current node.

    Returns:
        int: Local rank (0 if not in distributed mode)
    """
    if not is_dist_initialized():
        return int(os.environ.get("LOCAL_RANK", 0))
    # If using torchrun, LOCAL_RANK is set
    return int(os.environ.get("LOCAL_RANK", get_rank()))


def setup_ddp(
    backend: Optional[str] = None,
    init_method: str = "env://",
    timeout_minutes: int = 30,
) -> Tuple[int, int, int]:
    """Initialize distributed training with proper device setup.

    This function handles the complete DDP setup including:
    - Reading environment variables set by torchrun
    - Initializing the process group
    - Setting the correct CUDA device

    Args:
        backend: Backend to use ('nccl', 'gloo', or None for auto-detect).
                 If None, uses 'nccl' if CUDA is available, else 'gloo'.
        init_method: Initialization method (default: 'env://' for torchrun)
        timeout_minutes: Timeout for process group initialization

    Returns:
        Tuple[int, int, int]: (rank, local_rank, world_size)

    Example:
        >>> # When using torchrun
        >>> rank, local_rank, world_size = setup_ddp()
        >>> device = torch.device(
        ...     f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
        ... )
        >>> model = MyModel().to(device)
        >>> if world_size > 1:
        >>>     model = DDP(
        ...         model,
        ...         device_ids=[local_rank] if torch.cuda.is_available() else None
        ...     )
    """
    # Get distributed info from environment (set by torchrun)
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Only initialize if world_size > 1
    if world_size > 1:
        # Auto-select backend if not specified
        if backend is None:
            backend = "nccl" if torch.cuda.is_available() else "gloo"

        # Initialize process group
        if not is_dist_initialized():
            dist.init_process_group(
                backend=backend,
                init_method=init_method,
                timeout=timedelta(minutes=timeout_minutes),
            )

        # Set device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size


def cleanup_ddp() -> None:
    """Clean up the distributed process group.

    Should be called at the end of training to properly clean up resources.
    Safe to call even if DDP was not initialized.

    Example:
        >>> try:
        >>>     rank, local_rank, world_size = setup_ddp()
        >>>     # ... training code ...
        >>> finally:
        >>>     cleanup_ddp()
    """
    if is_dist_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    """Synchronize all processes.

    Only has effect in distributed mode. Safe to call even if not distributed.
    """
    if is_dist_initialized():
        dist.barrier()


def main_process_only(func: Optional[Callable] = None, default: Any = None):
    """Decorator to run a function only on the main process (rank 0).

    Args:
        func: Function to decorate
        default: Default value to return on non-main processes

    Usage:
        @main_process_only
        def save_checkpoint():
            torch.save(model.state_dict(), 'checkpoint.pt')

        # With default return value
        @main_process_only(default=0)
        def count_parameters():
            return sum(p.numel() for p in model.parameters())
    """

    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            if is_main_process():
                return f(*args, **kwargs)
            return default

        return wrapper

    # Handle both @main_process_only and @main_process_only(default=...)
    if func is None:
        return decorator
    return decorator(func)


def reduce_dict(input_dict: dict, average: bool = True) -> dict:
    """Reduce dictionary of tensors across all processes.

    Args:
        input_dict: Dictionary with tensor values
        average: If True, compute average; if False, sum

    Returns:
        dict: Reduced dictionary (only valid on rank 0)
    """
    if not is_dist_initialized():
        return input_dict

    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])

        values = torch.stack(values, dim=0)
        dist.all_reduce(values)

        if average:
            values /= world_size

        return {k: v.item() for k, v in zip(names, values)}


def gather_object(obj: Any) -> list:
    """Gather objects from all processes to rank 0.

    Args:
        obj: Object to gather (must be picklable)

    Returns:
        list: List of objects from all ranks (only valid on rank 0)
    """
    if not is_dist_initialized():
        return [obj]

    world_size = get_world_size()
    if world_size < 2:
        return [obj]

    gathered = [None] * world_size
    dist.all_gather_object(gathered, obj)
    return gathered


def mp_print(*args, **kwargs):
    """Print only from the main process (rank 0).

    Usage:
        mp_print("Training started")  # Only prints on rank 0
    """
    if is_main_process():
        print(*args, **kwargs)


def create_shared_run_dir(
    base_dir: Union[str, Path], run_id: Optional[str] = None
) -> Path:
    """Create a run directory that is shared across all distributed processes.

    This ensures that all processes use the same directory, even when called
    at slightly different times. Only rank 0 creates the directory, then
    all processes synchronize.

    Args:
        base_dir: Base directory for runs (e.g., 'runs/experiment')
        run_id: Optional run ID. If None, will be created only on rank 0

    Returns:
        Path: The shared run directory path

    Example:
        >>> # In distributed training
        >>> run_dir = create_shared_run_dir('runs', run_id=config.get('run_id'))
        >>> logger = ExperimentLogger(run_dir)  # All ranks use same directory
    """
    base_dir = Path(base_dir)

    if is_main_process():
        # Only rank 0 creates the directory and determines run_id
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S_exp")

        run_dir = base_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Store the path for broadcasting
        run_dir_str = str(run_dir.absolute())
    else:
        run_dir_str = None

    # Broadcast the directory path to all processes
    if is_dist_initialized():
        # Convert to list for broadcasting
        run_dir_list = [run_dir_str]
        dist.broadcast_object_list(run_dir_list, src=0)
        run_dir_str = run_dir_list[0]

    # Synchronize all processes before returning
    barrier()

    return Path(run_dir_str)

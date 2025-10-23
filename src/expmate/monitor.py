import json
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import psutil

try:
    import torch
except ImportError:
    torch = None


class ResourceMonitor:
    """Monitor system resources (GPU, CPU, memory) during training.

    Example:
        monitor = ResourceMonitor(run_dir, interval=10)
        monitor.start()
        # ... training code ...
        monitor.stop()
    """

    def __init__(
        self,
        run_dir: Path,
        interval: float = 10.0,
        track_gpu: bool = True,
        track_cpu: bool = True,
        track_memory: bool = True,
    ):
        self.run_dir = Path(run_dir)
        self.interval = interval
        self.track_gpu = track_gpu
        self.track_cpu = track_cpu
        self.track_memory = track_memory

        self.monitoring = False
        self.thread: Optional[threading.Thread] = None

        self.log_file = self.run_dir / "resource_usage.jsonl"

    def start(self):
        """Start monitoring in background thread."""
        if self.monitoring:
            return

        self.monitoring = True
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()
        print(f"📊 Resource monitoring started (interval={self.interval}s)")

    def stop(self):
        """Stop monitoring."""
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=self.interval + 1)
        print("📊 Resource monitoring stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            stats = self.get_current_stats()
            self._log_stats(stats)
            time.sleep(self.interval)

    def get_current_stats(self) -> Dict:
        """Get current resource usage statistics."""
        stats = {"timestamp": time.time()}

        # GPU stats
        if self.track_gpu:
            try:
                if torch is not None and torch.cuda.is_available():
                    gpu_stats = []
                    for i in range(torch.cuda.device_count()):
                        gpu_stats.append(
                            {
                                "device": i,
                                "allocated_mb": torch.cuda.memory_allocated(i) / 1e6,
                                "reserved_mb": torch.cuda.memory_reserved(i) / 1e6,
                                "max_allocated_mb": torch.cuda.max_memory_allocated(i)
                                / 1e6,
                            }
                        )
                    stats["gpu"] = gpu_stats
            except Exception as e:
                stats["gpu_error"] = str(e)

        # CPU stats
        if self.track_cpu:
            try:
                stats["cpu"] = {
                    "percent": psutil.cpu_percent(interval=0.1),
                    "count": psutil.cpu_count(),
                }
            except Exception as e:
                stats["cpu_error"] = str(e)

        # Memory stats
        if self.track_memory:
            try:
                mem = psutil.virtual_memory()
                stats["memory"] = {
                    "total_gb": mem.total / 1e9,
                    "available_gb": mem.available / 1e9,
                    "used_gb": mem.used / 1e9,
                    "percent": mem.percent,
                }
            except Exception as e:
                stats["memory_error"] = str(e)

        return stats

    def _log_stats(self, stats: Dict):
        """Log statistics to file."""
        with open(self.log_file, "a") as f:
            f.write(json.dumps(stats) + "\n")

    def get_summary(self) -> Dict:
        """Get summary statistics from logged data."""
        if not self.log_file.exists():
            return {}

        all_stats = []
        with open(self.log_file) as f:
            for line in f:
                all_stats.append(json.loads(line))

        if not all_stats:
            return {}

        summary = {}

        # GPU summary
        if "gpu" in all_stats[0]:
            gpu_allocated = []
            for stat in all_stats:
                if "gpu" in stat and stat["gpu"]:
                    total_alloc = sum(g["allocated_mb"] for g in stat["gpu"])
                    gpu_allocated.append(total_alloc)

            if gpu_allocated:
                summary["gpu_allocated_mb"] = {
                    "mean": sum(gpu_allocated) / len(gpu_allocated),
                    "max": max(gpu_allocated),
                    "min": min(gpu_allocated),
                }

        # CPU summary
        if "cpu" in all_stats[0]:
            cpu_percent = [s["cpu"]["percent"] for s in all_stats if "cpu" in s]
            if cpu_percent:
                summary["cpu_percent"] = {
                    "mean": sum(cpu_percent) / len(cpu_percent),
                    "max": max(cpu_percent),
                    "min": min(cpu_percent),
                }

        # Memory summary
        if "memory" in all_stats[0]:
            mem_percent = [s["memory"]["percent"] for s in all_stats if "memory" in s]
            if mem_percent:
                summary["memory_percent"] = {
                    "mean": sum(mem_percent) / len(mem_percent),
                    "max": max(mem_percent),
                    "min": min(mem_percent),
                }

        return summary

    def print_summary(self):
        """Print summary statistics."""
        summary = self.get_summary()

        if not summary:
            print("No resource monitoring data available")
            return

        print("\n" + "=" * 60)
        print("Resource Usage Summary")
        print("=" * 60)

        if "gpu_allocated_mb" in summary:
            gpu = summary["gpu_allocated_mb"]
            print("\nGPU Memory Allocated:")
            print(f"  Mean: {gpu['mean']:.1f} MB")
            print(f"  Max:  {gpu['max']:.1f} MB")
            print(f"  Min:  {gpu['min']:.1f} MB")

        if "cpu_percent" in summary:
            cpu = summary["cpu_percent"]
            print("\nCPU Usage:")
            print(f"  Mean: {cpu['mean']:.1f}%")
            print(f"  Max:  {cpu['max']:.1f}%")
            print(f"  Min:  {cpu['min']:.1f}%")

        if "memory_percent" in summary:
            mem = summary["memory_percent"]
            print("\nMemory Usage:")
            print(f"  Mean: {mem['mean']:.1f}%")
            print(f"  Max:  {mem['max']:.1f}%")
            print(f"  Min:  {mem['min']:.1f}%")

        print("=" * 60 + "\n")

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


def log_system_info(run_dir: Path):
    """Log system information to run directory.

    Args:
        run_dir: Run directory
    """
    run_dir = Path(run_dir)
    info = {"timestamp": time.time()}

    # Python version
    info["python_version"] = sys.version

    # PyTorch version and CUDA info
    if torch is not None:
        info["pytorch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = torch.backends.cudnn.version()
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    else:
        info["pytorch_installed"] = False

    # System info
    info["cpu_count"] = psutil.cpu_count()
    info["memory_total_gb"] = psutil.virtual_memory().total / 1e9

    # Hostname
    info["hostname"] = socket.gethostname()

    # Save to file
    info_file = run_dir / "system_info.json"
    with open(info_file, "w") as f:
        json.dump(info, f, indent=2)

    return info

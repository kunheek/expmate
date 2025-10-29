import argparse
import json
from pathlib import Path
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import polars as pl


def plot_metrics(
    run_dirs: Union[str, Path, List[Union[str, Path]]],
    metrics: Optional[List[str]] = None,
    output_file: Optional[str] = None,
    show: bool = True,
    style: str = "default",
):
    """Plot metrics from one or more runs.

    Args:
        run_dirs: Single run directory or list of run directories
        metrics: List of metrics to plot (None = all)
        output_file: Save plot to file (e.g., 'plot.png')
        show: Whether to display the plot
        style: Plot style ('default', 'seaborn', 'ggplot')
    """
    # Handle single run_dir
    if isinstance(run_dirs, (str, Path)):
        run_dirs = [run_dirs]

    # Set style
    if style != "default":
        plt.style.use(style)

    # Load data from all runs
    all_data = []
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        metrics_file = run_dir / "metrics.csv"

        if not metrics_file.exists():
            print(f"⚠️  No metrics found in {run_dir}")
            continue

        df = pl.read_csv(metrics_file)
        df = df.with_columns(run_id=pl.lit(run_dir.name))
        all_data.append(df)

    if not all_data:
        print("❌ No metrics data found")
        return

    combined_df = pl.concat(all_data)

    # Filter metrics if specified
    if metrics:
        combined_df = combined_df.filter(pl.col("name").is_in(metrics))

    # Get unique metrics and splits
    unique_metrics = combined_df["name"].unique()
    unique_splits = combined_df["split"].unique()

    # Create subplots
    n_metrics = len(unique_metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_metrics == 1:
        axes = [axes]
    elif n_rows > 1 and n_cols > 1:
        axes = axes.flatten()
    else:
        axes = list(axes) if n_rows > 1 or n_cols > 1 else [axes]

    # Plot each metric
    for idx, metric_name in enumerate(sorted(unique_metrics)):
        ax = axes[idx]

        for run_id in combined_df["run_id"].unique():
            for split in unique_splits:
                data = combined_df.filter(
                    (pl.col("name") == metric_name)
                    & (pl.col("run_id") == run_id)
                    & (pl.col("split") == split)
                )

                if data.height > 0:
                    label = f"{run_id}_{split}" if len(run_dirs) > 1 else split
                    ax.plot(
                        data["step"],
                        data["value"],
                        label=label,
                        marker="o",
                        markersize=3,
                    )

        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.set_title(metric_name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"📊 Plot saved to {output_file}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_resource_usage(
    run_dir: Union[str, Path],
    output_file: Optional[str] = None,
    show: bool = True,
):
    """Plot resource usage over time.

    Args:
        run_dir: Run directory
        output_file: Save plot to file
        show: Whether to display the plot
    """
    run_dir = Path(run_dir)
    resource_file = run_dir / "resource_usage.jsonl"

    if not resource_file.exists():
        print(f"❌ No resource monitoring data found in {run_dir}")
        return

    # Load data
    data = []
    with open(resource_file) as f:
        for line in f:
            data.append(json.loads(line))

    if not data:
        print("❌ No resource data found")
        return

    # Extract time series
    timestamps = [d["timestamp"] for d in data]
    start_time = timestamps[0]
    elapsed_times = [(t - start_time) / 60 for t in timestamps]  # Convert to minutes

    # Prepare subplots
    n_plots = 0
    has_gpu = "gpu" in data[0]
    has_cpu = "cpu" in data[0]
    has_memory = "memory" in data[0]

    n_plots = sum([has_gpu, has_cpu, has_memory])

    if n_plots == 0:
        print("❌ No resource data to plot")
        return

    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # Plot GPU memory
    if has_gpu:
        ax = axes[plot_idx]
        plot_idx += 1

        for i in range(len(data[0]["gpu"])):
            gpu_mem = [d["gpu"][i]["allocated_mb"] for d in data if "gpu" in d]
            ax.plot(elapsed_times, gpu_mem, label=f"GPU {i}", marker="o", markersize=2)

        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Memory (MB)")
        ax.set_title("GPU Memory Usage")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Plot CPU usage
    if has_cpu:
        ax = axes[plot_idx]
        plot_idx += 1

        cpu_percent = [d["cpu"]["percent"] for d in data if "cpu" in d]
        ax.plot(
            elapsed_times[: len(cpu_percent)], cpu_percent, marker="o", markersize=2
        )

        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("CPU %")
        ax.set_title("CPU Usage")
        ax.grid(True, alpha=0.3)

    # Plot memory usage
    if has_memory:
        ax = axes[plot_idx]
        plot_idx += 1

        mem_percent = [d["memory"]["percent"] for d in data if "memory" in d]
        ax.plot(
            elapsed_times[: len(mem_percent)], mem_percent, marker="o", markersize=2
        )

        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Memory %")
        ax.set_title("System Memory Usage")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches="tight")
        print(f"📊 Resource plot saved to {output_file}")

    if show:
        plt.show()
    else:
        plt.close()


def setup_visualize_parser(subparsers):
    """Setup the visualize subcommand parser."""
    viz_parser = subparsers.add_parser(
        "viz",
        help="Visualize experiment metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot all metrics from multiple runs
  expmate viz runs/exp1 runs/exp2 runs/exp3

  # Plot specific metrics
  expmate viz runs/exp_* --metrics loss accuracy

  # Save plot to file
  expmate viz runs/exp1 --output plot.png

  # Plot resource usage
  expmate viz runs/exp1 --resources
        """,
    )
    viz_parser.add_argument("run_dirs", nargs="+", help="Run directories to plot")
    viz_parser.add_argument(
        "--metrics", "-m", nargs="+", help="Specific metrics to plot"
    )
    viz_parser.add_argument("--output", "-o", help="Save plot to file (e.g., plot.png)")
    viz_parser.add_argument("--no-show", action="store_true", help="Don't display plot")
    viz_parser.add_argument(
        "--style",
        default="default",
        choices=["default", "seaborn", "ggplot"],
        help="Plot style",
    )
    viz_parser.add_argument(
        "--resources",
        "-r",
        action="store_true",
        help="Plot resource usage instead of metrics",
    )


def run_visualize_command(args):
    """Run the visualize command."""
    if args.resources:
        if len(args.run_dirs) > 1:
            print("⚠️  Resource plotting only supports single run directory")
            print(f"Plotting resources for: {args.run_dirs[0]}")
        plot_resource_usage(
            args.run_dirs[0], output_file=args.output, show=not args.no_show
        )
    else:
        plot_metrics(
            args.run_dirs,
            metrics=args.metrics,
            output_file=args.output,
            show=not args.no_show,
            style=args.style,
        )

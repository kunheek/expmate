import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

import polars as pl
import yaml


def load_run_info(run_dir: Union[str, Path]) -> Dict:
    """Load information about a single run.

    Args:
        run_dir: Path to run directory

    Returns:
        Dict with run information including config, metrics, git info
    """
    run_dir = Path(run_dir)
    info = {"run_dir": str(run_dir), "run_id": run_dir.name}

    # Load config
    config_file = run_dir / "run.yaml"
    if config_file.exists():
        with open(config_file) as f:
            info["config"] = yaml.safe_load(f)

    # Load best metrics
    best_file = run_dir / "best.json"
    if best_file.exists():
        with open(best_file) as f:
            info["best_metrics"] = json.load(f)

    # Load final metrics from CSV
    metrics_file = run_dir / "metrics.csv"
    if metrics_file.exists():
        df = pl.read_csv(metrics_file)
        if not df.is_empty():
            # Get last values for each metric
            final_metrics = {}
            for name in df["name"].unique():
                metric_df = df.filter(pl.col("name") == name)
                if not metric_df.is_empty():
                    final_metrics[name] = metric_df.tail(1)["value"][0]
            info["final_metrics"] = final_metrics

    # Load git info
    git_file = run_dir / "git_info.json"
    if git_file.exists():
        with open(git_file) as f:
            info["git"] = json.load(f)

    return info


def compare_runs(
    run_dirs: List[Union[str, Path]],
    metrics: Optional[List[str]] = None,
    show_config: bool = True,
    show_git: bool = False,
) -> pl.DataFrame:
    """Compare multiple experiment runs.

    Args:
        run_dirs: List of run directories to compare
        metrics: Specific metrics to compare (None = all)
        show_config: Whether to include config values
        show_git: Whether to include git info

    Returns:
        DataFrame with comparison
    """
    runs_data = []

    for run_dir in run_dirs:
        info = load_run_info(run_dir)
        row = {"run_id": info["run_id"], "run_dir": info["run_dir"]}

        # Add metrics
        if "best_metrics" in info:
            for metric_name, metric_info in info["best_metrics"].items():
                if metrics is None or metric_name in metrics:
                    row[f"best_{metric_name}"] = metric_info.get("value")
                    row[f"best_{metric_name}_step"] = metric_info.get("step")

        if "final_metrics" in info:
            for metric_name, value in info["final_metrics"].items():
                if metrics is None or metric_name in metrics:
                    row[f"final_{metric_name}"] = value

        # Add config
        if show_config and "config" in info:
            config = info["config"]
            # Flatten config
            for key, value in _flatten_dict(config).items():
                row[f"config.{key}"] = value

        # Add git info
        if show_git and "git" in info:
            row["git_sha"] = info["git"]["sha_short"]
            row["git_dirty"] = info["git"]["dirty"]

        runs_data.append(row)

    return pl.DataFrame(runs_data)


def _flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict:
    """Flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def find_config_diff(run_dirs: List[Union[str, Path]]) -> Dict[str, List]:
    """Find configuration differences between runs.

    Args:
        run_dirs: List of run directories

    Returns:
        Dict mapping config keys to their values across runs
    """
    all_configs = []
    for run_dir in run_dirs:
        info = load_run_info(run_dir)
        if "config" in info:
            all_configs.append(_flatten_dict(info["config"]))

    if not all_configs:
        return {}

    # Find keys that differ
    all_keys = set()
    for config in all_configs:
        all_keys.update(config.keys())

    diffs = {}
    for key in sorted(all_keys):
        values = [config.get(key, None) for config in all_configs]
        # Only include if values differ
        if len(set(str(v) for v in values)) > 1:
            diffs[key] = values

    return diffs


def print_comparison(
    run_dirs: List[Union[str, Path]],
    metrics: Optional[List[str]] = None,
    show_all_config: bool = False,
):
    """Print a formatted comparison of runs.

    Args:
        run_dirs: List of run directories to compare
        metrics: Specific metrics to compare (None = all)
        show_all_config: If False, only show differing config values
    """
    print(f"\n{'=' * 80}")
    print(f"Comparing {len(run_dirs)} runs")
    print(f"{'=' * 80}\n")

    # Load run info
    runs_info = [load_run_info(rd) for rd in run_dirs]

    # Print run IDs
    print("Run IDs:")
    for i, info in enumerate(runs_info):
        print(f"  [{i}] {info['run_id']}")
    print()

    # Print config differences
    if show_all_config:
        print("Configuration:")
        df = compare_runs(run_dirs, show_config=True, show_git=False)
        config_cols = [c for c in df.columns if c.startswith("config.")]
        if config_cols:
            for col in sorted(config_cols):
                print(f"  {col}:")
                for i, val in enumerate(df[col]):
                    print(f"    [{i}] {val}")
        print()
    else:
        diffs = find_config_diff(run_dirs)
        if diffs:
            print("Configuration Differences:")
            for key, values in diffs.items():
                print(f"  {key}:")
                for i, val in enumerate(values):
                    print(f"    [{i}] {val}")
            print()
        else:
            print("No configuration differences found.\n")

    # Print metrics comparison
    print("Metrics Comparison:")
    df = compare_runs(run_dirs, metrics=metrics, show_config=False, show_git=False)

    # Show best metrics
    best_cols = [
        c for c in df.columns if c.startswith("best_") and not c.endswith("_step")
    ]
    if best_cols:
        print("\n  Best Metrics:")
        for col in sorted(best_cols):
            metric_name = col.replace("best_", "")
            print(f"    {metric_name}:")
            for i, val in enumerate(df[col]):
                step_col = f"best_{metric_name}_step"
                step = df[step_col].iloc[i] if step_col in df.columns else "?"
                print(f"      [{i}] {val:.4f} (step {step})")

    # Show final metrics
    final_cols = [c for c in df.columns if c.startswith("final_")]
    if final_cols:
        print("\n  Final Metrics:")
        for col in sorted(final_cols):
            metric_name = col.replace("final_", "")
            print(f"    {metric_name}:")
            for i, val in enumerate(df[col]):
                print(f"      [{i}] {val:.4f}")

    print(f"\n{'=' * 80}\n")


def setup_compare_parser(subparsers):
    """Setup the compare subcommand parser."""
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare multiple experiment runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare runs
  expmate compare runs/exp1 runs/exp2 runs/exp3

  # Compare specific metrics
  expmate compare runs/exp_* --metrics loss accuracy

  # Show all config (not just diffs)
  expmate compare runs/exp1 runs/exp2 --all-config

  # Export to CSV
  expmate compare runs/exp_* --export comparison.csv
        """,
    )
    compare_parser.add_argument(
        "run_dirs", nargs="+", help="Run directories to compare"
    )
    compare_parser.add_argument(
        "--metrics", "-m", nargs="+", help="Specific metrics to compare"
    )
    compare_parser.add_argument(
        "--all-config", action="store_true", help="Show all config, not just diffs"
    )
    compare_parser.add_argument("--export", "-e", help="Export comparison to CSV file")


def run_compare_command(args):
    """Run the compare command."""
    if args.export:
        df = compare_runs(args.run_dirs, metrics=args.metrics)
        df.write_csv(args.export)
        print(f"✅ Comparison exported to {args.export}")
    else:
        print_comparison(
            args.run_dirs, metrics=args.metrics, show_all_config=args.all_config
        )

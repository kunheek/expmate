import argparse
import copy
import itertools
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def generate_sweep_configs(
    base_config: Dict[str, Any],
    sweep_params: Dict[str, List[Any]],
) -> List[Dict[str, Any]]:
    """Generate all configurations for a grid search.

    Args:
        base_config: Base configuration dictionary
        sweep_params: Dictionary mapping parameter paths to lists of values.
                     Example: {"training.lr": [0.001, 0.01],
                               "model.hidden_dim": [128, 256]}

    Returns:
        List of configuration dictionaries
    """
    # Get all parameter names and their values
    param_names = list(sweep_params.keys())
    param_values = [sweep_params[name] for name in param_names]

    # Generate all combinations
    configs = []
    for values in itertools.product(*param_values):
        # Deep copy the base config to avoid mutations
        config = copy.deepcopy(base_config)

        # Apply sweep parameters
        for param_name, value in zip(param_names, values):
            # Convert dot notation to nested dict
            keys = param_name.split(".")
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value

        configs.append(config)

    return configs


def run_sweep(
    command_template: str,
    sweep_params: Dict[str, List[Any]],
    base_config_file: Optional[str] = None,
    runs_dir: str = "runs",
    sweep_name: Optional[str] = None,
    dry_run: bool = False,
    parallel: bool = False,
    max_parallel: int = 4,
):
    """Run a hyperparameter sweep.

    Args:
        command_template: Command to run, e.g.:
                         "python train.py {config}"
                         "torchrun --nproc_per_node=2 train.py {config}"
                         "bash train.sh 2 {config}"
        sweep_params: Parameters to sweep over
        base_config_file: Base config file (optional)
        runs_dir: Directory to save runs
        sweep_name: Name for this sweep
        dry_run: If True, only print commands without running
        parallel: Run experiments in parallel
        max_parallel: Maximum number of parallel runs

    Example:
        >>> run_sweep(
        ...     command_template="python train.py {config}",
        ...     sweep_params={
        ...         "training.lr": [0.001, 0.01, 0.1],
        ...         "model.hidden_dim": [128, 256],
        ...     },
        ...     base_config_file="conf/default.yaml",
        ...     sweep_name="lr_hidden_sweep"
        ... )
    """
    # Create sweep directory
    if sweep_name is None:
        sweep_name = datetime.now().strftime("sweep_%Y%m%d_%H%M%S")

    sweep_dir = Path(runs_dir) / sweep_name
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Load base config if provided
    base_config = {}
    if base_config_file:
        with open(base_config_file) as f:
            base_config = yaml.safe_load(f)

    # Generate all configurations
    configs = generate_sweep_configs(base_config, sweep_params)

    print(f"\n{'=' * 80}")
    print(f"Hyperparameter Sweep: {sweep_name}")
    print(f"{'=' * 80}")
    print(f"Total experiments: {len(configs)}")
    print(f"Sweep directory: {sweep_dir}")
    print("\nSweep parameters:")
    for param, values in sweep_params.items():
        print(f"  {param}: {values}")
    print(f"{'=' * 80}\n")

    # Save sweep configuration
    sweep_info = {
        "sweep_name": sweep_name,
        "command_template": command_template,
        "sweep_params": sweep_params,
        "base_config_file": base_config_file,
        "num_experiments": len(configs),
        "timestamp": datetime.now().isoformat(),
    }

    with open(sweep_dir / "sweep_info.json", "w") as f:
        json.dump(sweep_info, f, indent=2)

    # Run each configuration
    results = []

    for idx, config in enumerate(configs):
        exp_name = f"exp_{idx:03d}"
        exp_dir = sweep_dir / exp_name
        exp_dir.mkdir(exist_ok=True)

        # Save config
        config_file = exp_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(config, f)

        # Build command
        command = command_template.format(config=str(config_file))

        # Add run_dir override if not in command
        if "run_dir" not in command:
            command += f" run_dir={exp_dir}"

        print(f"\n[{idx + 1}/{len(configs)}] Running: {exp_name}")
        print(f"Command: {command}")

        # Print sweep parameters for this run
        print("Parameters:")
        for param in sweep_params:
            keys = param.split(".")
            current = config
            for key in keys:
                current = current.get(key, {})
            print(f"  {param}: {current}")

        if dry_run:
            results.append({"exp_name": exp_name, "status": "dry_run"})
            continue

        # Run command
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=Path.cwd(),
            )

            success = result.returncode == 0
            results.append(
                {
                    "exp_name": exp_name,
                    "config": config,
                    "command": command,
                    "returncode": result.returncode,
                    "status": "success" if success else "failed",
                }
            )

            if success:
                print("✅ Completed successfully")
            else:
                print(f"❌ Failed with return code {result.returncode}")
                if result.stderr:
                    print(f"Error: {result.stderr[:500]}")

        except Exception as e:
            print(f"❌ Exception: {e}")
            results.append(
                {
                    "exp_name": exp_name,
                    "config": config,
                    "command": command,
                    "status": "exception",
                    "error": str(e),
                }
            )

    # Save results
    results_file = sweep_dir / "sweep_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n{'=' * 80}")
    print("Sweep Summary")
    print(f"{'=' * 80}")

    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] in ["failed", "exception"])

    print(f"Total experiments: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nResults saved to: {sweep_dir}")
    print(f"{'=' * 80}\n")

    return results


def setup_sweep_parser(subparsers):
    """Setup the sweep subcommand parser."""
    sweep_parser = subparsers.add_parser(
        "sweep",
        help="Run hyperparameter sweeps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic sweep
  expmate sweep "python train.py {config}" \\
    --config config.yaml \\
    --sweep "training.lr=[0.001,0.01]" "model.hidden_dim=[128,256]"

  # With torchrun
  expmate sweep "torchrun --nproc_per_node=2 train.py {config}" \\
    --config config.yaml \\
    --sweep "training.lr=[0.001,0.01]"

  # Dry run to see commands
  expmate sweep "python train.py {config}" \\
    --config config.yaml \\
    --sweep "training.lr=[0.001,0.01]" \\
    --dry-run
        """,
    )
    sweep_parser.add_argument(
        "command", help="Command template with {config} placeholder"
    )
    sweep_parser.add_argument(
        "--config", "-c", required=True, help="Base configuration file"
    )
    sweep_parser.add_argument(
        "--sweep",
        "-s",
        nargs="+",
        required=True,
        help="Sweep parameters in format: param=[val1,val2,...]",
    )
    sweep_parser.add_argument("--name", "-n", help="Sweep name")
    sweep_parser.add_argument(
        "--runs-dir", "-d", default="runs", help="Directory for runs"
    )
    sweep_parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without running"
    )


def run_sweep_command(args):
    """Run the sweep command."""
    # Parse sweep parameters
    sweep_params = {}

    for spec in args.sweep:
        match = re.match(r"(\S+)=\[([^\]]+)\]", spec)
        if not match:
            print(f"❌ Invalid sweep spec: {spec}")
            print("   Expected format: param=[val1,val2,...]")
            continue

        key, values_str = match.groups()
        values = []
        for v in values_str.split(","):
            v = v.strip()
            try:
                values.append(int(v))
            except ValueError:
                try:
                    values.append(float(v))
                except ValueError:
                    values.append(v)

        sweep_params[key] = values

    if not sweep_params:
        print("❌ No valid sweep parameters provided")
        sys.exit(1)

    run_sweep(
        command_template=args.command,
        sweep_params=sweep_params,
        base_config_file=args.config,
        sweep_name=args.name,
        runs_dir=args.runs_dir,
        dry_run=args.dry_run,
    )

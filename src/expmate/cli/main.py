#!/usr/bin/env python3
import argparse
import sys


def main():
    """Main CLI entrypoint for ExpMate."""
    parser = argparse.ArgumentParser(
        prog="expmate",
        description="ExpMate - ML Research Boilerplate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  sweep        Run hyperparameter sweeps
  viz          Visualize experiment metrics
  compare      Compare multiple experiment runs

Examples:
  expmate sweep "python train.py {config}" --config config.yaml --sweep "lr=[0.001,0.01]"
  expmate viz runs/exp_* --metrics loss accuracy
  expmate compare runs/exp1 runs/exp2 runs/exp3
        """,
    )

    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Import command setup functions
    from .sweep import setup_sweep_parser
    from .visualize import setup_visualize_parser
    from .compare import setup_compare_parser

    # Setup subcommands
    setup_sweep_parser(subparsers)
    setup_visualize_parser(subparsers)
    setup_compare_parser(subparsers)

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Dispatch to appropriate command
    if args.command == "sweep":
        from .sweep import run_sweep_command

        run_sweep_command(args)
    elif args.command in ("viz", "visualize"):
        from .visualize import run_visualize_command

        run_visualize_command(args)
    elif args.command == "compare":
        from .compare import run_compare_command

        run_compare_command(args)


if __name__ == "__main__":
    main()

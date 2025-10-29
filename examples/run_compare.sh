#!/bin/bash
# Compare multiple experiment runs using ExpMate CLI
#
# This script demonstrates how to compare experiment results,
# including metrics, configurations, and best values.
#
# Usage:
#   ./run_compare.sh [run_directories...]
#
# Examples:
#   ./run_compare.sh runs/exp1 runs/exp2 runs/exp3
#   ./run_compare.sh runs/mnist_*
#   ./run_compare.sh runs/sweep_*/exp*

set -e

# Check if arguments provided
if [ $# -eq 0 ]; then
  echo "Usage: $0 <run_dir1> <run_dir2> [run_dir3] ..."
  echo
  echo "Examples:"
  echo "  $0 runs/exp1 runs/exp2 runs/exp3"
  echo "  $0 runs/mnist_*"
  echo "  $0 runs/sweep_*/exp*"
  exit 1
fi

echo "=================================================="
echo "Comparing Experiment Runs"
echo "=================================================="
echo "Runs to compare: $#"
echo "=================================================="
echo

# Compare runs - show configuration differences
echo "📊 Comparing configurations..."
expmate compare "$@" --show-config

echo
echo "=================================================="

# Compare runs - show metrics
echo "📈 Comparing metrics..."
expmate compare "$@" --metrics loss accuracy

echo
echo "=================================================="

# Compare runs - export to CSV
OUTPUT_FILE="comparison_$(date +%Y%m%d_%H%M%S).csv"
echo "💾 Exporting comparison to $OUTPUT_FILE..."
expmate compare "$@" --output "$OUTPUT_FILE"

echo
echo "=================================================="
echo "Comparison Complete!"
echo "=================================================="
echo "Results exported to: $OUTPUT_FILE"
echo "=================================================="

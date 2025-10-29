#!/bin/bash
# Visualize experiment metrics using ExpMate CLI
#
# This script demonstrates how to create visualizations of
# training metrics, learning curves, and comparisons.
#
# Usage:
#   ./run_visualize.sh [run_directories...]
#
# Examples:
#   ./run_visualize.sh runs/exp1
#   ./run_visualize.sh runs/mnist_*
#   ./run_visualize.sh runs/sweep_*/exp*

set -e

# Check if arguments provided
if [ $# -eq 0 ]; then
  echo "Usage: $0 <run_dir1> [run_dir2] [run_dir3] ..."
  echo
  echo "Examples:"
  echo "  $0 runs/exp1"
  echo "  $0 runs/mnist_*"
  echo "  $0 runs/sweep_*/exp*"
  exit 1
fi

echo "=================================================="
echo "Visualizing Experiment Results"
echo "=================================================="
echo "Runs to visualize: $#"
echo "=================================================="
echo

# Create output directory
OUTPUT_DIR="visualizations_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# Visualize metrics over time
echo "📊 Creating learning curves..."
expmate viz "$@" \
  --metrics loss accuracy \
  --output "$OUTPUT_DIR/learning_curves.png"

echo
echo "📈 Creating metric comparison plots..."
expmate viz "$@" \
  --metrics loss \
  --split val \
  --output "$OUTPUT_DIR/val_loss_comparison.png"

echo
echo "🎯 Creating best metrics visualization..."
expmate viz "$@" \
  --best \
  --output "$OUTPUT_DIR/best_metrics.png"

echo
echo "=================================================="
echo "Visualization Complete!"
echo "=================================================="
echo "Charts saved to: $OUTPUT_DIR/"
echo "  - learning_curves.png"
echo "  - val_loss_comparison.png"
echo "  - best_metrics.png"
echo "=================================================="

#!/bin/bash
# Run hyperparameter sweep using ExpMate CLI
#
# This script demonstrates using the 'expmate sweep' command to run
# multiple experiments with different hyperparameter combinations.
#
# Usage:
#   ./run_sweep.sh [example_script] [base_config]
#
# Examples:
#   ./run_sweep.sh 01_linear_regression.py conf/linear_regression.yaml
#   ./run_sweep.sh 02_mnist_classification.py conf/mnist.yaml

set -e

# Default values
EXAMPLE_SCRIPT=${1:-02_mnist_classification.py}
BASE_CONFIG=${2:-conf/mnist.yaml}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=================================================="
echo "Hyperparameter Sweep with ExpMate CLI"
echo "=================================================="
echo "Script: $EXAMPLE_SCRIPT"
echo "Base config: $BASE_CONFIG"
echo "=================================================="
echo

# Create sweep configuration file
SWEEP_CONFIG=$(mktemp --suffix=.yaml)
cat > "$SWEEP_CONFIG" << EOF
# Sweep configuration
sweep:
  method: grid  # grid, random, or bayes
  metric:
    name: val_loss
    goal: minimize

# Parameter grid
parameters:
  training.lr:
    values: [0.0001, 0.001, 0.01]
  model.hidden_dim:
    values: [64, 128, 256]
  data.batch_size:
    values: [32, 64, 128]

# Command template
command:
  - python
  - "$SCRIPT_DIR/$EXAMPLE_SCRIPT"
  - "$SCRIPT_DIR/$BASE_CONFIG"
EOF

echo "Sweep configuration saved to: $SWEEP_CONFIG"
echo "Running sweep with ExpMate CLI..."
echo

# Run sweep using expmate CLI
expmate sweep \
  --config "$SWEEP_CONFIG" \
  --count 10 \
  --output "runs/sweep_$(date +%Y%m%d_%H%M%S)"

echo
echo "=================================================="
echo "Sweep Completed!"
echo "=================================================="
echo "Use these commands to analyze results:"
echo "  expmate compare runs/sweep_*       # Compare all sweep runs"
echo "  expmate visualize runs/sweep_*     # Visualize metrics"
echo "=================================================="

# Clean up
rm -f "$SWEEP_CONFIG"

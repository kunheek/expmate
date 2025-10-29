#!/bin/bash
# Run DDP training with torchrun
#
# Usage:
#   ./run_ddp.sh [num_gpus] [config_file] [overrides...]
#
# Examples:
#   ./run_ddp.sh 2                                     # 2 GPUs, default config
#   ./run_ddp.sh 4 conf/ddp.yaml                       # 4 GPUs, custom config
#   ./run_ddp.sh 2 conf/ddp.yaml +model.hidden_dim=256 # 2 GPUs with override

set -e

# Default values
NUM_GPUS=${1:-2}
CONFIG_FILE=${2:-conf/ddp.yaml}
shift 2 || true  # Remove first two args, keep the rest as overrides

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "=================================================="
echo "Distributed Training with PyTorch DDP"
echo "=================================================="
echo "Number of GPUs: $NUM_GPUS"
echo "Config file: $CONFIG_FILE"
echo "Overrides: $@"
echo "=================================================="
echo

# Run with torchrun
torchrun \
  --nproc_per_node=$NUM_GPUS \
  --standalone \
  "$SCRIPT_DIR/03_ddp_training.py" \
  "$CONFIG_FILE" \
  "$@"

echo
echo "✅ DDP training completed!"

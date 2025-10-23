#!/bin/bash
#
# Script to run DDP training with torchrun
#
# Usage:
#   ./train.sh                    # Run with 2 GPUs (or 2 CPUs if no GPU)
#   ./train.sh 4                  # Run with 4 processes
#   ./train.sh 1 conf/custom.yaml # Run single process with custom config

set -e  # Exit on error

# Number of processes (default: 2)
NPROC=${1:-2}

# Config file (default: conf/example.yaml)
CONFIG=${2:-conf/example.yaml}

# Additional arguments
EXTRA_ARGS="${@:3}"

echo "============================================================"
echo "PyTorch Distributed Training with torchrun"
echo "============================================================"
echo "Processes:      ${NPROC}"
echo "Config:         ${CONFIG}"
echo "Extra args:     ${EXTRA_ARGS}"
echo "============================================================"
echo ""

# Check if torch is installed
if ! python3 -c "import torch" 2>/dev/null; then
  echo "❌ Error: PyTorch is not installed!"
  echo "Install with: pip install torch"
  exit 1
fi

# Check if config file exists
if [ ! -f "${CONFIG}" ]; then
  echo "❌ Error: Config file not found: ${CONFIG}"
  exit 1
fi

# Run with torchrun
# torchrun automatically sets RANK, LOCAL_RANK, WORLD_SIZE environment variables
torchrun \
  --nproc_per_node=${NPROC} \
  --standalone \
  examples/train.py \
  ${CONFIG} \
  ${EXTRA_ARGS}

echo ""
echo "============================================================"
echo "✅ Training completed successfully!"
echo "============================================================"

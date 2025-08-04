#!/bin/bash
# run_muon_comparison.sh - Launch script for comparing optimizers in local miner test
#
# Usage:
#   ./run_muon_comparison.sh [adamw|muon] [num_gpus]
#
# Examples:
#   ./run_muon_comparison.sh muon 8      # Run with Muon on 8 GPUs
#   ./run_muon_comparison.sh adamw 4     # Run with AdamW on 4 GPUs
#   ./run_muon_comparison.sh             # Default: AdamW on 1 GPU

set -e

# Parse arguments
OPTIMIZER=${1:-adamw}
NGPU=${2:-1}

# Validate optimizer choice
if [[ "$OPTIMIZER" != "adamw" && "$OPTIMIZER" != "muon" ]]; then
    echo "Error: Invalid optimizer '$OPTIMIZER'. Must be 'adamw' or 'muon'."
    exit 1
fi

# Set script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Ensure we're in the templar root directory
cd "$SCRIPT_DIR/../.."

# Set dataset path if not already set
# Note, see Test Shard Usage explained here:
# https://github.com/tplr-ai/templar/tree/torchtitan-rebased/scripts/dataset_prep#local-testing-with-single-shard
export DATASET_BINS_PATH=${DATASET_BINS_PATH:-"./scripts/dataset_prep/test_shard"}

# Check if dataset exists (looking for tokens.bin specifically)
if [ ! -f "$DATASET_BINS_PATH/tokens.bin" ]; then
    echo "Dataset not found. Preparing test dataset, if this fails try exporting your HF_TOKEN..."
    
    # Save current directory
    ORIGINAL_DIR=$(pwd)
    
    # Go to dataset prep directory
    cd scripts/dataset_prep
    
    # Generate test shard
    echo "Generating test shard..."
    python single_testing_shard.py --output_dir ./test_shard
    
    # Consolidate the shard
    echo "Consolidating shard to create tokens.bin..."
    python 02_consolidate_shards.py --data_root ./test_shard --seq_len 2048 --skip_validation
    
    # Return to original directory
    cd "$ORIGINAL_DIR"
    
    echo "Dataset preparation complete!"
fi

echo "=========================================="
echo "Local Miner Test - Optimizer Comparison"
echo "=========================================="
echo "Optimizer: $OPTIMIZER"
echo "GPUs: $NGPU"
echo "Dataset: $DATASET_BINS_PATH"
echo "=========================================="

# Set optimizer-specific parameters
if [ "$OPTIMIZER" = "muon" ]; then
    # Muon typically uses higher learning rates
    INNER_LR="0.02"
    EXTRA_ARGS="--muon-momentum 0.95 --muon-weight-decay 0.01"
else
    # AdamW default
    INNER_LR="2e-4"
    EXTRA_ARGS=""
fi

# Run the test
if [ "$NGPU" -gt 1 ]; then
    echo "Running with torchrun on $NGPU GPUs..."
    torchrun \
        --standalone \
        --nproc_per_node="$NGPU" \
        scripts/muon/local_miner_test.py \
        --inner-optimizer "$OPTIMIZER" \
        --inner-learning-rate "$INNER_LR" \
        --inner-windows 100 \
        --micro-batch-size 1 \
        --enable-profiler \
        --profiler-wait 5 \
        --profiler-warmup 2 \
        --profiler-active 3 \
        $EXTRA_ARGS
else
    echo "Running on single GPU..."
    python scripts/muon/local_miner_test.py \
        --inner-optimizer "$OPTIMIZER" \
        --inner-learning-rate "$INNER_LR" \
        --inner-windows 100 \
        --micro-batch-size 2 \
        --enable-profiler \
        --profiler-wait 5 \
        --profiler-warmup 2 \
        --profiler-active 3 \
        $EXTRA_ARGS
fi

echo "=========================================="
echo "Test completed!"
echo "Check the results directory for metrics and profiler traces: $SCRIPT_DIR/results/"
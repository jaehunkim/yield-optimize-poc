#!/bin/bash
# DeepFM 평가 스크립트
#
# Usage:
#   ./evaluate_deepfm.sh [CAMPAIGN] [EMBEDDING_DIM] [BATCH_SIZE] [NUM_WORKERS] [MODEL_PATH]
#
# Example:
#   ./evaluate_deepfm.sh 2259 8 1024 4
#   ./evaluate_deepfm.sh 2259 8 1024 4 training/models/deepfm_emb8_lr0.0001_best.pth

set -e

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Set PYTHONPATH to include project root
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Suppress TensorFlow info messages
export TF_CPP_MIN_LOG_LEVEL=2

echo "=== DeepFM Evaluation Script ==="
echo ""

# Default parameters
CAMPAIGN=${1:-2259}
EMBEDDING_DIM=${2:-8}
BATCH_SIZE=${3:-1024}
NUM_WORKERS=${4:-4}
MODEL_PATH=${5:-""}  # Empty = auto-detect latest

echo "Evaluation Configuration:"
echo "  Campaign: $CAMPAIGN"
echo "  Embedding dim: $EMBEDDING_DIM"
echo "  Batch size: $BATCH_SIZE"
echo "  Workers: $NUM_WORKERS"

if [ -z "$MODEL_PATH" ]; then
    echo "  Model: Auto-detect latest for emb$EMBEDDING_DIM"
else
    echo "  Model: $MODEL_PATH"
fi
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Warning: venv not found, using system Python"
fi

# Evaluate
if [ -z "$MODEL_PATH" ]; then
    # Auto-detect latest model
    python training/src/evaluate/evaluate.py \
        --campaign $CAMPAIGN \
        --embedding-dim $EMBEDDING_DIM \
        --batch-size $BATCH_SIZE \
        --num-workers $NUM_WORKERS \
        --device auto
else
    # Use specified model path
    python training/src/evaluate/evaluate.py \
        --campaign $CAMPAIGN \
        --embedding-dim $EMBEDDING_DIM \
        --model-path "$MODEL_PATH" \
        --batch-size $BATCH_SIZE \
        --num-workers $NUM_WORKERS \
        --device auto
fi

echo ""
echo "=== Evaluation Complete ==="
echo "Results saved to: training/results/evaluation.json"

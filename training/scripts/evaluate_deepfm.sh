#!/bin/bash
# DeepFM 평가 스크립트

set -e

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Set PYTHONPATH to include project root
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=== DeepFM Evaluation Script ==="

# Default parameters
CAMPAIGN=${1:-2259}
DAYS=${2:-3}
MODEL_PATH=${3:-training/models/deepfm_emb8_lr0.0005_best.pth}
BATCH_SIZE=${4:-1024}
NUM_WORKERS=${5:-4}

echo "Campaign: $CAMPAIGN"
echo "Days: $DAYS"
echo "Model path: $MODEL_PATH"
echo "Batch size: $BATCH_SIZE"
echo "Workers: $NUM_WORKERS"
echo "(Embedding dim will be auto-detected from model filename)"
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Warning: venv not found, using system Python"
fi

# Evaluate (embedding-dim auto-detected from filename)
python training/src/evaluate/evaluate.py \
    --campaign $CAMPAIGN \
    --days $DAYS \
    --model-path "$MODEL_PATH" \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --device auto

echo ""
echo "=== Evaluation Complete ==="
echo "Results saved to: training/results/evaluation.json"

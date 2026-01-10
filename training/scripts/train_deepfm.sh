#!/bin/bash
# DeepFM 학습 스크립트

set -e

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Set PYTHONPATH to include project root
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=== DeepFM Training Script ==="

# Default parameters
CAMPAIGN=${1:-2259}
DAYS=${2:-3}
EPOCHS=${3:-15}
BATCH_SIZE=${4:-512}
NUM_WORKERS=${5:-4}
EMBEDDING_DIM=${6:-16}
LEARNING_RATE=${7:-0.001}

echo "Campaign: $CAMPAIGN"
echo "Days: $DAYS"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Workers: $NUM_WORKERS"
echo "Embedding dim: $EMBEDDING_DIM"
echo "Learning rate: $LEARNING_RATE"
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Warning: venv not found, using system Python"
fi

# Train
python training/src/train/train.py \
    --campaign $CAMPAIGN \
    --days $DAYS \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --embedding-dim $EMBEDDING_DIM \
    --lr $LEARNING_RATE \
    --device auto

# Evaluate
python training/src/evaluate/evaluate.py \
    --campaign $CAMPAIGN \
    --days $DAYS \
    --batch-size $(($BATCH_SIZE * 2)) \
    --num-workers $NUM_WORKERS \
    --embedding-dim $EMBEDDING_DIM \
    --device auto

echo ""
echo "=== Training Complete ==="
echo "Model saved to: training/models/deepfm_emb${EMBEDDING_DIM}_lr${LEARNING_RATE}_best.pth"
echo "History saved to: training/models/training_history_emb${EMBEDDING_DIM}_lr${LEARNING_RATE}.json"
echo "Results saved to: training/results/evaluation.json"

#!/bin/bash
# DeepFM 학습 스크립트
#
# Usage:
#   ./train_deepfm.sh [CAMPAIGN] [EPOCHS] [BATCH_SIZE] [NUM_WORKERS] [EMBEDDING_DIM] [LEARNING_RATE] [DROPOUT] [WEIGHT_DECAY]
#
# Example:
#   ./train_deepfm.sh 2259 20 512 4 8 0.0001 0.5 1e-4

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

echo "=== DeepFM Training Script ==="
echo ""

# Default parameters (Updated for overfitting prevention)
CAMPAIGN=${1:-2259}
EPOCHS=${2:-20}
BATCH_SIZE=${3:-512}
NUM_WORKERS=${4:-4}
EMBEDDING_DIM=${5:-8}         # Phase 1 optimal: 8
LEARNING_RATE=${6:-0.0001}    # Reduced from 0.0005 to prevent epoch-1 overfitting
DROPOUT=${7:-0.5}             # DNN dropout for regularization
WEIGHT_DECAY=${8:-1e-4}       # L2 regularization (stronger than default 1e-5)

echo "Training Configuration:"
echo "  Campaign: $CAMPAIGN (using ALL training days)"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Workers: $NUM_WORKERS"
echo ""
echo "Model Hyperparameters (Regularized):"
echo "  Embedding dim: $EMBEDDING_DIM"
echo "  Learning rate: $LEARNING_RATE (reduced for stability)"
echo "  Dropout: $DROPOUT"
echo "  Weight decay: $WEIGHT_DECAY (L2 reg)"
echo ""
echo "Data: training/data/processed/campaign_${CAMPAIGN}/"
echo "  - Train/Val: training3rd (all days, split 80/20)"
echo "  - Test: testing3rd (official leaderboard)"
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Warning: venv not found, using system Python"
fi

# Train
echo "=== Starting Training ==="
python training/src/train/train.py \
    --campaign $CAMPAIGN \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --embedding-dim $EMBEDDING_DIM \
    --lr $LEARNING_RATE \
    --dnn-dropout $DROPOUT \
    --weight-decay $WEIGHT_DECAY \
    --device auto

# Evaluate
echo ""
echo "=== Starting Evaluation ==="
python training/src/evaluate/evaluate.py \
    --campaign $CAMPAIGN \
    --batch-size $(($BATCH_SIZE * 2)) \
    --num-workers $NUM_WORKERS \
    --embedding-dim $EMBEDDING_DIM \
    --device auto

echo ""
echo "=== Training Complete ==="
echo "Model saved to: training/models/deepfm_emb${EMBEDDING_DIM}_lr${LEARNING_RATE}_best.pth"
echo "History saved to: training/models/training_history_emb${EMBEDDING_DIM}_lr${LEARNING_RATE}.json"
echo "Results saved to: training/results/evaluation.json"
echo ""
echo "Baseline LR Test AUC: 0.6727 (Target Encoding + Temporal)"
echo "Check if DeepFM beats this baseline!"

#!/bin/bash
# DeepFM 학습 스크립트
#
# Usage:
#   ./train_deepfm.sh [CAMPAIGN] [EPOCHS] [BATCH_SIZE] [NUM_WORKERS] [EMBEDDING_DIM] [LEARNING_RATE] [DROPOUT] [WEIGHT_DECAY] [NEG_POS_RATIO] [DNN_HIDDEN] [PATIENCE]
#
# Example:
#   ./train_deepfm.sh 2259 20 512 4 8 0.0001 0.5 1e-4           # No downsampling, default DNN hidden
#   ./train_deepfm.sh 2259 40 512 4 8 0.0001 0.5 1e-4 100       # 1:100 downsampling, default DNN hidden
#   ./train_deepfm.sh 2259 100 1024 4 8 0.0001 0.5 1e-4 200 "128,64" 10  # Full customization

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
NEG_POS_RATIO=${9:-}          # Negative downsampling ratio (e.g., 100 for 1:100). Leave empty for no downsampling
DNN_HIDDEN=${10:-"256,128,64"}  # DNN hidden layer sizes (comma-separated)
PATIENCE=${11:-3}             # Early stopping patience

echo "Training Configuration:"
echo "  Campaign: $CAMPAIGN (using ALL training days)"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Workers: $NUM_WORKERS"
echo ""
echo "Model Hyperparameters (Regularized):"
echo "  Embedding dim: $EMBEDDING_DIM"
echo "  DNN hidden: $DNN_HIDDEN"
echo "  Learning rate: $LEARNING_RATE (reduced for stability)"
echo "  Dropout: $DROPOUT"
echo "  Weight decay: $WEIGHT_DECAY (L2 reg)"
echo "  Patience: $PATIENCE"
echo ""
if [ -n "$NEG_POS_RATIO" ]; then
    echo "Data: training/data/processed/campaign_${CAMPAIGN}_neg${NEG_POS_RATIO}/"
    echo "  - Negative Downsampling: 1:${NEG_POS_RATIO} (TRAINING SET ONLY)"
    echo "  - Train: training3rd (downsampled, 80% of chronological data)"
    echo "  - Val: training3rd (original distribution, 20% of chronological data)"
    echo "  - Test: testing3rd (official leaderboard, original distribution)"
else
    echo "Data: training/data/processed/campaign_${CAMPAIGN}/"
    echo "  - Train/Val: training3rd (all days, split 80/20, original distribution)"
    echo "  - Test: testing3rd (official leaderboard)"
fi
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Warning: venv not found, using system Python"
fi

# Train
echo "=== Starting Training ==="
if [ -n "$NEG_POS_RATIO" ]; then
    python training/src/train/train.py \
        --campaign $CAMPAIGN \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --num-workers $NUM_WORKERS \
        --embedding-dim $EMBEDDING_DIM \
        --lr $LEARNING_RATE \
        --dnn-dropout $DROPOUT \
        --weight-decay $WEIGHT_DECAY \
        --dnn-hidden $DNN_HIDDEN \
        --patience $PATIENCE \
        --neg-pos-ratio $NEG_POS_RATIO \
        --device auto
else
    python training/src/train/train.py \
        --campaign $CAMPAIGN \
        --epochs $EPOCHS \
        --batch-size $BATCH_SIZE \
        --num-workers $NUM_WORKERS \
        --embedding-dim $EMBEDDING_DIM \
        --lr $LEARNING_RATE \
        --dnn-dropout $DROPOUT \
        --weight-decay $WEIGHT_DECAY \
        --dnn-hidden $DNN_HIDDEN \
        --patience $PATIENCE \
        --device auto
fi

# Evaluate
echo ""
echo "=== Starting Evaluation ==="
if [ -n "$NEG_POS_RATIO" ]; then
    python training/src/evaluate/evaluate.py \
        --campaign $CAMPAIGN \
        --batch-size $(($BATCH_SIZE * 2)) \
        --num-workers $NUM_WORKERS \
        --embedding-dim $EMBEDDING_DIM \
        --dnn-hidden $DNN_HIDDEN \
        --neg-pos-ratio $NEG_POS_RATIO \
        --device auto
else
    python training/src/evaluate/evaluate.py \
        --campaign $CAMPAIGN \
        --batch-size $(($BATCH_SIZE * 2)) \
        --num-workers $NUM_WORKERS \
        --embedding-dim $EMBEDDING_DIM \
        --dnn-hidden $DNN_HIDDEN \
        --device auto
fi

echo ""
echo "=== Training Complete ==="

# Build filename with DNN hidden and neg_pos_ratio
DNN_STR=$(echo $DNN_HIDDEN | tr -d ',')
if [ -n "$NEG_POS_RATIO" ]; then
    MODEL_FILE="deepfm_emb${EMBEDDING_DIM}_lr${LEARNING_RATE}_dnn${DNN_STR}_neg${NEG_POS_RATIO}_best.pth"
    HISTORY_FILE="training_history_emb${EMBEDDING_DIM}_lr${LEARNING_RATE}_dnn${DNN_STR}_neg${NEG_POS_RATIO}.json"
else
    MODEL_FILE="deepfm_emb${EMBEDDING_DIM}_lr${LEARNING_RATE}_dnn${DNN_STR}_best.pth"
    HISTORY_FILE="training_history_emb${EMBEDDING_DIM}_lr${LEARNING_RATE}_dnn${DNN_STR}.json"
fi

echo "Model saved to: training/models/${MODEL_FILE}"
echo "History saved to: training/models/${HISTORY_FILE}"
echo "Results saved to: training/results/evaluation.json"
echo ""
echo "Baseline LR Test AUC: 0.6727 (Target Encoding + Temporal)"
echo "Check if DeepFM beats this baseline!"

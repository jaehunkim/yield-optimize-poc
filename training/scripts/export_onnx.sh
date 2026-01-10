#!/bin/bash
# ONNX Export 및 검증 스크립트
#
# Usage:
#   ./export_onnx.sh [CAMPAIGN] [MODEL_PATH] [NEG_POS_RATIO]
#
# Examples:
#   ./export_onnx.sh 2259                                                      # Use default model (original distribution)
#   ./export_onnx.sh 2259 training/models/deepfm_emb8_lr0.0001_best.pth        # Specify model (original distribution)
#   ./export_onnx.sh 2259 training/models/deepfm_emb8_lr0.0001_best.pth 200    # Use downsampled data (1:200)

set -e

# Get script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Set PYTHONPATH to include project root
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "=== ONNX Export and Verification Script ==="

# Default parameters
CAMPAIGN=${1:-2259}
MODEL_PATH=${2:-training/models/deepfm_emb8_lr0.0001_dnn25612864_best.pth}
NEG_POS_RATIO=${3:-}  # Optional: use downsampled data (e.g., 100 or 200)
DNN_HIDDEN=${4:-"256,128,64"}  # DNN hidden layer sizes (comma-separated)

echo "Campaign: $CAMPAIGN"
echo "Model path: $MODEL_PATH"
echo "DNN hidden: $DNN_HIDDEN"
if [ -n "$NEG_POS_RATIO" ]; then
    echo "Using downsampled data: 1:$NEG_POS_RATIO"
else
    echo "Using original distribution data"
fi
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Warning: venv not found, using system Python"
fi

# Step 1: Export to ONNX
echo "=== Step 1: Exporting to ONNX ==="
if [ -n "$NEG_POS_RATIO" ]; then
    python training/src/export/export_onnx.py \
        --campaign $CAMPAIGN \
        --model-path "$MODEL_PATH" \
        --neg-pos-ratio $NEG_POS_RATIO \
        --dnn-hidden "$DNN_HIDDEN"
else
    python training/src/export/export_onnx.py \
        --campaign $CAMPAIGN \
        --model-path "$MODEL_PATH" \
        --dnn-hidden "$DNN_HIDDEN"
fi

echo ""

# Step 2: Verify ONNX model
echo "=== Step 2: Verifying ONNX Model ==="
if [ -n "$NEG_POS_RATIO" ]; then
    python training/src/export/verify_onnx.py \
        --campaign $CAMPAIGN \
        --model-path "$MODEL_PATH" \
        --neg-pos-ratio $NEG_POS_RATIO \
        --dnn-hidden "$DNN_HIDDEN" \
        --num-samples 1000
else
    python training/src/export/verify_onnx.py \
        --campaign $CAMPAIGN \
        --model-path "$MODEL_PATH" \
        --dnn-hidden "$DNN_HIDDEN" \
        --num-samples 1000
fi

echo ""
echo "=== ONNX Export Complete ==="
echo "Results saved to: training/results/onnx_verification.json"

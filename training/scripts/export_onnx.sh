#!/bin/bash
# ONNX Export 및 검증 스크립트

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
DAYS=${2:-3}
MODEL_PATH=${3:-training/models/deepfm_emb8_lr0.0005_best.pth}

echo "Campaign: $CAMPAIGN"
echo "Days: $DAYS"
echo "Model path: $MODEL_PATH"
echo ""

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Warning: venv not found, using system Python"
fi

# Step 1: Export to ONNX
echo "=== Step 1: Exporting to ONNX ==="
python training/src/export/export_onnx.py \
    --campaign $CAMPAIGN \
    --days $DAYS \
    --model-path "$MODEL_PATH"

echo ""

# Step 2: Verify ONNX model
echo "=== Step 2: Verifying ONNX Model ==="
python training/src/export/verify_onnx.py \
    --campaign $CAMPAIGN \
    --days $DAYS \
    --model-path "$MODEL_PATH" \
    --num-samples 1000

echo ""
echo "=== ONNX Export Complete ==="
echo "Results saved to: training/results/onnx_verification.json"

#!/bin/bash
# Thread tuning script - find optimal intra/inter thread settings
# Usage: ./tune_threads.sh [model_type]

set -e

MODEL_TYPE=${1:-autoint}
DURATION=10s

echo "=============================================="
echo "Thread Tuning for ONNX Runtime (AMD CPU)"
echo "Model: $MODEL_TYPE"
echo "=============================================="
echo ""

# Test configurations
# Format: "intra_threads:inter_threads"
CONFIGS=(
    "1:1"
    "2:1"
    "4:1"
    "8:1"
    "2:2"
    "4:2"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Testing configurations..."
echo ""

for config in "${CONFIGS[@]}"; do
    IFS=':' read -r intra inter <<< "$config"

    echo "----------------------------------------------"
    echo "Testing: intra_threads=$intra, inter_threads=$inter"
    echo "----------------------------------------------"

    MODEL=$MODEL_TYPE INTRA_THREADS=$intra INTER_THREADS=$inter \
        "$SCRIPT_DIR/benchmark.sh" single $DURATION 2>&1 | \
        grep -E "(Requests/sec|Latency|Transfer)" || true

    echo ""
    sleep 2
done

echo "=============================================="
echo "Tuning complete!"
echo "=============================================="

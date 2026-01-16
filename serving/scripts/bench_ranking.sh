#!/bin/bash
# Multi-Stage Ranking Benchmark Script
#
# Usage:
#   ./bench_ranking.sh [candidates] [iterations]
#
# Examples:
#   ./bench_ranking.sh 1000 100  # 1000 candidates, 100 iterations
#   ./bench_ranking.sh           # defaults: 1000 candidates, 50 iterations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Parameters
CANDIDATES=${1:-1000}
ITERATIONS=${2:-50}
PORT=8080
SERVER_URL="http://localhost:$PORT"

# Model paths
DEEPFM_MODEL="$PROJECT_ROOT/models/deepfm_emb8_lr0.0001_dnn25612864_neg150_best.onnx"
AUTOINT_MODEL="$PROJECT_ROOT/models/autoint_emb64_att3x4_dnn256128_neg150_best.onnx"

echo "=== Multi-Stage Ranking Benchmark ==="
echo "Candidates: $CANDIDATES"
echo "Iterations: $ITERATIONS"
echo "DeepFM: $DEEPFM_MODEL"
echo "AutoInt: $AUTOINT_MODEL"
echo ""

# Check if models exist
if [ ! -f "$DEEPFM_MODEL" ]; then
    echo "ERROR: DeepFM model not found at $DEEPFM_MODEL"
    exit 1
fi

if [ ! -f "$AUTOINT_MODEL" ]; then
    echo "ERROR: AutoInt model not found at $AUTOINT_MODEL"
    exit 1
fi

# Check if server is running
check_server() {
    curl -s "$SERVER_URL/health" > /dev/null 2>&1
}

# Generate random candidates
generate_candidates() {
    local n=$1
    python3 -c "
import json
import random
candidates = [[random.random() for _ in range(15)] for _ in range($n)]
print(json.dumps(candidates))
"
}

# Run benchmark
run_benchmark() {
    local mode=$1
    local multi_stage=$2

    echo "--- $mode ---"

    # Generate candidates JSON
    CANDIDATES_JSON=$(generate_candidates $CANDIDATES)

    # Build request
    if [ "$multi_stage" = "true" ]; then
        REQUEST="{\"candidates\": $CANDIDATES_JSON, \"top_k\": 10, \"stage1_k\": 100, \"multi_stage\": true}"
    else
        REQUEST="{\"candidates\": $CANDIDATES_JSON, \"top_k\": 10, \"multi_stage\": false}"
    fi

    # Warmup
    echo "Warming up..."
    for i in {1..5}; do
        curl -s -X POST "$SERVER_URL/rank" \
            -H "Content-Type: application/json" \
            -d "$REQUEST" > /dev/null
    done

    # Benchmark
    echo "Running $ITERATIONS iterations..."

    TOTAL_TIME=0
    STAGE1_TOTAL=0
    STAGE2_TOTAL=0

    for i in $(seq 1 $ITERATIONS); do
        RESPONSE=$(curl -s -X POST "$SERVER_URL/rank" \
            -H "Content-Type: application/json" \
            -d "$REQUEST")

        TOTAL_MS=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['latency']['total_ms'])")
        STAGE1_MS=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['latency']['stage1_ms'])")
        STAGE2_MS=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['latency']['stage2_ms'])")

        TOTAL_TIME=$(python3 -c "print($TOTAL_TIME + $TOTAL_MS)")
        STAGE1_TOTAL=$(python3 -c "print($STAGE1_TOTAL + $STAGE1_MS)")
        STAGE2_TOTAL=$(python3 -c "print($STAGE2_TOTAL + $STAGE2_MS)")
    done

    AVG_TOTAL=$(python3 -c "print(f'{$TOTAL_TIME / $ITERATIONS:.2f}')")
    AVG_STAGE1=$(python3 -c "print(f'{$STAGE1_TOTAL / $ITERATIONS:.2f}')")
    AVG_STAGE2=$(python3 -c "print(f'{$STAGE2_TOTAL / $ITERATIONS:.2f}')")

    echo "Results:"
    echo "  Average total latency: ${AVG_TOTAL}ms"
    echo "  Average Stage 1 (DeepFM): ${AVG_STAGE1}ms"
    echo "  Average Stage 2 (AutoInt): ${AVG_STAGE2}ms"
    echo ""
}

# Check server
if ! check_server; then
    echo "Server not running. Starting server..."
    echo ""
    echo "Run this command in another terminal:"
    echo "  cd $PROJECT_ROOT && cargo run --release -- \\"
    echo "    --deepfm-model models/deepfm_emb8_lr0.0001_dnn25612864_neg150_best.onnx \\"
    echo "    --autoint-model models/autoint_emb64_att3x4_dnn256128_neg150_best.onnx"
    echo ""
    exit 1
fi

echo "Server is running at $SERVER_URL"
echo ""

# Run benchmarks
run_benchmark "Single-Stage (AutoInt only)" "false"
run_benchmark "Multi-Stage (DeepFM -> AutoInt)" "true"

echo "=== Benchmark Complete ==="

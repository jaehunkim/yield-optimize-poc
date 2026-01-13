#!/bin/bash
# Benchmark script for CTR model serving
# Usage: ./benchmark.sh [single|multi] [duration]

set -e

MODE=${1:-single}
DURATION=${2:-30s}

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Model selection via MODEL env var
# Options: deepfm (default), autoint
MODEL_TYPE=${MODEL:-deepfm}

# ONNX Runtime thread configuration
INTRA_THREADS=${INTRA_THREADS:-4}
INTER_THREADS=${INTER_THREADS:-1}

if [ "$MODEL_TYPE" == "autoint" ]; then
    MODEL_PATH="$PROJECT_ROOT/models/autoint_emb64_att3x4_dnn256128_neg150_best_dynamic.onnx"
    MODEL_NAME="AutoInt"
elif [ "$MODEL_TYPE" == "autoint-int8" ]; then
    MODEL_PATH="$PROJECT_ROOT/models/autoint_emb64_att3x4_dnn256128_neg150_best_int8_dynamic.onnx"
    MODEL_NAME="AutoInt-INT8-Dynamic"
elif [ "$MODEL_TYPE" == "autoint-int8-static" ]; then
    MODEL_PATH="$PROJECT_ROOT/models/autoint_int8_static.onnx"
    MODEL_NAME="AutoInt-INT8-Static"
else
    MODEL_PATH="$PROJECT_ROOT/models/deepfm_emb8_lr0.0001_dnn25612864_neg150_best.onnx"
    MODEL_NAME="DeepFM"
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if wrk is installed
if ! command -v wrk &> /dev/null; then
    log_error "wrk is not installed. Install it with:"
    echo "  Ubuntu/Debian: sudo apt-get install wrk"
    echo "  macOS: brew install wrk"
    exit 1
fi

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    log_error "Model file not found: $MODEL_PATH"
    exit 1
fi

# Cleanup function
cleanup() {
    log_info "Cleaning up processes..."
    pkill -f "deepfm-serving" || true
    sleep 1
}

trap cleanup EXIT

# Build release binary
log_info "Building release binary..."
cd "$PROJECT_ROOT"
cargo build --release --quiet

BINARY="$PROJECT_ROOT/target/release/deepfm-serving"

if [ "$MODE" == "single" ]; then
    log_info "=== Single Process Benchmark ($MODEL_NAME) ==="
    log_info "Starting server on port 3000..."

    RUST_LOG=warn $BINARY --port 3000 --model "$MODEL_PATH" \
        --intra-threads $INTRA_THREADS --inter-threads $INTER_THREADS &
    SERVER_PID=$!
    sleep 3

    log_info "Running benchmark (duration: $DURATION)..."
    echo "=============================="
    log_info "Benchmark Configuration:"
    echo "  Model:         $MODEL_NAME"
    echo "  Processes:     1"
    echo "  Intra-threads: $INTRA_THREADS"
    echo "  Inter-threads: $INTER_THREADS"
    echo "  WRK Threads:   4"
    echo "  Connections:   100"
    echo "  Duration:      $DURATION"
    echo "=============================="

    wrk -t4 -c100 -d$DURATION \
        -s "$SCRIPT_DIR/wrk_predict.lua" \
        http://localhost:3000/predict_raw

    echo "=============================="
    log_info "Results Summary:"
    echo "  Model:       $MODEL_NAME"
    echo "  Mode:        Single Process"
    echo "  Threads:     4"
    echo "  Connections: 100"
    echo "=============================="

    kill $SERVER_PID

elif [ "$MODE" == "batch" ]; then
    BATCH_SIZE=${BATCH_SIZE:-32}
    log_info "=== Batched Inference Benchmark ($MODEL_NAME, batch_size=$BATCH_SIZE) ==="
    log_info "Starting server on port 3000..."

    RUST_LOG=warn $BINARY --port 3000 --model "$MODEL_PATH" \
        --intra-threads $INTRA_THREADS --inter-threads $INTER_THREADS &
    SERVER_PID=$!
    sleep 3

    log_info "Running benchmark (duration: $DURATION)..."
    echo "=============================="
    log_info "Benchmark Configuration:"
    echo "  Model:       $MODEL_NAME"
    echo "  Mode:        Batched Inference"
    echo "  Batch size:  $BATCH_SIZE"
    echo "  Threads:     4"
    echo "  Connections: 100"
    echo "  Duration:    $DURATION"
    echo "=============================="

    BATCH_SIZE=$BATCH_SIZE wrk -t4 -c100 -d$DURATION \
        -s "$SCRIPT_DIR/wrk_predict_batch.lua" \
        http://localhost:3000/predict_batch

    echo "=============================="
    log_info "Results Summary:"
    echo "  Model:       $MODEL_NAME"
    echo "  Mode:        Batched Inference"
    echo "  Batch size:  $BATCH_SIZE"
    echo "=============================="

    kill $SERVER_PID

elif [ "$MODE" == "multi" ]; then
    NUM_PROCESSES=${NUM_PROCESSES:-4}
    BASE_PORT=3001

    log_info "=== Multi-Process Benchmark ($MODEL_NAME, ${NUM_PROCESSES} processes) ==="

    # Check if nginx is installed
    if ! command -v nginx &> /dev/null; then
        log_error "nginx is not installed. Install it with:"
        echo "  Ubuntu/Debian: sudo apt-get install nginx"
        echo "  macOS: brew install nginx"
        exit 1
    fi

    # Start multiple processes
    log_info "Starting $NUM_PROCESSES server processes..."
    PIDS=()
    for i in $(seq 0 $((NUM_PROCESSES - 1))); do
        PORT=$((BASE_PORT + i))
        log_info "  Starting process $((i+1)) on port $PORT..."
        RUST_LOG=warn $BINARY --port $PORT --model "$MODEL_PATH" &
        PIDS+=($!)
    done

    sleep 3

    # Generate nginx config
    log_info "Generating nginx configuration..."
    NGINX_CONF="/tmp/deepfm_nginx.conf"
    cat > $NGINX_CONF <<EOF
worker_processes auto;
daemon off;
error_log /tmp/nginx_error.log warn;
pid /tmp/nginx.pid;

events {
    worker_connections 1024;
}

http {
    access_log /tmp/nginx_access.log;

    upstream deepfm_backend {
EOF

    for i in $(seq 0 $((NUM_PROCESSES - 1))); do
        PORT=$((BASE_PORT + i))
        echo "        server 127.0.0.1:$PORT;" >> $NGINX_CONF
    done

    cat >> $NGINX_CONF <<EOF
    }

    server {
        listen 3000;

        location / {
            proxy_pass http://deepfm_backend;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
        }
    }
}
EOF

    log_info "Starting nginx load balancer on port 3000..."
    nginx -c $NGINX_CONF &
    NGINX_PID=$!
    sleep 2

    log_info "Running benchmark (duration: $DURATION)..."
    # Allow overriding connections and threads via env vars
    CONNECTIONS=${CONNECTIONS:-$((100 * NUM_PROCESSES))}
    THREADS=${THREADS:-$((4 * NUM_PROCESSES))}

    echo "=============================="
    log_info "Benchmark Configuration:"
    echo "  Model:       $MODEL_NAME"
    echo "  Processes:   $NUM_PROCESSES"
    echo "  Threads:     $THREADS"
    echo "  Connections: $CONNECTIONS"
    echo "  Duration:    $DURATION"
    echo "=============================="

    wrk -t$THREADS -c$CONNECTIONS -d$DURATION \
        -s "$SCRIPT_DIR/wrk_predict.lua" \
        http://localhost:3000/predict_raw

    echo "=============================="
    log_info "Results Summary:"
    echo "  Model:       $MODEL_NAME"
    echo "  Mode:        Multi-Process ($NUM_PROCESSES processes)"
    echo "  Threads:     $THREADS"
    echo "  Connections: $CONNECTIONS"
    echo "=============================="

    # Cleanup
    kill $NGINX_PID
    for pid in "${PIDS[@]}"; do
        kill $pid
    done

else
    log_error "Invalid mode: $MODE"
    echo "Usage: $0 [single|batch|multi] [duration]"
    echo "  single - Single process benchmark (1 request = 1 inference)"
    echo "  batch  - Batched inference benchmark (1 request = N inferences)"
    echo "  multi  - Multi-process benchmark with nginx (default: 4 processes)"
    echo ""
    echo "Environment variables:"
    echo "  MODEL         - Model to use: deepfm (default), autoint, autoint-int8, autoint-int8-static"
    echo "  BATCH_SIZE    - Batch size for batched mode (default: 32)"
    echo "  NUM_PROCESSES - Number of processes for multi mode (default: 4)"
    echo "  CONNECTIONS   - Number of connections (default: 100 * NUM_PROCESSES)"
    echo "  THREADS       - Number of threads (default: 4 * NUM_PROCESSES)"
    echo "  INTRA_THREADS - ONNX intra-op threads (default: 4)"
    echo "  INTER_THREADS - ONNX inter-op threads (default: 1)"
    echo ""
    echo "Examples:"
    echo "  $0 single 30s                            # DeepFM single process"
    echo "  MODEL=autoint $0 single 30s              # AutoInt single process"
    echo "  BATCH_SIZE=64 $0 batch 30s               # Batched inference (64 items/request)"
    echo "  MODEL=autoint BATCH_SIZE=32 $0 batch 30s # AutoInt batched inference"
    echo "  MODEL=autoint NUM_PROCESSES=4 $0 multi 30s"
    exit 1
fi

log_info "Benchmark complete!"

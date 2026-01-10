#!/bin/bash
# Benchmark script for DeepFM serving
# Usage: ./benchmark.sh [single|multi] [duration]

set -e

MODE=${1:-single}
DURATION=${2:-30s}

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_PATH="$PROJECT_ROOT/models/deepfm_emb8_lr0.0001_dnn25612864_neg150_best.onnx"

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
    log_info "=== Single Process Benchmark ==="
    log_info "Starting server on port 3000..."

    RUST_LOG=warn $BINARY --port 3000 --model "$MODEL_PATH" &
    SERVER_PID=$!
    sleep 3

    log_info "Running benchmark (duration: $DURATION)..."
    wrk -t4 -c100 -d$DURATION \
        -s "$SCRIPT_DIR/wrk_predict.lua" \
        http://localhost:3000/predict_raw

    kill $SERVER_PID

elif [ "$MODE" == "multi" ]; then
    NUM_PROCESSES=${NUM_PROCESSES:-4}
    BASE_PORT=3001

    log_info "=== Multi-Process Benchmark (${NUM_PROCESSES} processes) ==="

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
    CONNECTIONS=$((100 * NUM_PROCESSES))
    THREADS=$((4 * NUM_PROCESSES))
    wrk -t$THREADS -c$CONNECTIONS -d$DURATION \
        -s "$SCRIPT_DIR/wrk_predict.lua" \
        http://localhost:3000/predict_raw

    # Cleanup
    kill $NGINX_PID
    for pid in "${PIDS[@]}"; do
        kill $pid
    done

else
    log_error "Invalid mode: $MODE"
    echo "Usage: $0 [single|multi] [duration]"
    echo "  single - Single process benchmark"
    echo "  multi  - Multi-process benchmark with nginx (default: 4 processes)"
    echo "           Set NUM_PROCESSES env var to change: NUM_PROCESSES=8 $0 multi"
    exit 1
fi

log_info "Benchmark complete!"

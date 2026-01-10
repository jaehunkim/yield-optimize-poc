# Multi-stage build for DeepFM serving
FROM rust:1.83-slim as builder

WORKDIR /build

# Install dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy manifests
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY src ./src

# Build release binary
RUN cargo build --release

# Runtime stage
FROM debian:bookworm-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy binary from builder
COPY --from=builder /build/target/release/deepfm-serving /app/deepfm-serving

# Default model path (override with volume mount)
ENV MODEL_PATH=/app/models/deepfm_emb8_lr0.0001_dnn25612864_neg150_best.onnx
ENV PORT=3001
ENV RUST_LOG=info

EXPOSE ${PORT}

CMD ["/bin/sh", "-c", "/app/deepfm-serving --port $PORT --model $MODEL_PATH"]

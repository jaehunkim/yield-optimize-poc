# Phase 3: Serving 최적화 실험

## 결론

**71M AutoInt 모델에서 INT8 양자화 + 멀티스레드로 약 30% 성능 향상 달성.**

| 최적화 | 단독 효과 | 비고 |
|--------|----------|------|
| XNNPACK | ~10% | FP32 모델에서만 효과 |
| INT8 양자화 | ~15% | 모델 크기 55% 감소 (273MB → 123MB) |
| 멀티스레드 (4 threads) | ~13% | 8 threads는 오히려 악화 |
| **INT8 + 4 threads** | **~30%** | XNNPACK과 INT8 조합은 무의미 |

---

## 실험 환경

- **모델**: AutoInt 71M (emb768, att8x12, dnn2048-1024-512)
- **ONNX 크기**: FP32 273MB, INT8 123MB
- **CPU**: AMD Ryzen 7 7800X3D
- **Runtime**: ONNX Runtime 2.0 (Rust ort crate)

---

## 단건 Inference 결과

| 설정 | P50 Latency | P95 Latency | Throughput |
|------|-------------|-------------|------------|
| Baseline (FP32, 1 thread) | 11.03 ms | 13.02 ms | 88/s |
| XNNPACK ON | 10.19 ms | 11.53 ms | 96/s |
| INT8 only | 9.15 ms | 10.22 ms | 108/s |
| INT8 + XNNPACK | 9.32 ms | 10.45 ms | 106/s |
| **INT8 + 4 threads** | **7.79 ms** | **8.34 ms** | **128/s** |

### 멀티스레드 튜닝 결과 (INT8)

| Threads | P50 | P95 | vs 1 thread |
|---------|-----|-----|-------------|
| 1 | 8.96 ms | 9.69 ms | baseline |
| 2 | 8.07 ms | 8.69 ms | -10% |
| **4** | **7.79 ms** | **8.34 ms** | **-13%** |
| 8 | 8.55 ms | 9.60 ms | +5% (악화) |

---

## 부하 테스트 결과 (wrk)

### 10 Concurrent Connections

| 설정 | P50 | P99 | Throughput |
|------|-----|-----|------------|
| Baseline (FP32, 1 thread) | 32.18 ms | 64.75 ms | 286 req/s |
| **INT8 + 4 threads** | **18.51 ms** | **55.17 ms** | **448 req/s** |
| **개선율** | **-42%** | **-15%** | **+57%** |

---

## 분석

### XNNPACK + INT8 조합이 무의미한 이유

1. **XNNPACK은 FP32 연산에 최적화됨**: INT8 양자화 연산자(QuantizeLinear, DequantizeLinear, MatMulInteger)는 XNNPACK이 지원하지 않음
2. **Execution Provider Fallback 오버헤드**: XNNPACK이 처리 못하는 연산은 CPU EP로 전환되며 오버헤드 발생
3. **ONNX Runtime CPU EP의 INT8 최적화**: 기본 CPU EP가 이미 AVX2/AVX-512 VNNI로 INT8을 잘 처리함

### 8 threads에서 성능 악화 이유

- 71M 모델의 행렬 연산 크기 대비 스레드 동기화 오버헤드가 큼
- 최적 스레드 수는 모델 크기와 CPU 아키텍처에 따라 다름

---

## 권장 설정

```rust
ModelConfig {
    intra_threads: 4,      // 행렬 연산 병렬화
    inter_threads: 1,      // 연산자 간 병렬화 불필요
    enable_mem_pattern: true,
    pool_size: 8,          // 동시 요청 처리용 세션 풀
    enable_xnnpack: false, // INT8 모델에서는 비활성화
}
```

---

## 이전 실험 (작은 모델)

**모델이 너무 작아서 최적화 효과를 측정할 수 없었다.**

| 모델 | 파라미터 | ONNX 크기 | 단건 Latency |
|------|----------|-----------|--------------|
| 기존 AutoInt | 2.8M | 11.6MB | 0.1ms |
| 25x AutoInt | 15.7M | 60MB | 1.1ms |
| **71M AutoInt** | **71M** | **273MB** | **~10ms** |
| 280M AutoInt | 153M | 584MB | 35ms |

→ 최적화 효과를 측정하려면 단건 latency가 최소 5-10ms 이상 필요

---

## 관련 코드

- [serving/src/model.rs](../serving/src/model.rs) - Session Pool 및 XNNPACK 토글 구현
- [serving/examples/bench_single_inference.rs](../serving/examples/bench_single_inference.rs) - 단건 Latency 벤치마크
- [training/src/export/quantize_onnx.py](../training/src/export/quantize_onnx.py) - INT8 양자화 스크립트
- [serving/scripts/benchmark.sh](../serving/scripts/benchmark.sh) - 부하 테스트 스크립트

---

## 벤치마크 실행 방법

```bash
# 단건 inference 벤치마크
cargo build --release --example bench_single_inference
./target/release/examples/bench_single_inference --int8 --no-xnnpack --threads=4

# 부하 테스트
MODEL=autoint-71m-int8 INTRA_THREADS=4 CONNECTIONS=10 ./scripts/benchmark.sh single 30s
```

# Phase 4: Multi-Stage Ranking (레이턴시 최적화)

## 목표

Multi-Stage Ranking 파이프라인의 **레이턴시 최적화** 검증
- 1000개 배치 inference 구현
- Stage 1 → Top-K 필터링 → Stage 2 파이프라인 구현
- 레이턴시 측정 및 비교

## 배경

### 모델 성능 현황

| 모델 | Val AUC | Test AUC | 모델 크기 |
|------|---------|----------|-----------|
| DeepFM (emb8, dnn256-128-64) | 0.7305 | 0.6960 | 18KB |
| AutoInt (emb64, att3x4, dnn256-128) | 0.7236 | 0.6959 | 12MB |

**Test AUC가 거의 동일** (0.6960 vs 0.6959)하므로 랭킹 품질 비교는 무의미.
→ **레이턴시 최적화에만 집중**

### 왜 Multi-Stage인가?

Single-Stage로 1000개 후보를 처리하면:
- 개별 추론: 1000개 × 1ms = ~1000ms (불가능)
- 배치 추론: 대량 후보 처리 시 레이턴시 증가

Multi-Stage는 **단계적 필터링으로 연산량 감소**:
- Stage 1: 전체 후보를 빠르게 스코어링
- Stage 2: Top-K만 정밀 랭킹
- 실제 프로덕션 패턴: Google, Meta, 오늘의집 등에서 사용

---

## 실험 설계

### 비교 대상

| 방식 | 설명 |
|------|------|
| **Single-Stage** | AutoInt (12MB)로 1000개 전체 배치 추론 → Top-10 |
| **Multi-Stage** | DeepFM (18KB)로 1000→100 필터링 → AutoInt (12MB)로 100→10 |

### 사용 모델

| Stage | 모델 | 크기 | 역할 |
|-------|------|------|------|
| Stage 1 | `deepfm_emb8_lr0.0001_dnn25612864_neg150_best.onnx` | 18KB | 빠른 필터링 |
| Stage 2 | `autoint_emb64_att3x4_dnn256128_neg150_best.onnx` | 12MB | 정밀 랭킹 |

> **Note**: Test AUC가 거의 동일하므로 Recall@K ≈ 100% 예상 (랭킹 품질 평가 생략)

---

## 아키텍처

```
Request: 1000 candidates
         ↓
┌─────────────────────────────────────┐
│ Stage 1: DeepFM (18KB)              │
│ - Batch inference: 1000개           │
│ - Top-100 선택                      │
│ - 빠른 모델로 대량 필터링           │
└─────────────────────────────────────┘
         ↓ (100 candidates)
┌─────────────────────────────────────┐
│ Stage 2: AutoInt (12MB)             │
│ - Batch inference: 100개            │
│ - Top-10 선택                       │
│ - 정밀 모델로 최종 랭킹             │
└─────────────────────────────────────┘
         ↓
Response: Top-10 with CTR scores
```

---

## API 설계

### POST /rank

**Request:**
```json
{
  "candidates": [
    {"weekday": 4, "hour": 15, "region": 1, ...},
    ...
  ],
  "top_k": 10,
  "stage1_k": 100
}
```

**Response:**
```json
{
  "rankings": [
    {"index": 42, "ctr": 0.0234, "rank": 1},
    {"index": 157, "ctr": 0.0198, "rank": 2},
    ...
  ],
  "latency": {
    "stage1_ms": 5.2,
    "stage2_ms": 0.8,
    "total_ms": 6.0
  },
  "stats": {
    "input": 1000,
    "after_stage1": 100,
    "output": 10
  }
}
```

---

## 실험 결과

### 벤치마크 환경

- CPU: AMD Ryzen 7 7800X3D
- ONNX Runtime: 2.0 (Rust ort crate)
- intra_threads: 4
- 후보 수: 1000개, 반복: 20회

### 레이턴시 비교

| 방식 | 평균 레이턴시 | 상세 |
|------|---------------|------|
| **Single-Stage** (AutoInt 1000개) | **24.38ms** | Min: 16.24ms, Max: 33.03ms |
| **Multi-Stage** (DeepFM→AutoInt) | **5.66ms** | Stage1: 2.49ms, Stage2: 3.13ms |

### 결과 분석

| 지표 | 값 |
|------|-----|
| **Speedup** | **4.31x** |
| Stage 1 (DeepFM 1000개) | 2.49ms |
| Stage 2 (AutoInt 100개) | 3.13ms |

**Multi-Stage가 4배 이상 빠른 이유:**
- Stage 1: 작은 모델(18KB)로 1000개 빠르게 필터링 → 2.5ms
- Stage 2: 큰 모델(12MB)이지만 100개만 처리 → 3.1ms
- 총 연산량: 1000×작은모델 + 100×큰모델 < 1000×큰모델

---

## 구현 파일

| 파일 | 작업 |
|------|------|
| `serving/src/api.rs` | `/rank` 엔드포인트 추가 |
| `serving/src/model.rs` | `OnnxModel` 범용 모델 로더, 배치 inference |
| `serving/src/main.rs` | `--deepfm-model`, `--autoint-model` CLI 옵션 |
| `serving/scripts/bench_ranking.sh` | 레이턴시 벤치마크 스크립트 |

---

## 서버 실행 방법

```bash
cd serving

# Multi-Stage 모드 (DeepFM + AutoInt)
cargo run --release -- \
  --deepfm-model models/deepfm_emb8_lr0.0001_dnn25612864_neg150_best.onnx \
  --autoint-model models/autoint_emb64_att3x4_dnn256128_neg150_best_dynamic.onnx \
  --no-xnnpack \
  --intra-threads 4

# 테스트
curl -X POST http://localhost:8080/rank \
  -H "Content-Type: application/json" \
  -d '{"candidates": [[0.1,0.2,...]], "top_k": 10, "stage1_k": 100, "multi_stage": true}'
```

---

## 핵심 인사이트

1. **4.31x 속도 향상**: Multi-Stage로 레이턴시 24ms → 5.6ms
2. **실제 프로덕션 패턴**: Google, Meta 등에서 사용하는 구조 구현
3. **작은 모델의 가치**: 18KB DeepFM이 12MB AutoInt 대비 10배 빠른 필터링
4. **확장성**: 후보 수 증가 시에도 Stage 2 연산량이 제한됨

---

## 관련 문서

- [Phase 3: Serving 최적화](./phase3_serving_optimization.md) - INT8 양자화, 멀티스레드
- [Phase 2: Rust 서빙 엔진](./phase2_plan.md) - ONNX Runtime 통합

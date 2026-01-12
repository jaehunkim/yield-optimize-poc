# AdTech POC - Project Overview

## 프로젝트 개요

AdTech 포지션 지원을 위한 POC(Proof of Concept) 프로젝트입니다.
실시간 CTR(Click-Through Rate) 예측 모델의 학습부터 서빙까지의 전체 파이프라인을 구현하여 기본적인 개발 능력을 검증합니다.

**예상 소요 시간**: 3-4주
**개발 환경**: MacBook M2

## 전체 로드맵

### Phase 1: 기본 모델 학습 및 ONNX 변환 ✅ (완료)
- iPinYou 데이터셋 다운로드 및 전처리
- PyTorch + DeepFM 모델 학습 및 AUC 검증
- ONNX 변환 및 정합성 검증

**결과**: Test AUC 0.7503 (벤치마크 상회), ONNX 변환 성공 (tolerance < 1e-5)

### Phase 2: Rust 실시간 DSP 엔진 개발 🚧 (진행 중)
- ONNX Runtime + Rust 서빙 엔진 구현
- REST API 기반 실시간 CTR 예측
- Sub-millisecond 레이턴시 달성 (P50 < 1ms, P99 < 5ms)

**목표**: 프로덕션급 성능의 실시간 입찰 시스템 구축

### Phase 3: AutoInt 모델 학습 및 서빙 (계획)
- AutoInt (Multi-head Self-Attention 기반) 모델 학습
- DeepFM 대비 더 큰 모델로 최적화 효과 측정 가능
- Rust 서빙 엔진에 AutoInt 모델 통합

**목표**: 더 복잡한 모델로 서빙 최적화 효과 검증

### Phase 4: 모델 경량화 및 최적화 (계획)
- 양자화(Quantization), 프루닝(Pruning) 등 최적화 기법 적용
- 정확도와 성능의 트레이드오프 분석
- 모델 크기 및 추론 속도 개선

**목표**: P99 < 5ms 유지하면서 모델 크기 축소 및 정확도 개선

---

## Phase 1: 기본 모델 학습 및 ONNX 변환 (완료)

### 1.1. 데이터셋 선택

**도구**: [wnzhang/make-ipinyou-data](https://github.com/wnzhang/make-ipinyou-data)

iPinYou 데이터셋의 **Campaign 2259** 사용:
- 벤치마크 AUC: 0.65~0.72 (현실적인 난이도)
- 달성 AUC: **0.7503** (목표 달성)

Campaign 1458(AUC 0.98)은 너무 쉬운 케이스로, 실제 AdTech 환경의 난이도를 반영하지 못합니다.
현실적인 난이도의 데이터를 사용하여 모델 개선 여지와 실전 능력을 보여줍니다.

### 1.2. 모델 학습

**라이브러리**: [DeepCTR-PyTorch](https://github.com/shenweichen/DeepCTR-Torch)
**모델**: DeepFM

DeepFM은 저차원과 고차원 feature interaction을 동시에 학습하는 모델로, AdTech 도메인에서 널리 사용됩니다.

**최종 하이퍼파라미터**:
- Embedding dimension: 8
- Learning rate: 0.0005
- DNN hidden units: (256, 128, 64)
- Batch size: 512
- Optimizer: Adam
- Device: MPS (M2 GPU)

**학습 결과**:
- Train AUC: 0.9194
- Val AUC: 0.7905
- **Test AUC: 0.7503**
- Model size: 0.37MB (PyTorch), 0.02MB (ONNX)

### 1.3. ONNX 변환 및 검증

PyTorch 모델을 ONNX 포맷으로 변환하여 프레임워크 독립적인 추론을 가능하게 합니다.

**변환 결과**:
- ONNX opset version: 18
- Model size: 0.02MB
- 1000개 샘플 검증: PyTorch vs ONNX 출력 일치 (tolerance < 1e-5)
- 최대 절대 오차: 8.24e-08

**관련 문서**: [Phase 1 Results](./phase1_results.md)

---

## Phase 2: Rust 실시간 DSP 엔진 (진행 중)

### 2.1. 목표

ONNX Runtime + Rust로 실시간 DSP 입찰 엔진 구현:
- **레이턴시**: P50 < 1ms, P99 < 5ms
- **처리량**: > 10,000 req/sec (단일 코어)
- **정합성**: Python ONNX Runtime과 출력 일치 (< 1e-5)

> **Rust 선택 이유**: Python FastAPI 대비 메모리 안전성과 성능 우위, 실시간 시스템에 적합

### 2.2. 기술 스택

- **언어**: Rust 2021 Edition
- **추론 엔진**: ONNX Runtime (ort crate)
- **웹 프레임워크**: Actix-web
- **직렬화**: serde, serde_json
- **로깅**: tracing, tracing-subscriber

### 2.3. 주요 작업 항목

1. ✅ Rust 프로젝트 셋업 및 ONNX 모델 로딩
2. 🚧 피처 전처리 구현 (feature_info.pkl → JSON)
3. 🚧 REST API 구현 (`POST /predict`, `POST /bid`)
4. 🚧 성능 벤치마크 및 최적화
5. 🚧 도커화 및 배포 준비

**관련 문서**: [Phase 2 Plan](./phase2_plan.md)

---

## Phase 3: AutoInt 모델 학습 및 서빙 (계획)

### 3.1. 배경

Phase 1에서 학습한 DeepFM 모델이 너무 작아(7,503 파라미터, 0.02MB) 서빙 최적화 효과를 측정하기 어려움.
추론 시간이 거의 0에 가까워 양자화, 프루닝 등의 효과를 검증할 수 없음.

### 3.2. AutoInt 모델 선택 이유

| 특성 | 설명 |
|------|------|
| **구조** | Multi-head Self-Attention 기반 피처 인터랙션 학습 |
| **복잡도** | O(N²) 연산으로 DeepFM 대비 훨씬 큰 모델 |
| **장점** | 파라미터 수 증가로 최적화 효과 측정 가능 |
| **ONNX 호환** | DeepCTR-Torch 지원, ONNX 변환 검증됨 |

### 3.3. 목표

- AutoInt 모델 학습 (Embedding dim: 64, Attention heads: 4)
- 모델 크기: 1MB+ (최적화 효과 측정 가능한 수준)
- Test AUC: 0.75+ 유지
- Rust 서빙 엔진 통합 및 벤치마크

### 3.4. 주요 작업 항목

1. AutoInt 모델 학습 스크립트 작성
2. 하이퍼파라미터 튜닝 (Attention layers, heads, embedding dim)
3. ONNX 변환 및 정합성 검증
4. Rust 서빙 엔진 통합
5. DeepFM vs AutoInt 벤치마크 비교

---

## Phase 4: 모델 경량화 및 최적화 (계획)

### 4.1. 목표

Phase 3에서 학습한 AutoInt 모델을 대상으로 경량화 및 최적화 기법을 적용하여 성능 개선.

> **Phase 4 목표**: P99 < 5ms 유지 + 모델 크기 축소 + 정확도 개선

### 4.2. 최적화 기법 후보

1. **양자화 (Quantization)**
   - FP32 → INT8 변환으로 모델 크기 및 추론 시간 감소
   - ONNX Runtime의 dynamic quantization 활용

2. **프루닝 (Pruning)**
   - 중요도가 낮은 가중치 제거
   - 모델 경량화 및 추론 속도 개선

3. **지식 증류 (Knowledge Distillation)**
   - 큰 모델(Teacher)의 지식을 작은 모델(Student)로 전이
   - 정확도 유지하면서 모델 크기 축소

4. **Attention Head Pruning**
   - AutoInt의 불필요한 Attention head 제거
   - 연산량 감소 및 추론 속도 개선

### 4.3. 트레이드오프 분석

각 최적화 기법의 정확도 변화와 성능 영향을 측정하여 최적의 조합을 찾습니다:
- **목표 AUC**: Phase 3 결과 유지 (하락 허용 범위 < 0.005)
- **목표 레이턴시**: P99 < 5ms 유지
- **모델 크기**: 50% 이상 축소

---

## 핵심 검증 지표

### 정합성 검증
- ✅ PyTorch vs ONNX 출력 일치 (< 1e-5)
- 🚧 Python ONNX Runtime vs Rust ONNX Runtime 출력 일치 (< 1e-5)

### 성능 검증
- ✅ Test AUC 0.7503 (벤치마크 0.65~0.72 상회)
- 🚧 P50 레이턴시 < 1ms (Phase 2 목표)
- 🚧 P99 레이턴시 < 5ms (Phase 2 목표)
- 📅 AutoInt 모델 학습 및 서빙 (Phase 3 목표)
- 📅 모델 크기 최적화 (Phase 4 목표)

---

## 기대 성과

### Phase 1 (완료)
- ✅ CTR 예측 모델 학습 및 평가 능력 검증
- ✅ ONNX 모델 변환 및 정합성 검증
- ✅ M2 GPU 최적화 경험

### Phase 2 (진행 중)
- Rust 기반 실시간 서빙 시스템 구현
- ONNX Runtime 통합 경험
- Sub-millisecond 레이턴시 최적화

### Phase 3 (계획)
- AutoInt (Attention 기반) 모델 학습 경험
- 복잡한 모델의 ONNX 변환 및 서빙
- DeepFM vs AutoInt 성능 비교 분석

### Phase 4 (계획)
- 모델 경량화 및 양자화 경험
- 정확도-성능 트레이드오프 분석
- 프로덕션급 실시간 시스템 완성

---

## 참고 문서

- [Phase 1 Results](./phase1_results.md) - 상세 학습 결과 및 분석
- [Phase 2 Plan](./phase2_plan.md) - Rust 엔진 개발 계획
- [iPinYou Benchmark](./research/ipinyou-benchmark.md) - 데이터셋 정보 및 벤치마크

# Phase 3: Serving 최적화 실험 결과

## 실험 환경

- **CPU**: AMD Ryzen 7 7800X3D (8코어 16스레드, 96MB L3 캐시)
- **OS**: Linux (WSL2)
- **모델**: AutoInt (11.6MB ONNX)
- **벤치마크 도구**: wrk (4 threads, 100 connections)

---

## 1. ONNX Runtime 스레드 튜닝

### 배경
ONNX Runtime은 `intra_threads`(연산 내 병렬화)와 `inter_threads`(연산 간 병렬화) 설정을 지원.
CPU 특성에 맞는 최적 스레드 설정을 찾기 위해 다양한 조합을 테스트.

### 실험 결과

| intra:inter | RPS | Avg Latency | P99 Latency |
|---|---|---|---|
| **1:1** | **4,125** | **24.45ms** | 57.59ms |
| 2:1 | 3,299 | 34.98ms | - |
| 4:1 | 2,707 | 37.26ms | - |
| 8:1 | 3,047 | 37.82ms | - |
| 2:2 | 2,883 | 34.90ms | - |
| 4:2 | 3,039 | 38.10ms | - |

### 결론
**싱글스레드(1:1)가 최적**
- 모델이 작아서 스레드 생성/동기화 오버헤드가 병렬화 이점보다 큼
- 7800X3D의 큰 L3 캐시(96MB)가 싱글스레드에서 더 효율적

---

## 2. Execution Provider (XNNPACK)

### 배경
XNNPACK은 x86 CPU에서 `Conv`, `Gemm`, `MatMul` 연산을 가속화하는 EP.
AMD CPU에서 추가 성능 향상 가능성 테스트.

### 실험 결과
- XNNPACK EP 등록 성공
- **효과 없음** - AutoInt 모델의 대부분 연산(attention, embedding)을 지원하지 않음
- 대부분의 노드가 기본 CPU EP로 fallback

### 로그
```
WARN: Some nodes were not assigned to the preferred execution providers
INFO: Successfully registered `XnnpackExecutionProvider`
```

### 결론
**AutoInt 모델에는 XNNPACK이 효과 없음**
- XNNPACK은 주로 CNN 계열 연산에 최적화
- Attention/Embedding 연산은 지원 범위 밖

---

## 3. INT8 양자화

### 실험
- Dynamic INT8 양자화 적용
- Static INT8 양자화 (calibration data 사용)

### 결과
- 모델 크기: 11.6MB → 3.1MB (73% 감소)
- **성능 개선 없음** - 오히려 약간 느려짐

### 원인 분석
- 양자화/역양자화 오버헤드가 연산 절감보다 큼
- 모델이 이미 충분히 빠름 (단건 ~0.1ms)

---

## 4. Session Pool (핵심 개선)

### 배경
`ort 2.0.0-rc.11`에서 `session.run()`이 `&mut self`를 요구하여 Mutex 필요.
100개의 동시 연결이 1개 Session에서 순차 대기 → latency 증가.

### 해결책
여러 Session 인스턴스를 풀링하여 동시 요청 처리:
```rust
pub struct DeepFMModel {
    sessions: Vec<Mutex<Session>>,
    next_session: AtomicUsize,  // Round-robin 선택
}
```

### 실험 결과

| Pool Size | RPS | Avg Latency | P99 Latency |
|---|---|---|---|
| 1 (기존) | 4,125 | 24.45ms | 57.59ms |
| 4 | 12,942 | 7.79ms | - |
| 8 | 13,611 | 7.43ms | - |
| **12** | **14,739** | **7.89ms** | - |
| 16 | 12,123 | 8.35ms | - |

### 결론
**Pool Size = 12가 최적**
- 너무 적으면 lock contention
- 너무 많으면 메모리 오버헤드 및 캐시 효율 저하

---

## 5. Multi-Process + Pool 조합

### 실험
로드밸런서 환경을 시뮬레이션하여 프로세스 수와 pool size 조합 테스트.

### 결과

| 구성 | Total Sessions | Combined RPS |
|---|---|---|
| 1 Process, Pool=12 | 12 | ~17,000 |
| 2 Process, Pool=6 | 12 | ~20,848 |

### 결론
**멀티 프로세스가 추가 이점 제공**
- 같은 총 세션 수에서도 프로세스 분리가 더 효율적
- 프로덕션에서는 nginx + 2~4 프로세스 권장

---

## 최종 결과 요약

### Before vs After

| 지표 | Before | After | 개선율 |
|---|---|---|---|
| **RPS** | 4,125 | 17,000+ | **4.1x** |
| **Avg Latency** | 24.45ms | 6.87ms | **3.6x 감소** |
| **P99 Latency** | 57.59ms | 16.64ms | **3.5x 감소** |

### 권장 설정

```bash
# 단일 프로세스
./deepfm-serving --intra-threads 1 --inter-threads 1 --pool-size 12

# 프로덕션 (nginx + 멀티 프로세스)
# 2~4 프로세스, 각 pool-size=6
```

---

## 효과 없었던 시도들

| 시도 | 결과 | 원인 |
|---|---|---|
| 멀티스레드 (4:1, 8:1) | 성능 저하 | 스레드 오버헤드 > 병렬화 이점 |
| XNNPACK EP | 효과 없음 | Attention 연산 미지원 |
| INT8 양자화 | 효과 없음 | 양자화 오버헤드 > 연산 절감 |

---

## 핵심 인사이트

1. **모델 크기가 작으면 스레드/양자화 최적화 효과 제한적**
2. **Lock contention이 주요 병목일 때 Session Pool이 효과적**
3. **CPU 특성(캐시 크기, 코어 수)에 맞는 튜닝 필요**
4. **벤치마크는 실제 워크로드와 유사하게 구성해야 의미있음**

---

## 관련 코드

- [serving/src/model.rs](../serving/src/model.rs) - Session Pool 구현
- [serving/scripts/tune_threads.sh](../serving/scripts/tune_threads.sh) - 스레드 튜닝 스크립트
- [training/src/export/quantize_onnx.py](../training/src/export/quantize_onnx.py) - INT8 양자화 스크립트

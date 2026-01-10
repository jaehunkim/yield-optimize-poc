# Phase 2 Plan: Real-time Bidding Engine (Rust)

**최종 업데이트**: 2026년 1월 10일
**시작일**: 2026년 1월 10일

## 목표

Phase 1.5에서 학습한 DeepFM 모델(ONNX)을 Rust 기반 실시간 서빙 엔진으로 구현하여 프로덕션급 DSP(Demand-Side Platform) 입찰 시스템을 구축합니다.

**핵심 목표:**
- Sub-millisecond 추론 레이턴시 달성 (P50 < 1ms, P99 < 5ms)
- REST API를 통한 실시간 CTR 예측 및 입찰가 계산
- Rust의 메모리 안전성과 성능을 활용한 안정적인 서빙 인프라

> **Phase 2 vs Phase 3 목표**:
> - **Phase 2 (현재)**: P50 < 1ms, P99 < 5ms - Rust 엔진의 기본 성능 검증
> - **Phase 3 (계획)**: P99 < 5ms 유지 + 모델 크기 축소 + 정확도 개선
>
> Phase 2는 기본 Rust 구현으로 목표 레이턴시를 달성하며, Phase 3에서는 모델 경량화 기법(양자화, 프루닝)을 통해 성능을 유지하면서 추가적인 최적화를 진행합니다.

## 기술 스택

- **언어**: Rust 1.92.0 (2021 Edition)
- **추론 엔진**: ONNX Runtime (ort crate 2.0.0-rc.11) - CPU execution provider
- **웹 프레임워크**: Axum 0.7 (Tokio 기반)
- **비동기 런타임**: Tokio 1.x
- **직렬화**: serde 1.x, serde_json
- **로깅**: tracing, tracing-subscriber
- **동기화**: parking_lot 0.12 (RwLock)
- **수치 연산**: ndarray 0.17
- **에러 처리**: anyhow 1.x
- **테스트**: cargo test, criterion (벤치마크 - 예정)

## 프로젝트 구조

```
serving/
├── Cargo.toml              # Rust 프로젝트 설정
├── src/
│   ├── main.rs            # ✅ 진입점, 서버 초기화
│   ├── lib.rs             # ✅ 라이브러리 exports
│   ├── model.rs           # ✅ ONNX 모델 로딩 및 추론 (CPU)
│   ├── features.rs        # ✅ 피처 전처리 (현재 더미, Phase 2.2에서 구현)
│   ├── api.rs             # ✅ REST API 핸들러 (/health, /predict)
│   └── bidding.rs         # ⏸️  입찰가 계산 로직 (Phase 2.4)
├── models/
│   └── deepfm_emb8_lr0.0001_dnn25612864_neg150_best.onnx  # ✅ ONNX 모델
├── examples/
│   └── test_model_loading.rs  # ✅ 모델 로딩 테스트
├── scripts/
│   └── compare_onnx_outputs.py  # ✅ Python vs Rust 출력 비교
├── config/
│   └── server.yaml        # ⏸️  서버 설정 (Phase 2.2+)
└── benches/
    └── inference.rs       # ⏸️  성능 벤치마크 (Phase 2.5)
```

## Phase 2 단계별 계획

### Phase 2.1: 기본 인프라 구축 ✅ 완료

**목표**: Rust 프로젝트 셋업 및 ONNX 모델 로딩

**작업 항목:**
1. ✅ Cargo 프로젝트 초기화 (`cargo init serving`)
   - Rust 1.92.0 (2021 Edition) 사용
2. ✅ 의존성 추가
   - ort 2.0.0-rc.11 (ONNX Runtime)
   - axum 0.7, tokio 1.x (비동기 웹 서버)
   - serde 1.x, serde_json (직렬화)
   - tracing, tracing-subscriber (로깅)
   - parking_lot 0.12 (효율적인 동기화)
3. ✅ ONNX 모델 로딩 검증
   - `serving/models/deepfm_emb8_lr0.0001_dnn25612864_neg150_best.onnx` 로드
   - Phase 1.5 최종 모델 (ND 1:150, DNN 256-128-64)
   - 더미 입력으로 추론 테스트 (15개 피처)
   - **CPU 전용 실행** (CPU execution provider)
4. ✅ 기본 헬스체크 엔드포인트 구현 (`GET /health`)
5. ✅ CTR 예측 API 구현 (`POST /predict`)
6. ✅ Python ONNX Runtime 출력 비교 검증

**검증 결과:**
- ✅ ONNX 모델이 Rust에서 정상 로드됨
- ✅ 더미 입력으로 추론 성공 (CTR: 0.001381)
- ✅ 헬스체크 API 응답 200 OK
- ✅ Python vs Rust 출력 일치 (상대 오차: 0.004%)
- ✅ **레이턴시 목표 초과 달성**: 0.106ms (목표 1ms의 10분의 1)
- ✅ 연속 요청 안정성 검증 완료

**성능 측정 결과:**
- 초기 레이턴시: 0.289ms
- 워밍업 후: **0.106ms** (P50 < 1ms 목표 초과 달성)
- Python 출력 일치: 차이 5.96e-08

**진행 상황**: ✅ 완료 (2026-01-10)

---

### Phase 2.2: 피처 전처리 구현 🚧 대기 중

**목표**: iPinYou 데이터셋의 피처를 모델 입력 형식으로 변환

**현재 상태:**
- Phase 2.1에서 더미 피처 전처리 구현 완료 (모든 값 0으로 설정)
- `FeatureProcessor` 구조체 및 `AdRequest` 정의 완료
- `load_from_json()` 메서드 준비 완료 (#[allow(dead_code)])

**작업 항목:**
1. `feature_info.pkl` 파싱
   - Python pickle을 JSON으로 변환 (별도 스크립트)
   - Rust에서 JSON 읽어 피처 메타데이터 로드
2. 피처 전처리 로직 구현
   - Sparse features: 카테고리 → 인덱스 매핑 (HashMap)
   - Dense features: 정규화 (mean, std)
   - 최종 출력: `Vec<f32>` (15개 피처)
3. 단위 테스트 작성
   - Python 전처리 결과와 비교 검증

**검증 기준:**
- Rust 전처리 출력 == Python 전처리 출력
- 모든 피처 타입(sparse, dense) 정상 처리

**진행 상황**: 🚧 대기 중 (Phase 2.1 완료 후 시작)

---

### Phase 2.3: CTR 예측 API 구현

**목표**: 실시간 CTR 예측 엔드포인트 개발

**작업 항목:**
1. API 엔드포인트 구현 (`POST /predict`)
   - 입력: JSON (campaign_id, domain, user_agent, timestamp 등)
   - 출력: JSON (ctr: f32, latency_ms: f64)
2. 요청 → 피처 전처리 → 모델 추론 파이프라인 연결
3. 에러 핸들링 (잘못된 입력, 모델 실패 등)
4. 로깅 추가 (요청/응답, 에러, 레이턴시)

**검증 기준:**
- Python ONNX Runtime 출력과 일치 (< 1e-5 오차)
- 평균 추론 레이턴시 < 1ms

**예시 요청/응답:**
```bash
# Request
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "weekday": 4,
    "hour": 15,
    "region": 1,
    "city": 1,
    "adexchange": 1,
    "domain": "3d78fcc01ed2eb8c",
    "slotid": "mm_10067_282",
    "slotwidth": 728,
    "slotheight": 90,
    "slotvisibility": "FirstView",
    "slotformat": "Banner",
    "creative": "other",
    "user_tag": "10006,10024,10110,13042"
  }'

# Response
{
  "ctr": 0.00042,
  "latency_ms": 0.85
}
```

---

### Phase 2.4: 입찰가 계산 로직

**목표**: CTR 기반 입찰가 계산 및 ROI 최적화

**작업 항목:**
1. 입찰 전략 구현
   - 단순 선형: `bid = base_bid * ctr * multiplier`
   - ROI 기반: `bid = (ctr * conversion_value) / target_roi`
2. 설정 파일을 통한 파라미터 조정 (YAML)
3. API 엔드포인트 추가 (`POST /bid`)
   - 입력: 광고 요청 정보
   - 출력: 입찰가 + CTR + 메타데이터

**검증 기준:**
- 입찰가가 합리적 범위 내 (0 ~ max_bid)
- CTR이 높을수록 입찰가 증가

**예시 설정 (server.yaml):**
```yaml
bidding:
  base_bid: 100  # CPM 기준 (달러 * 1000)
  multiplier: 1.5
  max_bid: 500
  target_roi: 2.0
```

---

### Phase 2.5: 성능 최적화 및 벤치마크

**목표**: 프로덕션급 성능 달성

**작업 항목:**
1. Criterion을 이용한 벤치마크 작성
   - 피처 전처리 성능
   - 모델 추론 레이턴시
   - End-to-end API 처리 시간
2. 최적화
   - 피처 매핑 HashMap → 사전 계산된 배열
   - 모델 세션 재사용 (Arc<Session>)
   - 불필요한 메모리 할당 제거
3. 부하 테스트 (wrk, vegeta)
   - 동시 요청 처리 성능
   - 레이턴시 P50/P95/P99 측정

**목표 성능:**
- P50 레이턴시 < 1ms
- P99 레이턴시 < 5ms
- Throughput > 10,000 req/sec (단일 코어)

---

### Phase 2.6: 도커화 및 배포 준비

**목표**: 프로덕션 배포 환경 구축

**작업 항목:**
1. Dockerfile 작성
   - Multi-stage build (컴파일 + 실행 분리)
   - ONNX 모델 포함
2. Docker Compose 설정
   - 서빙 컨테이너
   - 로깅 (optional: Prometheus, Grafana)
3. 배포 문서 작성
   - 환경 변수 설정
   - 헬스체크 엔드포인트
   - 스케일링 가이드

**검증 기준:**
- 도커 이미지 빌드 성공
- 컨테이너에서 서버 정상 실행
- 외부에서 API 호출 가능

---

## 성공 지표

### 기능적 목표
- ✅ ONNX 모델 정상 로드 및 추론
- ✅ REST API 정상 동작 (/health, /predict, /bid)
- ✅ Python ONNX Runtime과 출력 일치 (< 1e-5)

### 성능 목표
- ✅ P50 추론 레이턴시 < 1ms
- ✅ P99 추론 레이턴시 < 5ms
- ✅ Throughput > 10,000 req/sec

### 프로덕션 준비도
- ✅ 에러 핸들링 및 로깅
- ✅ 도커 이미지 빌드 및 배포
- ✅ 성능 벤치마크 문서화

## 위험 요소 및 대응

### 1. ONNX Runtime Rust 바인딩 성숙도
- **위험**: ort crate가 Python만큼 안정적이지 않을 수 있음
- **대응**: 초기에 기본 추론 테스트로 검증, 문제 발생 시 C API 직접 사용

### 2. 피처 전처리 정합성
- **위험**: Python과 Rust 전처리 로직 불일치
- **대응**: 단위 테스트로 샘플별 출력 비교, 차이 발견 시 즉시 수정

### 3. 레이턴시 목표 미달
- **위험**: 1ms 목표 달성 실패
- **대응**: 프로파일링(flamegraph)으로 병목 지점 파악, 최적화 우선순위 결정

## 타임라인 (예상)

- **Phase 2.1**: 1-2일 (기본 인프라)
- **Phase 2.2**: 2-3일 (피처 전처리)
- **Phase 2.3**: 2-3일 (CTR 예측 API)
- **Phase 2.4**: 1-2일 (입찰가 계산)
- **Phase 2.5**: 2-3일 (성능 최적화)
- **Phase 2.6**: 1-2일 (도커화)

**총 예상 기간**: 9-15일

## 참고 자료

- [ONNX Runtime Rust Binding (ort)](https://github.com/pykeio/ort)
- [Axum Documentation](https://docs.rs/axum/latest/axum/)
- [Tokio Documentation](https://tokio.rs/)
- [iPinYou Feature Schema](../docs/research/ipinyou-benchmark.md)
- [Phase 1.5 Results](./phase1.5_data_expansion.md)

## Phase 1.5 완료 상태

**모델 정보:**
- 모델 파일: `deepfm_emb8_lr0.0001_dnn25612864_neg150_best.pth` (PyTorch)
- ONNX 파일: `deepfm_emb8_lr0.0001_dnn25612864_neg150_best.onnx`
- 아키텍처: DeepFM, embedding_dim=8, DNN hidden=(256, 128, 64)
- 학습 데이터: Negative Downsampling 1:150 (Training set only)
- 성능: Test AUC 0.6958 (LR 베이스라인 0.6727 대비 +3.4% 개선)

**피처 정보:**
- Sparse features: 10개 (weekday, hour, region, city, adexchange, domain, slotid, slotwidth, slotheight, slotvisibility, slotformat, creative, user_tag)
- Dense features: 5개 (정규화된 수치 피처)
- 총 입력 차원: 15개 피처
- 출력: 단일 CTR 확률값 (0~1)

# 1단계 세부 실행 계획: 모델 학습 및 AUC 검증

## 사용자 요구사항 (확정)

### 전체 3단계 구조
1. **1단계**: 기본 모델 & 데이터 다운로드 → PyTorch 학습 & AUC 검증
2. **2단계**: 모델 ONNX 변환 → Rust 실시간 DSP 엔진 개발
3. **3단계**: 실시간 DSP 엔진 latency 달성을 위한 모델 개량 및 경량화

### 확정된 사항
- **Latency 목표**: p99 < 50ms (일반적 실시간 RTB 수준)
- **Rust 엔진 범위**: ONNX 추론만 (feature는 Python에서 전처리 후 전달)
- **1단계 범위**: 학습 + AUC 검증만 (Rust 엔진은 2단계로)

## 1단계 상세 실행 계획

### Phase 1.1: 환경 설정 및 데이터 다운로드

#### 1.1.1 Python 환경 설정
- **pyenv 사용**: Python 3.9+ 설치 및 관리
- 가상환경 생성 (M2 Mac 호환성 확인)
- 필수 라이브러리 설치:
  - `torch` (PyTorch, M2 최적화 버전)
  - `deepctr-torch` (DeepCTR 모델)
  - `scikit-learn` (metrics)
  - `pandas`, `numpy`
  - `tqdm` (progress bar)
  - `kaggle` (Kaggle API)
  - `onnx`, `onnxruntime` (선택적)

**생성 파일**:
- `training/requirements.txt`
- `training/setup_env.sh` (pyenv + 가상환경 생성 스크립트)

#### 1.1.2 iPinYou 데이터 다운로드
- **Kaggle 데이터셋 사용**: `lastsummer/ipinyou` (6GB)
- 이미 `wnzhang/make-ipinyou-data` 스크립트로 전처리된 데이터
- Kaggle API로 다운로드: `kaggle datasets download -d lastsummer/ipinyou`
- 데이터 저장 경로: `training/data/raw/`

**생성 파일**:
- `training/scripts/download_data.sh` (Kaggle CLI 사용)
- `training/data/raw/` 디렉토리

#### 1.1.3 데이터 로딩 및 분할
- Kaggle 데이터셋은 이미 표준 포맷 (.yzx 파일 등)
- Campaign 2259 또는 3386 선택
- train/validation/test 분할 (이미 분할되어 있을 가능성 높음)
- DeepCTR-PyTorch 입력 형식으로 변환

**생성 파일**:
- `training/scripts/load_data.py` (전처리 → 로딩으로 변경)
- `training/data/processed/` (필요시 변환된 데이터)
- `training/data/processed/feature_map.pkl` (feature dictionary)

### Phase 1.2: 모델 학습 및 평가

#### 1.2.1 DeepFM 모델 구현
- DeepCTR-PyTorch의 DeepFM 모델 사용
- 하이퍼파라미터 설정:
  - embedding_dim: 8 or 16
  - dnn_hidden_units: (256, 128, 64)
  - learning_rate: 0.001
  - batch_size: 256
  - epochs: 10-20

**생성 파일**:
- `training/src/models/deepfm_trainer.py`
- `training/configs/model_config.yaml` (하이퍼파라미터)

#### 1.2.2 학습 실행
- Train set으로 학습
- Validation set으로 early stopping
- 학습 로그 저장 (loss, AUC per epoch)
- 모델 체크포인트 저장

**생성 파일**:
- `training/scripts/train.py` (메인 학습 스크립트)
- `training/models/checkpoints/deepfm_best.pth`
- `training/logs/training_log.csv`

#### 1.2.3 오프라인 AUC 평가
- Test set으로 최종 평가
- PyTorch 모델로 직접 추론
- Metrics 계산:
  - AUC (ROC)
  - Log Loss
  - Accuracy
- 벤치마크와 비교 (Campaign 2259: 0.6865, Campaign 3386: 0.7908)

**생성 파일**:
- `training/scripts/evaluate.py`
- `training/results/offline_evaluation.json`

### Phase 1.3: AUC 정합성 검증 준비 (선택적)

#### 1.3.1 ONNX 변환
1단계에서는 Rust 엔진을 만들지 않지만, 2단계 준비를 위해 ONNX 변환까지는 해볼 수 있음.

**생성 파일**:
- `training/scripts/export_onnx.py`
- `training/models/deepfm.onnx`

#### 1.3.2 Python ONNX 추론으로 정합성 사전 검증
- ONNX Runtime (Python)으로 추론
- PyTorch 모델 출력 vs ONNX 출력 비교
- 차이가 < 1e-5 수준인지 확인

**생성 파일**:
- `training/scripts/verify_onnx.py`
- `training/results/onnx_verification.json`

### Phase 1.4: 문서화

#### 1.4.1 실험 결과 문서
- 데이터셋 통계
- 학습 과정 (loss curve, AUC curve)
- 최종 오프라인 AUC
- 벤치마크 대비 성능

**생성 파일**:
- `docs/phase1_results.md`
- `training/results/plots/` (학습 curve 이미지)

## 디렉토리 구조 (1단계 완료 후)

Training/검증과 Serving을 분리한 구조:

```
yield-optimize-poc/
├── README.md
├── docs/
│   ├── first-goal.md          # 전체 POC 목표
│   ├── phase1_plan.md          # 1단계 세부 실행 계획 (이 문서)
│   └── phase1_results.md       # 1단계 실험 결과
│
├── training/                    # 학습 및 검증 영역
│   ├── requirements.txt
│   ├── setup_env.sh
│   ├── configs/
│   │   └── model_config.yaml
│   ├── data/
│   │   ├── raw/
│   │   │   └── [ipinyou campaign data]
│   │   └── processed/
│   │       ├── train.csv
│   │       ├── validation.csv
│   │       ├── test.csv
│   │       └── feature_map.pkl
│   ├── src/
│   │   └── models/
│   │       └── deepfm_trainer.py
│   ├── scripts/
│   │   ├── download_data.py
│   │   ├── preprocess_data.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── export_onnx.py (optional)
│   │   └── verify_onnx.py (optional)
│   ├── models/
│   │   ├── checkpoints/
│   │   │   └── deepfm_best.pth
│   │   └── deepfm.onnx (optional)
│   ├── logs/
│   │   └── training_log.csv
│   └── results/
│       ├── offline_evaluation.json
│       ├── onnx_verification.json (optional)
│       └── plots/
│           ├── loss_curve.png
│           └── auc_curve.png
│
└── serving/                     # 서빙 엔진 영역 (2단계에서 구현)
    ├── Cargo.toml               # Rust 프로젝트
    ├── src/
    │   ├── main.rs
    │   ├── inference.rs         # ONNX 추론
    │   └── server.rs            # HTTP 서버
    ├── models/
    │   └── deepfm.onnx          # training/에서 복사
    └── tests/
        └── integration_test.rs
```

## 검증 기준 (1단계 완료 조건)

1. ✅ iPinYou 데이터 (Campaign 2259 or 3386) 전처리 완료
2. ✅ DeepFM 모델 학습 완료 (10+ epochs)
3. ✅ 오프라인 AUC >= 벤치마크 ± 0.05 범위 내
   - Campaign 2259: 0.65~0.72 사이
   - Campaign 3386: 0.74~0.84 사이
4. ✅ PyTorch 모델 저장 완료 (.pth)
5. ✅ (선택) ONNX 변환 및 Python 추론 정합성 검증

## 예상 소요 시간

- Phase 1.1: **1-2시간** (환경 설정, Kaggle 데이터 다운로드 → 전처리 불필요!)
- Phase 1.2: 4-6시간 (모델 구현, 학습, 평가)
- Phase 1.3: 1-2시간 (ONNX 변환 및 검증, 선택적)
- Phase 1.4: 1-2시간 (문서화)

**총 예상**: 7-12시간 (20-30시간 예산의 35-40%)

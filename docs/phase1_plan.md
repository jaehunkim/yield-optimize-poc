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

#### 1.1.3 데이터 로딩 및 분할 ✅
- **Raw 데이터 읽기**: Kaggle 데이터셋은 이미 feature 추출된 탭 구분 텍스트 (bz2 압축)
- **클릭 레이블 추가**: impression 파일 + click 파일 매칭 (bid_id로 조인, set lookup)
- **DeepCTR 형식 변환**:
  - Categorical features → pd.Categorical().codes로 정수 인코딩
  - Numerical features → (x - mean) / std로 정규화
- **데이터 분할**: 시간순 split (72%/8%/20%) - data leakage 방지

**생성 파일**:
- ✅ `training/src/data/load_data.py` (iPinYouDataLoader 클래스)
- ✅ `training/data/processed/` (train.csv, val.csv, test.csv)
- ✅ `training/data/processed/feature_info.pkl` (feature 메타데이터)

#### 1.1.4 파이프라인 검증 ✅
- **LR baseline**: One-Hot Encoding + Sparse matrix
- **결과**: Campaign 2259, Test AUC 0.7068 (벤치마크 0.65~0.72 범위 내)
- **검증 완료**: 데이터 로딩, 전처리, 분할 모두 정상 작동

**생성 파일**:
- ✅ `training/src/validate/pipeline.py`
- ✅ `training/scripts/validate_pipeline.sh`

### Phase 1.2: 모델 학습 및 평가

#### 1.2.1 DeepFM 모델 구현 ✅
- DeepCTR-PyTorch의 DeepFM 모델 사용
- M2 GPU (MPS) 가속 지원
- 멀티프로세싱 데이터 로더
- Early stopping 자동 적용

**하이퍼파라미터:**
- embedding_dim: 8 or 16
- dnn_hidden_units: (256, 128, 64)
- learning_rate: 0.001 or 0.0005
- batch_size: 512
- epochs: 10-15
- patience: 3

**생성 파일:**
- ✅ `training/src/models/deepfm_trainer.py`
- ✅ `training/src/train/train.py`

#### 1.2.2 학습 실행 ✅

**Shell 스크립트 실행:**
```bash
# 기본 설정 (embedding=16, lr=0.001)
./training/scripts/train_deepfm.sh

# 커스텀 설정
./training/scripts/train_deepfm.sh <campaign> <days> <epochs> <batch> <workers> <emb_dim> <lr>

# 예시: 과적합 완화 설정
./training/scripts/train_deepfm.sh 2259 3 10 512 4 8 0.0005
```

**Python 직접 실행:**
```bash
python training/src/train/train.py \
    --campaign 2259 \
    --days 3 \
    --epochs 10 \
    --embedding-dim 16 \
    --lr 0.001 \
    --device auto
```

**생성 파일:**
- ✅ `training/models/deepfm_best.pth` - 최고 성능 모델
- ✅ `training/models/training_history.json` - 에포크별 학습 로그

#### 1.2.3 오프라인 AUC 평가 ✅

**자동 실행:** `train_deepfm.sh` 실행 시 자동으로 평가 포함

**수동 실행:**
```bash
python training/src/evaluate/evaluate.py \
    --campaign 2259 \
    --days 3 \
    --device auto
```

**Metrics:**
- AUC (ROC)
- Log Loss
- CTR (실제 클릭률)

**생성 파일:**
- ✅ `training/results/evaluation.json` - Train/Val/Test 결과

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

### Phase 1.4: 문서화 ✅

#### 1.4.1 실험 결과 문서 ✅
- 데이터셋 통계 (Campaign 2259, 3일치 데이터)
- 학습 과정 (epoch별 Train/Val AUC, Loss)
- 최종 오프라인 AUC: 0.7191 (벤치마크 범위 내)
- 벤치마크 대비 성능: LR baseline 0.7068 → DeepFM 0.7191 (+1.7%)

**생성 파일**:
- ✅ `docs/phase1_results.md` - Experiment 1 결과 및 과적합 분석

## 디렉토리 구조 (실제 구현)

```
yield-optimize-poc/
├── README.md
├── CLAUDE.md                    # Claude 컨텍스트 (프로젝트 원칙)
├── docs/
│   ├── first-goal.md           # 전체 POC 목표
│   ├── phase1_plan.md          # 1단계 세부 실행 계획 (이 문서)
│   ├── phase1_results.md       # 1단계 실험 결과
│   └── research/               # 리서치 자료
│       ├── README.md
│       ├── ipinyou-benchmark.md
│       └── 2014-bench-bottomline.pdf
│
├── venv/                        # 가상환경 (루트 레벨)
│
└── training/                    # 학습 및 검증 영역
    ├── requirements.txt
    ├── data/
    │   ├── raw/                 # Kaggle 원본 데이터
    │   └── processed/           # 전처리된 데이터 (캠페인/days별 서브디렉토리)
    │       └── campaign_2259/
    │           └── 3days/
    │               ├── train.csv
    │               ├── val.csv
    │               ├── test.csv
    │               └── feature_info.pkl
    ├── src/
    │   ├── data/
    │   │   └── load_data.py    # 데이터 로더
    │   ├── models/
    │   │   └── deepfm_trainer.py  # DeepFM 트레이너 (MPS 지원)
    │   ├── validate/
    │   │   └── pipeline.py     # 파이프라인 검증 + LR baseline
    │   ├── train/
    │   │   └── train.py        # 학습 스크립트
    │   └── evaluate/
    │       └── evaluate.py     # 평가 스크립트
    ├── scripts/                 # Shell 스크립트만
    │   ├── download_data.sh    # Kaggle 다운로드
    │   ├── validate_pipeline.sh  # 파이프라인 검증
    │   ├── train_deepfm.sh     # DeepFM 학습 + 평가
    │   └── evaluate_deepfm.sh  # DeepFM 평가 전용
    ├── models/
    │   ├── deepfm_emb8_lr0.0005_best.pth     # 최종 모델 (Exp 2)
    │   ├── deepfm_emb16_lr0.001_best.pth     # Exp 1 모델
    │   └── training_history_emb8_lr0.0005.json  # 학습 히스토리
    └── results/
        └── evaluation.json     # 평가 결과
```

## 검증 기준 (1단계 완료 조건)

1. ✅ iPinYou 데이터 (Campaign 2259) 전처리 완료
2. ✅ DeepFM 모델 학습 완료 (2회 실험)
3. ✅ 오프라인 AUC >= 벤치마크 범위 내
   - **Campaign 2259: Test AUC 0.7503** (목표: 0.65~0.72, 상회 달성)
4. ✅ PyTorch 모델 저장 완료 (하이퍼파라미터 포함 파일명)
5. ⏸️ (선택) ONNX 변환 및 Python 추론 정합성 검증 - Phase 2로 이연

## 실제 소요 시간

- Phase 1.1: ✅ 완료 (환경 설정, Kaggle 데이터 다운로드, 전처리)
- Phase 1.2: ✅ 완료 (모델 구현, 실험 2회, 총 학습 11.5분)
  - Experiment 1: 5분 (Test AUC 0.7191)
  - Experiment 2: 6.5분 (Test AUC 0.7503)
- Phase 1.3: ⏸️ Phase 2로 이연 (ONNX 변환)
- Phase 1.4: ✅ 완료 (문서화)

**Phase 1 완료**: 벤치마크 목표 달성 및 상회 (Test AUC 0.7503)

## 추가 개선 사항

### 모델 파일명 자동화 ✅
- **저장 형식**: `deepfm_emb{dim}_lr{lr}_best.pth`
- **자동 파싱**: evaluate.py가 파일명에서 하이퍼파라미터 자동 감지
- **히스토리**: `training_history_emb{dim}_lr{lr}.json`
- **효과**: 실험 관리 용이, 평가 시 파라미터 불일치 방지

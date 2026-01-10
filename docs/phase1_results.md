# Phase 1 실험 결과

## 목표
- iPinYou Campaign 2259 데이터로 DeepFM 모델 학습
- 벤치마크 AUC 범위 달성: 0.65~0.72

## 데이터셋
- **Campaign**: 2259
- **Days**: 3일치 데이터
- **Train**: 161,512 samples (CTR: 0.03%)
- **Val**: 18,364 samples (CTR: 0.03%)
- **Test**: 45,911 samples (CTR: 0.03%)
- **Features**: 17 sparse + 2 dense = 19 features

## Baseline: Logistic Regression
- **방법**: One-Hot Encoding + Sparse Matrix
- **Solver**: SAGA (L2 regularization, class_weight='balanced')
- **Max iterations**: 200
- **결과**:
  - Train AUC: 0.9608
  - Val AUC: 0.7289
  - **Test AUC: 0.7068**

## DeepFM 실험

### 실험 1: 기본 설정 (Embedding Dim 16)

**하이퍼파라미터:**
- Embedding dimension: 16
- DNN hidden units: (256, 128, 64)
- Learning rate: 0.001
- Batch size: 512
- Optimizer: Adam
- Early stopping patience: 3
- Device: MPS (M2 GPU)

**학습 과정:**
```
Epoch 1/10 (76.3s) - Train Loss: 0.0187, Train AUC: 0.6220, Val AUC: 0.7464
  -> Best model saved
Epoch 2/10 (75.1s) - Train Loss: 0.0027, Train AUC: 0.8553, Val AUC: 0.7301
Epoch 3/10 (75.5s) - Train Loss: 0.0024, Train AUC: 0.9068, Val AUC: 0.7415
Epoch 4/10 (75.4s) - Train Loss: 0.0023, Train AUC: 0.9106, Val AUC: 0.7305
Early stopping at epoch 4
```

**최종 결과 (Best Val AUC: 0.7464):**
- Train AUC: 0.8717
- Val AUC: 0.7464
- **Test AUC: 0.7191**
- Test LogLoss: 0.0029

**분석:**
- ✅ **목표 달성**: Test AUC 0.7191 (벤치마크 0.65~0.72 범위 내)
- ⚠️ **과적합 징후**: Train AUC (0.8717) >> Val AUC (0.7464)
- ✅ **LR 대비 개선**: +0.0123 (1.7% 향상)
- ⚠️ **Epoch 1이 최고**: 이후 과적합으로 성능 저하
- ⚠️ **데이터 불균형**: Positive samples 158개 vs 모델 파라미터 669,940개

**학습 시간:**
- 에폭당 평균: 75초
- 총 학습 시간: 5분 (4 epochs)
- 평가 시간: 약 30초

## 벤치마크 비교

| Method | Test AUC | 벤치마크 범위 | 달성 여부 |
|--------|----------|--------------|----------|
| LR (Label Encoding) | - | - | - |
| LR (One-Hot Encoding) | 0.7068 | 0.65~0.72 | ✅ |
| DeepFM (Emb 16) | **0.7191** | 0.65~0.72 | ✅ |
| 논문 벤치마크 | 0.65~0.72 | - | - |

## 주요 발견

### 1. 과적합 문제
- 클릭 데이터(158개)가 모델 파라미터(66만개)에 비해 매우 부족
- Epoch 2부터 Train AUC는 상승하지만 Val AUC는 하락
- Early stopping이 효과적으로 작동

### 2. M2 GPU 성능
- MPS 가속 정상 작동
- 에폭당 75초 (161K samples, batch size 512)
- CPU 대비 예상 3~5배 빠름

### 3. DeepCTR 라이브러리
- ✅ 빠른 프로토타이핑 가능
- ✅ MPS 지원
- ⚠️ 입력 형식 주의 필요 (concatenated tensor)

## 실험 2: 과적합 완화 ✅

### 하이퍼파라미터
- **Embedding dimension**: 8 (16에서 축소)
- **Learning rate**: 0.0005 (0.001에서 축소)
- **DNN hidden units**: (256, 128, 64)
- **Batch size**: 512
- **Optimizer**: Adam
- **Early stopping patience**: 3
- **Device**: MPS (M2 GPU)

### 학습 과정
```
Epoch 1/10 (77.5s) - Train Loss: 0.0274, Train AUC: 0.6390, Val AUC: 0.7517
  -> Best model saved
Epoch 2/10 (77.3s) - Train Loss: 0.0027, Train AUC: 0.8138, Val AUC: 0.7905
  -> Best model saved
Epoch 3/10 (76.8s) - Train Loss: 0.0025, Train AUC: 0.8900, Val AUC: 0.7702
Epoch 4/10 (78.2s) - Train Loss: 0.0023, Train AUC: 0.9269, Val AUC: 0.7667
Epoch 5/10 (78.0s) - Train Loss: 0.0022, Train AUC: 0.9311, Val AUC: 0.7608
Early stopping at epoch 5
```

### 최종 결과 (Best Val AUC: 0.7905)
- Train AUC: 0.9194
- Val AUC: 0.7905
- **Test AUC: 0.7503**
- Test LogLoss: 0.0028

### 분석
- ✅ **목표 달성**: Test AUC 0.7503 (벤치마크 범위 내, Exp 1 대비 +0.0312)
- ✅ **Val AUC 대폭 개선**: 0.7464 → 0.7905 (+0.0441)
- ⚠️ **여전히 과적합**: Train AUC (0.9194) >> Val AUC (0.7905)
- ✅ **일반화 개선**: Epoch 2가 최고 성능 (Exp 1은 Epoch 1)
- ✅ **모델 경량화 효과**: 파라미터 감소 + 성능 향상

### Experiment 1 vs 2 비교

| Metric | Exp 1 (emb=16, lr=0.001) | Exp 2 (emb=8, lr=0.0005) | 개선폭 |
|--------|--------------------------|--------------------------|--------|
| Train AUC | 0.8717 | 0.9194 | +0.0477 |
| Val AUC | 0.7464 | 0.7905 | **+0.0441** |
| Test AUC | 0.7191 | **0.7503** | **+0.0312** |
| Best Epoch | 1 | 2 | - |
| 학습 시간 | 5분 (4 epochs) | 6.5분 (5 epochs) | - |

## 결론

✅ **Phase 1 완료**
- LR Baseline: Test AUC 0.7068
- DeepFM Exp 1: Test AUC 0.7191 (벤치마크 달성)
- **DeepFM Exp 2: Test AUC 0.7503 (최종 모델, 벤치마크 상회)**

### 주요 성과
1. ✅ 벤치마크 목표(0.65~0.72) 달성 및 상회
2. ✅ 과적합 완화 성공 (Val AUC 0.7464 → 0.7905)
3. ✅ M2 GPU 최적화 (MPS 가속, 에폭당 ~77초)
4. ✅ 모델 파일명 자동화 (하이퍼파라미터 포함)

### 학습한 교훈
- **모델 크기 축소 + 낮은 LR** = 일반화 성능 향상
- 극도의 데이터 불균형(CTR 0.03%)에서도 DeepFM 효과적
- Early stopping 필수 (Epoch 2가 최적)

**다음 단계:**
- Phase 1.3 (선택): ONNX 변환 및 Python 정합성 검증
- Phase 2: Rust 실시간 DSP 엔진 개발

## 생성 파일
- `training/models/deepfm_emb16_lr0.001_best.pth` - Experiment 1 모델
- `training/models/deepfm_emb8_lr0.0005_best.pth` - **Experiment 2 모델 (최종)**
- `training/models/training_history_emb8_lr0.0005.json` - 학습 히스토리
- `training/results/evaluation.json` - 최종 평가 결과

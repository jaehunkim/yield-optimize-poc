# Yield Optimization POC

AdTech 포지션 지원용 POC - iPinYou 데이터셋으로 DeepFM CTR 예측 모델 학습

## Quick Start

```bash
# 1. 환경 설정
./training/setup_env.sh
source activate.sh

# 2. Kaggle API 토큰 설정 (training/.env)
KAGGLE_API_TOKEN=your_token_here

# 3. 데이터 다운로드 및 학습
./training/scripts/download_data.sh
python training/scripts/train.py --campaign 2259
```

## 목표

- **Phase 1**: PyTorch DeepFM 학습 및 AUC 검증 (Campaign 2259: 0.65~0.72)
- **Phase 2**: ONNX + Rust 서빙 엔진
- **Phase 3**: 레이턴시 최적화 (p99 < 50ms)

자세한 내용은 [Phase 1 계획](docs/phase1_plan.md) 참고

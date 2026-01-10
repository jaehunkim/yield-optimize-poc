# AdTech POC - First Goal

## 프로젝트 개요

AdTech 포지션 지원을 위한 POC(Proof of Concept) 프로젝트입니다.
실시간 CTR(Click-Through Rate) 예측 모델의 학습부터 서빙까지의 전체 파이프라인을 구현하여 기본적인 개발 능력을 검증합니다.

**예상 소요 시간**: 20-30시간
**개발 환경**: MacBook M2

## 핵심 목표

오프라인에서 학습된 모델의 AUC와 실시간 서빙 환경에서 예측된 결과의 AUC 정합성을 검증합니다.
이는 모델이 프로덕션 환경에서도 학습 시와 동일한 성능을 발휘하는지 확인하는 중요한 지표입니다.

## 1단계: 데이터 전처리

**도구**: [wnzhang/make-ipinyou-data](https://github.com/wnzhang/make-ipinyou-data)

iPinYou 데이터셋의 **Campaign 2259** 또는 **Campaign 3386**을 사용합니다.
- Campaign 2259: 벤치마크 AUC 0.6865 (현실적인 난이도)
- Campaign 3386: 벤치마크 AUC 0.7908 (중간 난이도)

Campaign 1458(AUC 0.9881)은 너무 쉬운 케이스로, 실제 AdTech 환경의 난이도를 반영하지 못합니다.
현실적인 난이도의 데이터를 사용하여 모델 개선 여지와 실전 능력을 보여줍니다.

해당 레포의 스크립트를 사용하여 업계 표준 포맷으로 데이터를 변환합니다.

## 2단계: 모델 선택 및 학습

**라이브러리**: [DeepCTR-PyTorch](https://github.com/shenweichen/DeepCTR-Torch)
**모델 후보**: DeepFM 또는 PNN

DeepCTR-PyTorch는 검증된 CTR 예측 모델들을 제공하며, 빠른 프로토타이핑에 적합합니다.
DeepFM은 저차원과 고차원 feature interaction을 동시에 학습하는 모델로, AdTech 도메인에서 널리 사용됩니다.

학습 시 train/validation/test 세트로 분리하여 오프라인 AUC를 측정합니다.

## 3단계: 모델 익스포트

**포맷**: ONNX (Open Neural Network Exchange)

PyTorch로 학습된 모델을 ONNX 포맷으로 변환합니다.
ONNX는 프레임워크 독립적이며 최적화된 추론 성능을 제공합니다.

## 4단계: 서빙 API 구현

**프레임워크**: FastAPI

학습된 ONNX 모델을 로드하여 실시간 예측 API를 구현합니다.
입력으로 광고 요청 feature를 받아 CTR 예측값을 반환합니다.

주요 엔드포인트:
- `POST /predict`: 단일 예측
- `POST /predict/batch`: 배치 예측
- `GET /health`: 헬스체크

## 5단계: 검증 및 정합성 확인

테스트 데이터셋을 사용하여 두 가지 AUC를 계산합니다:

1. **오프라인 AUC**: 학습 환경에서 PyTorch 모델로 직접 계산
2. **서빙 AUC**: FastAPI를 통해 ONNX 모델로 예측 후 계산

두 AUC 값의 차이가 허용 범위(예: 0.001 이내) 내에 있는지 확인합니다.
이를 통해 모델 변환 및 서빙 과정에서 정보 손실이 없음을 검증합니다.

## 기대 성과

- CTR 예측 모델 학습 및 평가 능력 검증
- ONNX 모델 변환 및 최적화 경험
- FastAPI 기반 실시간 서빙 시스템 구현
- 오프라인-온라인 정합성 검증 능력

## 다음 단계 (2단계 POC)

1단계 성공 후 확장 가능한 방향:
- 더 복잡한 모델 (xDeepFM, AutoInt 등) 실험
- 멀티 캠페인 데이터로 확장
- Redis 캐싱, 로깅, 모니터링 추가
- Docker 컨테이너화 및 배포 자동화

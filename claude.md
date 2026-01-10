# Claude Context: Yield Optimization POC

## 프로젝트 목표

AdTech 포지션 지원용 POC - **빠른 구현**이 최우선. 학습 목적이므로 프로덕션 엔지니어링은 최소화.

## 핵심 원칙

1. **빠른 프로토타이핑 우선**: 추후 확장성, 추상화, 범용성 고려 X
2. **단순함 유지**: 3줄 반복 코드 > 추상화 레이어
3. **검증된 라이브러리 사용**: DeepCTR-PyTorch 등 바로 쓸 수 있는 것 활용
4. **문서화**: 코드보다 "왜 이렇게 했는지" 설명이 중요

## 주요 설계 결정

### 1. 데이터셋: Kaggle iPinYou 선택
- **왜?**: wnzhang/make-ipinyou-data로 이미 전처리됨 → 시간 절약
- **대안**: 원본 데이터 + 직접 전처리 → 불필요한 시간 소요

### 2. 모델: DeepFM (DeepCTR-PyTorch)
- **왜?**: 검증된 구현체, 빠른 프로토타이핑 가능
- **트레이드오프**: 라이브러리 의존성 증가 but 개발 시간 단축

### 3. 환경: pyenv + venv (루트 레벨)
- **왜?**: pyenv로 Python 버전 고정, venv는 프로젝트 루트에 두어 전역 접근성 확보
- **구조**: training/ 안이 아닌 루트에 venv/ → 프로젝트 전체에서 일관된 환경

### 4. 디렉토리 구조: training/ vs serving/ 분리
- **왜?**: Phase 1(Python)과 Phase 2(Rust) 명확히 분리
- **트레이드오프**: 모노레포 복잡도 증가 but 관심사 분리

### 5. 캠페인 선택: 2259 또는 3386
- **왜?**: 1458은 AUC 0.98로 너무 쉬움 → 현실적인 난이도 필요
- **벤치마크**: 2259 (0.65~0.72), 3386 (0.74~0.84)

## 파일 구조 가이드

```
training/
├── requirements.txt       # 의존성 (torch, deepctr-torch, kaggle)
├── setup_env.sh          # pyenv + venv 설정
├── configs/              # YAML 하이퍼파라미터
├── scripts/              # 실행 스크립트 (download, train, evaluate)
├── src/models/           # 모델 구현 (trainer 클래스)
├── data/                 # 데이터 (gitignore)
└── models/               # 체크포인트 (gitignore)
```

## 작업 시 참고사항

1. **코드 작성**: 최소한의 추상화, 명확한 변수명, 주석 대신 self-documenting code
2. **실험 추적**: 간단한 JSON 로그 or print문으로 충분 (MLflow 등 무거운 툴 X)
3. **에러 핸들링**: 예상 가능한 케이스만 (edge case 과도하게 처리 X)
4. **테스트**: Phase 1에서는 생략 (시간 제약)
5. **문서화**: 왜 이렇게 했는지 간단히 (phase1_results.md에 기록)

## 현재 진행 상황

- [x] 환경 설정 (pyenv, venv, requirements.txt)
- [x] Kaggle 다운로드 스크립트
- [ ] 데이터 로딩 (load_data.py)
- [ ] DeepFM 학습 로직 (deepfm_trainer.py)
- [ ] 학습 스크립트 (train.py)
- [ ] 평가 스크립트 (evaluate.py)

## 참고 문서

- [Phase 1 상세 계획](docs/phase1_plan.md) - 전체 로드맵
- [iPinYou 벤치마크](docs/first-goal.md) - 캠페인별 AUC 목표치

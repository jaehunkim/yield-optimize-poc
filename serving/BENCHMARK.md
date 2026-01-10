# DeepFM Serving Benchmark Guide

이 문서는 DeepFM 추론 서버의 성능 벤치마크를 수행하는 방법을 설명합니다.

## 목차
1. [사전 준비](#사전-준비)
2. [로컬 벤치마크 (스크립트 사용)](#로컬-벤치마크-스크립트-사용)
3. [Docker 벤치마크](#docker-벤치마크)
4. [결과 해석](#결과-해석)

---

## 사전 준비

### 필수 도구 설치

#### wrk (HTTP 벤치마크 도구)
```bash
# Ubuntu/Debian
sudo apt-get install wrk

# macOS
brew install wrk
```

#### nginx (멀티 프로세스 벤치마크용)
```bash
# Ubuntu/Debian
sudo apt-get install nginx

# macOS
brew install nginx
```

#### Docker & Docker Compose (선택사항)
```bash
# Ubuntu
sudo apt-get install docker.io docker-compose

# macOS
brew install docker docker-compose
```

### 모델 파일 준비
```bash
# 모델 파일이 올바른 위치에 있는지 확인
ls -lh models/deepfm_emb8_lr0.0001_dnn25612864_neg150_best.onnx
```

---

## 로컬 벤치마크 (스크립트 사용)

### 1. 단일 프로세스 벤치마크

단일 프로세스의 baseline 성능을 측정합니다.

```bash
cd serving/scripts
chmod +x benchmark.sh
./benchmark.sh single 30s
```

**출력 예시:**
```
[INFO] Building release binary...
[INFO] Starting server on port 3000...
[INFO] Running benchmark (duration: 30s)...
Running 30s test @ http://localhost:3000/predict
  4 threads and 100 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency     X.XXms   X.XXms   X.XXms   XX.XX%
    Req/Sec     X.XXk    X.XXk    X.XXk    XX.XX%
  XXXXXX requests in 30.00s, XX.XXMB read
Requests/sec:  XXXX.XX
Transfer/sec:   X.XXMB

Latency Distribution:
  50%: X ms
  75%: X ms
  90%: X ms
  99%: XX ms
```

### 2. 멀티 프로세스 벤치마크 (4 프로세스)

nginx를 통한 로드 밸런싱으로 4개 프로세스의 성능을 측정합니다.

```bash
./benchmark.sh multi 30s
```

**프로세스 수 변경:**
```bash
NUM_PROCESSES=8 ./benchmark.sh multi 30s
```

---

## Docker 벤치마크

Docker Compose를 사용한 프로덕션 환경 시뮬레이션.

### 1. 컨테이너 빌드 및 실행

```bash
cd serving

# 빌드 및 실행 (4개 프로세스 + nginx)
docker-compose up --build

# 백그라운드 실행
docker-compose up -d --build
```

### 2. 벤치마크 실행

```bash
# 다른 터미널에서 실행
wrk -t16 -c400 -d60s \
    -s scripts/wrk_predict.lua \
    http://localhost:3000/predict
```

### 3. 컨테이너 정리

```bash
docker-compose down
```

### 프로세스 수 조정

[docker-compose.yml](docker-compose.yml)을 수정하여 프로세스 수를 조정할 수 있습니다:

```yaml
services:
  # 프로세스 추가 예시
  deepfm-5:
    build: .
    environment:
      - RUST_LOG=info
      - PORT=3005
    # ...
```

[nginx.conf](nginx.conf)의 upstream 블록도 수정:
```nginx
upstream deepfm_backend {
    least_conn;
    server deepfm-1:3001;
    server deepfm-2:3002;
    server deepfm-3:3003;
    server deepfm-4:3004;
    server deepfm-5:3005;  # 추가
}
```

---

## 결과 해석

### 주요 메트릭

1. **Requests/sec (QPS)**: 초당 처리한 요청 수
   - 단일 프로세스: ~X,XXX QPS 예상
   - 4 프로세스: ~X,XXX QPS 예상 (4배 scale)

2. **Latency (지연시간)**
   - P50 (중간값): 일반적인 응답 시간
   - P99: 최악의 경우 응답 시간 (SLA 결정에 중요)

3. **Throughput**: 초당 전송한 데이터량

### 성능 분석 팁

#### Lock Contention 확인
```bash
# 단일 프로세스에서 CPU 사용률 확인
top -p $(pgrep deepfm-serving)

# CPU 사용률이 100% 미만이면 lock contention 의심
```

#### 멀티 프로세스 효율성
```
Scale Efficiency = (Multi QPS / Single QPS) / 프로세스 수

예시:
- Single: 1,000 QPS
- 4 Processes: 3,800 QPS
- Efficiency = 3,800 / (1,000 × 4) = 0.95 (95%)
```

효율성이 90% 이상이면 훌륭한 scale-out입니다.

#### 병목 구간 찾기

1. **CPU bound**: top으로 CPU 사용률이 100%에 가까움
   - 프로세스 수를 CPU 코어 수만큼 증가

2. **I/O bound**: CPU 사용률이 낮고 대기 시간이 김
   - 디스크/네트워크 최적화 필요

3. **Memory bound**: RSS가 지속적으로 증가
   - 메모리 누수 확인 또는 프로세스당 메모리 제한

---

## 고급 벤치마크

### 다양한 부하 패턴 테스트

#### 1. Spike Test (급격한 부하 증가)
```bash
# 낮은 부하에서 시작
wrk -t4 -c50 -d10s -s scripts/wrk_predict.lua http://localhost:3000/predict

# 갑자기 높은 부하
wrk -t16 -c500 -d10s -s scripts/wrk_predict.lua http://localhost:3000/predict
```

#### 2. Sustained Load Test (지속적 부하)
```bash
# 장시간 테스트 (5분)
wrk -t8 -c200 -d300s -s scripts/wrk_predict.lua http://localhost:3000/predict
```

#### 3. Connection Pool Test
```lua
-- scripts/wrk_keepalive.lua
wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"
wrk.headers["Connection"] = "keep-alive"

-- ... (기존 wrk_predict.lua와 동일)
```

### 프로파일링

#### 1. perf로 CPU 프로파일링
```bash
# 프로세스 PID 확인
ps aux | grep deepfm-serving

# 10초간 프로파일링
sudo perf record -F 99 -p <PID> -g -- sleep 10
sudo perf report
```

#### 2. flamegraph 생성
```bash
# cargo-flamegraph 설치
cargo install flamegraph

# 프로파일링 실행
cargo flamegraph --bin deepfm-serving
```

---

## 최적화 체크리스트

- [ ] Release 빌드 사용 (`cargo build --release`)
- [ ] ONNX Runtime의 `with_optimization_level(Level3)` 적용
- [ ] 프로세스 수 = CPU 코어 수 (또는 코어 수 - 1)
- [ ] nginx `worker_processes auto` 설정
- [ ] nginx keepalive 연결 풀 활성화
- [ ] 충분한 `ulimit -n` (open files) 설정
- [ ] 적절한 커널 파라미터 튜닝 (필요시)

---

## 문제 해결

### 1. "Too many open files" 에러
```bash
# 현재 제한 확인
ulimit -n

# 제한 증가 (임시)
ulimit -n 65536

# 영구 설정 (/etc/security/limits.conf)
* soft nofile 65536
* hard nofile 65536
```

### 2. Port already in use
```bash
# 사용 중인 프로세스 확인
sudo lsof -i :3000

# 강제 종료
pkill -f deepfm-serving
```

### 3. Nginx 설정 오류
```bash
# 설정 검증
nginx -t -c /path/to/nginx.conf

# 로그 확인
tail -f /tmp/nginx_error.log
```

---

## 참고 자료

- [wrk Documentation](https://github.com/wg/wrk)
- [ONNX Runtime Performance Tuning](https://onnxruntime.ai/docs/performance/tune-performance.html)
- [Nginx Performance Tuning](https://www.nginx.com/blog/tuning-nginx/)

# gnss-doppler-lab

실제 GPS 천체력과 지정한 시각·위치를 이용해 **정상 GPS L1 C/A 복소 IQ 파형**을 생성하는 연구용 프로젝트입니다.

현재 단계는 공격 신호나 탐지 AI를 다루지 않습니다. 먼저 재현 가능한 정상 IQ 생성 기반을 확정하는 것이 목적입니다.

> 이 프로젝트는 소프트웨어 IQ 파일만 생성합니다. 실제 RF를 공중으로 송신하지 않습니다.

## 쉽게 보는 동작 구조

```text
설정 파일(YAML)
  ├─ GPS RINEX NAV
  ├─ 정확한 UTC
  ├─ 위도·경도·고도
  ├─ 생성 시간
  └─ RF sample rate
         │
         ▼
설정 검사(rf_config.py)
         │
         ▼
GPS-SDR-SIM 실행(gps_sdr_sim.py)
         │
         ▼
정상 GPS L1 C/A 연속 IQ 파일
         │
         ├─ gps_l1ca_s8_iq.bin
         ├─ gps-sdr-sim.log
         └─ manifest.json
```

- `gps_l1ca_s8_iq.bin`: 여러 가시 GPS 위성 신호가 합쳐진 연속 복소 IQ 파형
- `gps-sdr-sim.log`: 가시 PRN과 simulator 실행 정보
- `manifest.json`: NAV/IQ 해시, UTC, 위치, 실제 sample 수, 실행 명령 등 재현 정보

향후 이 IQ를 GNSS-SDR에 입력하여 PRN별 Doppler, C/N0, correlator, pseudorange 등을 추출합니다.

## 현재 지원 범위

- Constellation: GPS
- Signal: L1 C/A
- Receiver position: 정적 위도·경도·고도
- IQ format: signed 8-bit interleaved I/Q
- 기본 RF sample rate: 2.6 MHz
- NAV input: GPS RINEX 2 NAV
- Output: 연속 IQ + 로그 + 실행 manifest

아직 지원하지 않는 항목:

- UAV trajectory
- Mixed RINEX 3 NAV 직접 입력
- GNSS-SDR 자동 처리
- 스푸핑 IQ 합성
- 탐지 모델 학습

지원하지 않는 입력은 조용히 무시하지 않고 명시적인 오류로 중단합니다.

## 프로젝트 구성

```text
gnss-doppler-lab/
├── configs/
│   └── gps_l1ca_static.example.yaml   # 정상 신호 시나리오 예제
├── src/gnss_doppler_lab/
│   ├── rf_config.py                   # YAML 읽기와 입력 검증
│   ├── gps_sdr_sim.py                 # GPS-SDR-SIM 실행 어댑터
│   ├── rf_pipeline.py                 # IQ·로그·manifest 생성 관리
│   └── cli.py                         # gnss-iq 명령어
├── scripts/
│   ├── generate_iq.py                 # 직접 실행용 진입점
│   └── run_tests.sh                   # 테스트 실행
├── tools/
│   └── build-gps-sdr-sim.sh           # pinned simulator 빌드
├── tests/
│   ├── test_rf_generation.py
│   └── test_rf_cli.py
├── docker/
│   └── Dockerfile                     # simulator까지 포함한 이미지
├── artifacts/                         # 생성 결과, Git에서 제외
├── docker-compose.yml
└── pyproject.toml
```

기존 관측값 수준 Doppler simulator, synthetic constellation, visibility PoC 및 관련 설정·테스트는 현재 RF 방향과 맞지 않아 제거했습니다.

## 빠른 실행

### 1. GPS-SDR-SIM 빌드

```bash
cd /opt/data/gnss-doppler-lab
./tools/build-gps-sdr-sim.sh
```

다음 upstream commit을 고정해 빌드합니다.

```text
osqzss/gps-sdr-sim
28ca29a6719475195e3aabd5930c4ed02d67190f
```

### 2. 도구 확인

package 설치 없이 실행할 때:

```bash
PYTHONPATH=src python -m gnss_doppler_lab.cli probe \
  --executable .tools/gps-sdr-sim-src/gps-sdr-sim
```

정상 결과 예시:

```json
{"available": true, "identity": "osqzss/gps-sdr-sim@28ca29a..."}
```

### 3. 정상 IQ 생성

```bash
PYTHONPATH=src python -m gnss_doppler_lab.cli generate \
  configs/gps_l1ca_static.example.yaml \
  --executable .tools/gps-sdr-sim-src/gps-sdr-sim
```

예제는 pinned upstream 소스에 포함된 실제 GPS RINEX 2 NAV 샘플과 일치하는 2022년 UTC를 사용합니다.

### 4. 생성 결과 확인

```text
artifacts/rf_runs/
└── seoul-normal-smoke-2022_20220101T000000Z/
    ├── gps_l1ca_s8_iq.bin
    ├── gps-sdr-sim.log
    └── manifest.json
```

같은 시나리오와 UTC의 실행 디렉터리가 이미 있으면 기존 결과를 덮어쓰지 않습니다.

## 설정 파일

`configs/gps_l1ca_static.example.yaml`:

```yaml
version: 1
scenario:
  name: seoul-normal-smoke-2022
  constellation: GPS
  signal: L1CA
  utc: "2022-01-01T00:00:00Z"
  duration_seconds: 1
  position:
    type: static
    latitude_deg: 37.5665
    longitude_deg: 126.9780
    altitude_m: 38.0
input:
  rinex_nav: ../.tools/gps-sdr-sim-src/brdc0010.22n
output:
  root: ../artifacts/rf_runs
  rf_sample_rate_hz: 2600000
  sample_format: s8_iq
simulator:
  executable: gps-sdr-sim
```

연구용 실행에서는 `rinex_nav`와 `utc`를 반드시 같은 날짜·유효구간으로 맞춰야 합니다.

## RINEX 입력 주의사항

고정한 GPS-SDR-SIM commit의 parser는 **GPS RINEX 2 NAV 형식**을 사용합니다. 현재 IGS에서 자주 제공하는 `RINEX 3.x MIXED NAV`를 그대로 넣으면 안 됩니다.

프로젝트는 RINEX 3.x를 발견하면 잘못된 IQ를 생성하지 않고 다음과 같이 중단합니다.

```text
the pinned gps-sdr-sim parser requires a GPS RINEX 2 NAV file
```

Mixed RINEX 3 → GPS-only 호환 NAV 변환은 후속 모듈로 추가할 예정입니다.

## 재현성과 안전 처리

파이프라인은 다음을 보장합니다.

- GPS-SDR-SIM commit 고정
- `shell=False`로 안전하게 실행
- RINEX 형식 및 입력 경로 검증
- 짧은 임시 출력명 사용
- 성공 후에만 최종 IQ 이름으로 원자적 변경
- 빈 IQ 파일 거부
- 기존 run 덮어쓰기 금지
- NAV와 IQ SHA-256 기록
- 실제 complex sample 수와 유효 duration 기록
- 전체 simulator stdout/stderr 보존

## Docker

Docker 이미지에는 Python package와 pinned GPS-SDR-SIM이 같이 설치됩니다.

```bash
docker compose build
docker compose run --rm lab gnss-iq probe
docker compose run --rm lab python -m pytest -q
```

생성 결과는 compose 설정에 따라 host의 outputs 디렉터리에 저장할 수 있습니다.

## 테스트

```bash
python -m pytest -q
```

테스트는 다음을 검증합니다.

- config schema와 UTC/좌표 검증
- GPS L1 C/A 이외 입력 거부
- RINEX 3 입력 거부
- 실제 GPS-SDR-SIM CLI 옵션 계약
- 안전한 subprocess 실행
- 긴 출력 경로 문제 방지
- atomic output
- 실패·빈 파일 처리
- manifest 및 해시
- 실제 sample 수와 유효 duration 계산
- CLI 반환 코드

## 다음 개발 순서

```text
1. 정상 정적 IQ 생성                ← 현재 완료
2. Mixed RINEX 3 전처리
3. UAV trajectory 기반 동적 IQ
4. IQ → GNSS-SDR 자동 처리
5. PRN별 tracking/observable export
6. 정상 데이터셋 생성
7. 스푸핑 IQ 합성
8. 탐지 알고리즘 학습·평가
```

탐지 모델은 정상 IQ → GNSS-SDR 처리 결과가 충분히 검증된 후 추가합니다.

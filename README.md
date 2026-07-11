# GNSS Doppler Lab

GPS L1 C/A 정상·스푸핑 RF/IQ를 생성하고 GNSS-SDR 내부값과 Doppler 기반 탐지 방법을 연구하는 프로젝트입니다.

## 현재 연구 방향

```text
Notebook 실험 설정
→ 실제 RINEX NAV/SP3·수신기 위치/궤적
→ GPS-SDR-SIM IQ + simulator truth
→ GNSS-SDR acquisition/tracking/observables
→ 정상/스푸핑 비교
→ 통계·ML·DL 탐지
→ TEXBAT/OAKBAT/FGI 외부 일반화
```

현재 구현 완료 범위는 **정적 위치 GPS L1 C/A 정상 IQ 생성과 기초 IQ 시각화**입니다. 이동 trajectory, GNSS-SDR 자동 처리, 스푸핑 생성과 탐지 모델은 후속 단계입니다.

## 시작점: 통합 Jupyter Notebook

VS Code에서 다음 파일 하나를 엽니다.

```text
notebooks/gnss_spoofing_research_workflow.ipynb
```

Kernel은 다음을 선택합니다.

```text
/home/ubuntu/projects/gnss-doppler-lab/.venv/bin/python
```

Notebook 맨 앞 설정 셀에서 실험 조건을 바꿉니다.

```python
RUN_GENERATION = False
SCENARIO_NAME = "seoul-normal-static-2022"
SCENARIO_UTC = "2022-01-01T00:00:00Z"
DURATION_SECONDS = 1
RF_SAMPLE_RATE_HZ = 2_600_000

POSITION_MODE = "static"
LATITUDE_DEG = 37.5665
LONGITUDE_DEG = 126.9780
ALTITUDE_M = 38.0
TRAJECTORY_FILE = None

RINEX_NAV_PATH = PROJECT_ROOT / ".tools" / "gps-sdr-sim-src" / "brdc0010.22n"
```

- 기존 최신 IQ 분석: `RUN_GENERATION = False`
- 새 설정으로 IQ 생성: `RUN_GENERATION = True`
- 현재 위치 변경: 위도·경도·고도 변경
- 향후 이동체: `POSITION_MODE="trajectory"`와 `TRAJECTORY_FILE`을 사용할 예정이며 현재는 명시적으로 중단됩니다.

Notebook은 설정으로 임시 YAML을 만들고 `scripts/generate_iq.py`를 호출합니다. 임시 설정 파일은 실행 후 삭제되며 실험 조건은 최종 `manifest.json`에 보존됩니다.

## 주요 구조

```text
gnss-doppler-lab/
├── notebooks/
│   └── gnss_spoofing_research_workflow.ipynb
├── scripts/
│   ├── generate_iq.py
│   └── run_tests.sh
├── src/gnss_doppler_lab/
│   ├── cli.py
│   ├── gps_sdr_sim.py
│   ├── iq_visualization.py
│   ├── research_sequence.py
│   ├── rf_config.py
│   └── rf_pipeline.py
├── tools/
│   └── build-gps-sdr-sim.sh
├── tests/
├── artifacts/
└── pyproject.toml
```

## IQ 산출물

```text
artifacts/rf_runs/<run-id>/
├── gps_l1ca_s8_iq.bin
├── gps-sdr-sim.log
├── iq_dashboard.png
└── manifest.json
```

- IQ 형식: signed 8-bit interleaved complex I/Q
- 기본 sample rate: 2.6 MHz
- `manifest.json`: UTC, 위치, NAV/IQ hash, sample 수, 실행 명령과 simulator 정보

## 환경 준비

Ubuntu VM 프로젝트에서:

```bash
cd /home/ubuntu/projects/gnss-doppler-lab
source .venv/bin/activate
pip install -e '.[dev]'
./tools/build-gps-sdr-sim.sh
pytest -q
```

Jupyter Kernel이 보이지 않을 때:

```bash
python -m ipykernel install --user \
  --name gnss-doppler-lab \
  --display-name "GNSS Doppler Lab"
```

## 직접 CLI 실행

Notebook 이외에서 실행할 때는 명시적인 YAML 경로가 필요합니다.

```bash
python scripts/generate_iq.py generate /path/to/config.yaml \
  --executable .tools/gps-sdr-sim-src/gps-sdr-sim
```

기본 설정 파일은 따로 유지하지 않습니다. 연구 실험 설정은 통합 Notebook 맨 앞에서 관리합니다.

## 테스트

```bash
pytest -q
```

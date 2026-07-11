# GNSS Doppler Lab

## GPS-SDR-SIM `-t` timescale

Although upstream documents `-t` as a date/time calendar, the pinned GPS-SDR-SIM
passes those fields directly to `date2gps`; they are therefore **GPST-like**, not
UTC. `scenario.utc` remains actual UTC. Generation strictly reads the fixed-column
`LEAP SECONDS` record from the selected RINEX NAV header and passes
`scenario.utc + GPS_MINUS_UTC` to `-t` (including date rollover). Missing,
duplicate, malformed, non-ASCII, or unterminated headers fail closed; no leap
second value is hardcoded.

Manifest schema 2 records the requested UTC, GPST simulator calendar, GPS-UTC
offset, GPS week/TOW, and the source NAV path/hash/header record. Static and
trajectory generation share this conversion. Dynamic validation directly aligns
PVT UTC to requested UTC for corrected manifests (zero correction). Old manifests
without timescale metadata are rejected as ambiguous unless the user explicitly
passes `--legacy-gps-utc-offset-seconds N`; that override is recorded in the
validation summary and is never inferred automatically.

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

현재 구현 완료 범위는 **정적 위치 및 최대 300초의 10 Hz LLH/ECEF trajectory를 사용하는 GPS L1 C/A 정상 IQ 생성, IQ 시각화, GNSS-SDR acquisition/tracking 및 Doppler·C/N₀ CSV 추출**입니다. navigation/PVT용 장시간 run, 스푸핑 생성과 탐지 모델은 후속 단계입니다. trajectory 지원은 라이브러리/CLI에 한하며, 통합 Notebook은 안전 기본값으로 static-only guard를 유지합니다.

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
RUN_RECEIVER = False
GNSS_SDR_PATH = Path("/usr/bin/gnss-sdr")
GNSS_SDR_CHANNEL_COUNT = 11
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
- 최신 IQ를 GNSS-SDR로 처리: `RUN_RECEIVER = True`
- 현재 위치 변경: 위도·경도·고도 변경
- Notebook 위치 모드는 안전 기본값인 static-only guard로 제한됩니다. 10 Hz LLH/ECEF trajectory(최대 300초)는 아래의 라이브러리/CLI 사용법을 따르십시오.

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

GNSS-SDR 처리 결과:

```text
artifacts/receiver_runs/<run-id>/
├── receiver.conf
├── receiver.log
├── tracking.csv
├── tracking_summary.csv
├── raw/
│   ├── epl_tracking_ch_*.mat
│   ├── epl_tracking_ch_*.dat
│   └── observables.mat
└── manifest.json
```

`tracking.csv`에는 시간, PRN, carrier Doppler, Doppler rate, C/N₀, Prompt I/Q, carrier/code loop error가 저장됩니다. `manifest.json`은 GNSS-SDR 버전·실행파일/config/IQ hash와 acquisition된 PRN을 보존합니다.

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

## Dynamic trajectories (10 Hz)

Generate headerless WGS-84 LLH truth for Seoul (all distances in m, speeds in m/s, angles in deg):

```bash
python -m gnss_doppler_lab.trajectory straight trajectories/S1.csv --latitude-deg 37.5665 --longitude-deg 126.9780 --altitude-m 50 --duration-seconds 60 --speed-mps 5
python -m gnss_doppler_lab.trajectory circle trajectories/S2.csv --latitude-deg 37.5665 --longitude-deg 126.9780 --altitude-m 50 --duration-seconds 60 --speed-mps 5 --radius-m 30
# Exactly two closed laps in 60 s; --laps determines the effective speed.
python -m gnss_doppler_lab.trajectory circle trajectories/S2-closed.csv --latitude-deg 37.5665 --longitude-deg 126.9780 --altitude-m 50 --duration-seconds 60 --speed-mps 5 --radius-m 30 --laps 2
python -m gnss_doppler_lab.trajectory parallel-sweep trajectories/S3.csv --latitude-deg 37.5665 --longitude-deg 126.9780 --altitude-m 50 --duration-seconds 60 --speed-mps 5 --leg-length-m 100 --lane-spacing-m 20
```

The time contract is exact: duration `D` produces exactly `D * 10` headerless rows at timestamps `0.0, 0.1, ..., D - 0.1`; extra rows are rejected rather than silently truncated. A normal `circle` is constant-radius circular motion and may cover only an arc. Its sidecar reports effective arc, lap count, closure error, and `closed_orbit` without claiming closure. Use a positive integer `--laps` for an exactly closed conceptual orbit (the endpoint at `D` is excluded by the sampling contract); this overrides effective speed while preserving the existing required `--speed-mps` CLI argument.

Each command atomically writes the truth CSV and JSON sidecar. The sidecar includes the CSV SHA-256, generator schema/version, WGS-84 constants/frame, actual row/time bounds, effective distance/speed/laps/closure, parameters, and literature provenance. Pair failure policy removes the old sidecar before CSV publication, so failure can leave an unmistakable CSV-without-sidecar but never a stale mismatched pair. At RF config load, a present standard sidecar is schema-checked and its `csv_sha256` must match the exact CSV bytes; therefore project-generated trajectories retain their integrity binding. External trajectory CSVs without a sidecar remain supported, but are still hashed during validation and the validated snapshot hash is mandatory in the run manifest. The runner rejects source mutation after validation and verifies the staged copy before simulator execution. Configure `scenario.position` as `{type: trajectory, path: ../trajectories/S1.csv, coordinate_system: llh}` (`ecef` is also accepted). Paths resolve relative to the YAML. Library/CLI trajectory runs are strictly validated at 10 Hz and limited to 300 s. Existing static configs remain supported; the integrated notebook intentionally retains its static-only guard as a safe default and does not accept trajectory mode.

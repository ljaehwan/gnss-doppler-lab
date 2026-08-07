# ACAF-NF Stage-1 R1 (continuous tracker)

## 목표

- 기존 Stage-1 정적 fail-closed 결과를 넘어, 1 ms tracker cadence와 상태 바인딩의 연속성을 검증한다.
- cleanStatic / DS3 / DS4 / DS7 / DS8의 기존 MAT/DAT 구조를 감사한다.
- checkpoint 방식으로 산출물(감사 JSON/CSV) 기반으로 다음 단계 진행을 게이팅한다.

## Checkpoint A

- 입력: `configs/acaf_nf_stage1_source_binding.json`
- 출력:
  - `artifacts/acaf_nf_stage1_r1_continuous_tracker/tracker_cadence_audit.json`
  - `artifacts/acaf_nf_stage1_r1_continuous_tracker/tracker_cadence_by_channel.csv`
- 실행:
  - `python3 scripts/run_acaf_nf_stage1_r1.py --checkpoint A`

## 현재 구현 상태

- Checkpoint A 빌더/감사 모듈 도입 완료:
  - `src/gnss_doppler_lab/acaf_nf_stage1_continuous_tracker.py`
  - `scripts/build_acaf_nf_continuous_tracker.py`
  - `scripts/run_acaf_nf_stage1_r1.py`
  - `scripts/verify_acaf_nf_stage1_r1.py`

## 다음 단계

- Checkpoint B: cleanStatic 연속 tracker 검증
- Checkpoint C: DS3/DS4/DS7/DS8 tracker 재생성 및 source-binding
- Checkpoint D: 실제 Stage-1 성능 실험 및 독립 검증

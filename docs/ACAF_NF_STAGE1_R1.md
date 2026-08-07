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

## Checkpoint B

- exact 25,000-sample half-open support만 출력한다. 24,999/25,001 cadence는 감사 증거로 보존하지만 연속 tracker support로 사용하지 않는다.
- NCO/code/carrier/aux는 previous MAT/DAT row, Prompt는 current row에 결속한다.
- C/N0와 carrier lock은 previous/current/next same-PRN triple 전체에서 gate한다.
- GNSS-SDR의 148-byte DAT record에서 offset 80의 little-endian uint64 sample stamp를 직접 읽어 MAT 전체 row와 대조한다.
- 서로 다른 receiver channel은 동일한 전역 IQ 시간을 의도적으로 공유한다. interval uniqueness는 channel/PRN 내부에서 검증하고 L20은 channel/PRN을 넘지 않는다.
- cleanStatic 969 epoch CAF, exact L20 aggregation, R1.4 공통 epoch 수치/complex-surface SHA 재현을 producer와 독립 verifier가 각각 검증한다.

## 현재 구현 상태

- Checkpoint A 빌더/감사 모듈 도입 완료:
  - `src/gnss_doppler_lab/acaf_nf_stage1_continuous_tracker.py`
  - `scripts/build_acaf_nf_continuous_tracker.py`
  - `scripts/run_acaf_nf_stage1_r1.py`
  - `scripts/verify_acaf_nf_stage1_r1.py`

Checkpoint B가 `CONTINUOUS_TRACKER_VALID`일 때만 C로 진행한다. DS4의 manifest/raw alignment는 별도 근거 없이 보정하지 않고 fail-closed로 유지한다.

## Checkpoint C

- DS3/DS7/DS8은 raw SHA가 receiver manifest에 직접 결속된 경우에만 primary tracker로 VALID 판정한다.
- DS4는 약 128.22초까지만 transition-only coverage를 기록한다. 기존 receiver manifest에 raw SHA와 build SHA가 없어 `INVALID_RECORD_ALIGNMENT`이며 공격 점수 입력으로 사용할 수 없다.
- phase 경계와 L20은 각 scenario에서 독립 계산하며 서로 다른 channel/PRN을 연결하지 않는다.
- stock GNSS-SDR dump는 1 ms correlation 설정에서도 navigation-bit 경계에서만 기록한다. 복구 replay는 tracking loop 수식과 telemetry gate를 유지하고 valid 1 ms loop마다 dump만 추가한 별도 hash-bound binary를 사용한다.

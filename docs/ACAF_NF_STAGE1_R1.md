# ACAF-NF Stage-1 R1 continuous tracker evaluation

Checkpoint D consumes independently verified Checkpoint B/C artifacts. It samples exact same-PRN L=20 windows by receiver-second and recomputes every support from raw complex IQ with the frozen dense delay-Doppler CAF.

Only chronological cleanStatic roles may fit T0, diagonal variance, H1 templates, pooling, calibration, and thresholds. Attack pre-onset windows are external false-positive evaluation data only. The bounded R1 run reports actual numeric metrics and controls but does not promote B0 beyond `PROVISIONAL_UNAVAILABLE`.

## Frozen result

- Checkpoint B independent verifier: `PASS` on the authenticated original clean tracker.
- Checkpoint C independent verifier: `PASS`; all DS3/DS4/DS7/DS8 bindings and DAT stamps pass after raw-bound receiver replays.
- Checkpoint D independent verifier: `PASS` with scientific verdict `PHYSICS_FEASIBILITY_NO_GO`.
- Artifact pytest: `44 passed`.
- D verification report SHA256: `96821d45ae5206aaf6027ce2aae8b298d8b3f82128df50c91364a9175aa677bf`.
- D checksums SHA256: `803b9d8840b2051ec43a91aeaddf208eacd387d86712f01c87607cdf8c951bea`.

The full-duration cleanStatic exporter replay is preserved as NO-GO evidence and is not used for fitting: it reproduced Prompt and delay but failed the frozen L20 Doppler and R1.4-common gates. The validated clean source provides only three chronological receiver-second bins per role, so calibration uncertainty remains a material limitation.

```bash
PYTHONPATH=src pytest tests/test_acaf_nf_stage1_continuous_tracker.py tests/test_acaf_nf_stage1_static_feasibility.py tests/test_acaf_nf_stage1_r1.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python3 scripts/run_acaf_nf_stage1_r1.py --checkpoint D --source-binding configs/acaf_nf_stage1_r1_source_binding.json --output artifacts/acaf_nf_stage1_r1_continuous_tracker
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python3 scripts/verify_acaf_nf_stage1_r1.py artifacts/acaf_nf_stage1_r1_continuous_tracker --checkpoint D --write-report
```

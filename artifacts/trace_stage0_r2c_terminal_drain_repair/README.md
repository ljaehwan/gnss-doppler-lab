# TRACE Stage-0 R2c Terminal Drain Repair

R2c diagnoses the R2b one-row mismatch as an immediate finite-source control-plane stop racing the downstream GNU Radio channel drain. The opt-in repair propagates natural EOS and waits for buffered work before normal shutdown; no TRACE score math, threshold, tolerance, window, quality gate, block-key gate, or alarm gate changed.

Phase A status: `PASS`. Phase B authorized: `True`. Final verdict: `INCONCLUSIVE_INPUT_OR_RECEIVER`. Attack/normal-FPR/control metrics available: `False`.

R1/R2/R2a/R2b artifacts and fail-closed verdicts remain preserved. Large receiver builds and native dumps remain outside Git and are bound here by manifests and SHA-256. Hermes independent verification remains required.

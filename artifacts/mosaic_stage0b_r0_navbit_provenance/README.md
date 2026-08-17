# MOSAIC Stage-0B R0 navigation-bit provenance

Verdict: **STAGE0B_NAVBIT_PROVENANCE_PASS**

The receiver run did not emit decoded telemetry or observables (`TelemetryDecoder_1C.dump=false`, `Observables.dump=false`). The primary source is therefore direct decoding of the authenticated native 1 ms complex Prompt. Exact 20 ms timing comes only from the receiver `data_symbol_boundary` field; GPS preamble, every-word parity, HOW TOW, and subframe-ID continuity independently validate the recovered sequence.

No attack input, synthetic injection, detector, neural model, threshold tuning, outcome-selected boundary/polarity, or constant-+1 fallback was used. Stage-0A remains unchanged: its per-epoch complex-amplitude normalization is NAV-sign invariant, while multi-bit Stage-0B requires this transition sequence.

## Valid coverage

- OAKBAT.cleanStatic: PRNs [10, 11, 21, 24, 27]; 600 validated bits = 20 words = 2 consecutive subframes per PRN; 2 preambles/PRN; zero parity or TOW-continuity failures.
- TEXBAT.cleanStatic: PRNs [3, 13, 16, 19, 30]; 600 validated bits = 20 words = 2 consecutive subframes per PRN; 2 preambles/PRN; zero parity or TOW-continuity failures.

Every listed mapping has zero transcription error: starts and exclusive ends are copied from TRACE raw intervals, not rounded seconds. Receiver code-NCO integer rounding produces observed +/-1-sample joins and 4999-5001 or 24999-25001-sample epochs; these are preserved rather than normalized. Where validated bits precede receiver bit-sync acquisition, the fixed modulo-20 phase is extended backward only over the uninterrupted 1-ms TRACE sequence, and the distance is recorded per PRN. The usable Stage-0B interval for each PRN is exactly the two-subframe range in `coverage_summary.json` and `per_prn_validation.csv`.

Bits before/after those complete subframes remain directly recovered candidates but are excluded from validated injection coverage; their ranges are in `rejected_intervals.csv`. Global physical carrier sign has the normal BPSK 180-degree representative ambiguity, so the stored ±1 uses the receiver Prompt real-axis convention fixed before structural validation; transitions and decoded NAV data are invariant.

## Reproduce

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 /home/ubuntu/projects/gnss-doppler-lab/.venv/bin/python scripts/verify_mosaic_stage0b_r0.py
```

Next action: freeze these checksummed sidecars and use only the listed valid intervals in the next, separately authorized Stage-0B receiver-in-the-loop injection task.

# WCL CGC fresh-clone reproduction v1

## Outcome

**EXACT_MATCH.** The primary WCL CGC analysis was rerun from a separate Git
clone and independent Python virtual environment. The regenerated logical
result and all three final analysis CSV files exactly match the sealed result.

## Execution identity

- execution date: 2026-09-03;
- fresh clone:
  `/home/ubuntu/hdd_data/reproductions/gnss-doppler-lab-wcl-cgc-v1`;
- tested commit: `d93dc04b309e90ce14a0930987452bdfd4d19765`;
- Python: 3.12.3;
- environment: packages pinned by `requirements/wcl-cgc-v1.txt` in a new
  `.venv`; `pip check` reported no broken requirements;
- retained input: `/home/ubuntu/hdd_data/cgc_temporal_final_static_v1`;
- sealed input summary SHA-256:
  `b154e411f4447fb534c0e10dee45f2f14f84b0e4e4eeedaa771cfefaf721045d`.

The clone reused only the ignored, hash-pinned simulator/GNSS-SDR builds and
large input artifacts through symbolic links. Source, environment, generated
analysis outputs, and Git state were separate from the development worktree.

## Exact comparison

All of the following checks passed:

- retained pair records, LOS logs, and 20 receiver manifests passed their
  recorded SHA-256 checks;
- `decision`, `aggregates`, all release `gates`, and all five per-pair result
  objects were JSON-equal to the sealed full summary;
- regenerated `delay_estimates.csv` SHA-256:
  `d7bf248fddd04394c832881b6da5dfccd37ee6471efb91626c7cd5f10479a2c5`;
- regenerated `stabilized_delay_estimates.csv` SHA-256:
  `e2b569d0d85434f081e5df8136e6597d010b1c22b0648d625361ad978ebc165a`;
- regenerated `geometry_scores.csv` SHA-256:
  `1b3056f2dde2fa19a76b208ff9257ef5c5916cf162186b739d9e22d167359f93`.

The regenerated primary metrics were:

| Metric | Reproduced value |
|---|---:|
| Carrier-coupled persistent detections | 5/5 |
| Doppler-locked persistent detections | 5/5 |
| Normal pairs with persistent alarm | 0/5 |
| Multipath pairs with persistent alarm | 1/5 |
| Doppler-locked versus multipath AUC | 0.9800854700854701 |
| Median Doppler-locked latency | 8.0 s |

The current focused WCL core and reproduction suite passed `72` tests in the
fresh clone. The formula audit was also rerun there: its JSON SHA-256 was
`fa2969f118c468d28c9e3dccc37d3ef062ce40628263e16481ee3009c69f8c0a`,
and the tracked JSON, PNG, and PDF had no Git difference after regeneration.

## Manual command

```bash
cd /home/ubuntu/hdd_data/reproductions/gnss-doppler-lab-wcl-cgc-v1
.venv/bin/python scripts/reproduce_wcl_cgc_final_analysis_v1.py \
  --data-root /home/ubuntu/hdd_data/cgc_temporal_final_static_v1 \
  --output-dir reproduction-output/wcl-cgc-final-analysis-v1
```

The local machine-readable report is
`reproduction-output/wcl-cgc-final-analysis-v1/reproduction_report.json`.
Environment setup and the focused test command are in
`research/wcl-cgc/reproduction/README.md`.

## Boundary

This is an exact replay from retained GNSS-SDR 9-tap tracking outputs. It
repeats signed-delay extraction, five-bin causal stabilization, observability
gating, Partial-F scoring, persistence, and the final metrics. It does not
regenerate the removed 25 MHz intermediate IQ files. A full simulator-to-RF
rebuild is a new versioned experiment and must use a new output root rather
than overwrite the sealed v1 endpoint.

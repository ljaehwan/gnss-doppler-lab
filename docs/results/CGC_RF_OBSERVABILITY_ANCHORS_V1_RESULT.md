# CGC RF observability anchors v1 result

## Decision

`NOT REPRODUCED`. The profile-level `40/60/80 m` observability ordering did
not pass its preregistered receiver-RF gates. The six anchors, AUC threshold,
comparison interval, and decision rules were committed in `ccc39db` before
outcome access. No anchor substitution, threshold change, post-release tuning,
or rerun was performed.

## Primary result

All six 30-second, 25 MHz conditions completed GNSS-SDR tracking and supplied
14 eligible spoof bins and the same 14 eligible multipath bins. The minimum
per-bin satellite support was 10 PRNs, above the frozen minimum of eight.

| Power | Separation | Profile-screen AUC | Receiver-RF AUC | Frozen role passed? |
|---:|---:|---:|---:|---|
| -6 dB | 40 m | 0.7027 | 0.2296 | yes: below 0.80 |
| -6 dB | 60 m | 0.8741 | 0.3827 | no: below 0.80 |
| -6 dB | 80 m | 0.9569 | 0.6582 | no: below 0.80 |
| +3 dB | 40 m | 0.7210 | 0.5153 | yes: below 0.80 |
| +3 dB | 60 m | 0.8874 | 0.6378 | no: below 0.80 |
| +3 dB | 80 m | 0.9696 | 0.5714 | no: below 0.80 |

At `-6 dB`, AUC increased strictly from `0.2296` to `0.3827` to `0.6582`,
so the predicted direction with spatial separation survived, but neither the
60 m nor 80 m anchor crossed 0.80. At `+3 dB`, AUC increased from `0.5153` to
`0.6378` and then fell to `0.5714`; therefore both the 60/80 m threshold gates
and strict ordering failed.

| Frozen gate | -6 dB | +3 dB |
|---|---|---|
| At least 8 bins in both streams | pass | pass |
| 40 m AUC below 0.80 | pass | pass |
| 60 m AUC at least 0.80 | fail | fail |
| 80 m AUC at least 0.80 | fail | fail |
| Strict 40-to-60-to-80 m increase | pass | fail |

## Physical interpretation

The ideal profile-level separation cannot be treated as a calibrated
receiver-RF observability boundary. The front end, acquisition/tracking loop,
prompt ownership, finite nine-tap sampling, and secondary-delay estimator form
a power- and state-dependent transfer function between physical code
separation and the recovered delay geometry. The current RF result directly
shows that this transfer is not captured by spatial separation alone.

The monotone `-6 dB` result is still useful mechanism evidence: greater
physical separation made the common spoof geometry more distinguishable from
independent multipath. It does not locate the crossing because the largest
tested point remained below 0.80. The nonmonotone `+3 dB` result shows that a
stronger counterfeit component cannot be assumed to improve observability
monotonically. Prompt takeover or tracking-state ambiguity is a plausible
explanation, but this pilot did not instrument those states well enough to
identify the cause.

Consequently, the earlier `60 m = 0.2047 chip` profile boundary must not be
claimed as a receiver-RF boundary. The defensible result is narrower: CGC
observability is distorted by receiver tracking dynamics, and its RF boundary
remains unresolved beyond the tested range and across geometries.

## Integrity and retention

Every spoof waveform had a byte-identical pre-onset prefix to the pinned normal
RF. All comparison streams contained 14 bins, and their minimum satellite
support was 10 or 11 PRNs. The common authentic and multipath inputs were not
deleted.

After each receiver run and score write completed, the two campaign-created
intermediate IQ files were hash-checked and removed as preregistered: six
counterfeit component files and six composed spoof files, 12 total. Their
hashes, byte counts, trajectories, RF manifests, receiver outputs, score CSVs,
and deterministic regeneration inputs remain. The retained result tree is
483,476,650 bytes.

The immutable raw result is
`artifacts/cgc_rf_observability_anchors_v1/summary.json`, SHA-256
`b9ca6b22ee65a90adb0e387aaba40ec3ece0a2d583b3fed61e3e0f1d32378c99`.

## Claim boundary and next experiment

This is a single previously used static satellite geometry with a shared
simulated multipath control. It is not held-out geometry validation, a field
boundary, real-multipath accuracy, a TEXBAT displacement calibration, or an
operational alarm result.

The next mechanism experiment should measure the receiver transfer function
itself: sweep separation and power with multiple LOS geometries while logging
prompt ownership, recovered secondary-delay magnitude/sign, and lock state.
Only after locating a receiver-level crossing should a new boundary be frozen
and evaluated on held-out geometries.

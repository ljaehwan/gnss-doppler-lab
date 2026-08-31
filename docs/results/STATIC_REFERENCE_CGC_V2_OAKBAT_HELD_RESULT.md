# Static reference-CGC v2 OAKBAT held terminal result

## Decision

**`STATIC_REFERENCE_CGC_HELD_NOT_SUPPORTED`**.  The v2 held release was
committed and pushed at
`63ca8cf2a9ff6a2d34d0f9d9f337aa243745f968` before OS5 or OS6 was processed.
The complete HDD result is
`/home/ubuntu/hdd_data/oakbat/analysis/static_reference_cgc_v2_oakbat_held/summary.json`
(SHA-256 `32f7ed6ea5540aa928d5d71e94f429c7ecbd1c745e52395243a1095165d5ffdd`).

Both held recordings failed the pre-onset reference-eligibility gate:

| Held recording | Baseline bins | PRNs | Baseline displacement q99 | Clean-only limit | Post score read? |
|---|---:|---:|---:|---:|:---:|
| OAKBAT OS5, time push | 68 | 11 | 1,899,539 m | 75.682 m | no |
| OAKBAT OS6, ECEF-Z position push | 37 | 10 | 11,579,261 m | 75.682 m | no |

The runner therefore abstained on both recordings before reading any
post-onset pseudorange bin.  Only `os5_baseline_scores.csv` and
`os6_baseline_scores.csv` were written; no held post-score CSV exists.
Abstention was preregistered as failure, not counted as a correct decision.

## Interpretation

The v1 OS3 failure was not an isolated post-attack anomaly.  With the current
GNSS-SDR `Hybrid_Observables` output, multiple OAKBAT recordings contain
PRN-dependent code-time/ambiguity groups during the nominally clean baseline.
A common clock nuisance cannot remove those groups, and a per-PRN linear trend
does not turn them into a trustworthy absolute displacement reference.

This terminal result rules out the proposed shortcut:

> Do not claim that the existing receiver pseudoranges provide a reliable
> pre-attack static reference that repairs prompt-local nine-tap observability.

It does **not** invalidate the original prompt-local CGC mechanism.  V1 still
showed zero held-clean alarms, zero OS2 alarms, zero alarms on both static
matched multipath controls, and a 43-s OS4 position-push detection.  Those are
development observations only because the reference method failed its full
protocol and misses the opened 70-m FGI change under the frozen 89.1-m
threshold.

## Paper decision

The WCL draft must not add static reference-CGC as a validated contribution.
The paper should remain about prompt-local complex multi-tap signed delays and
cross-satellite LOS consistency, with the already documented scope:

- it classifies an observable overlapping two-component pull-off regime;
- finite tap aperture creates an onset and saturation boundary;
- after the tracking loop recenters, prompt-local taps may lose absolute
  displacement observability, as shown by FGI; and
- absolute pseudorange anchoring is future work unless the receiver exports a
  verified ambiguity-consistent code-phase observable or a fully corrected
  pseudorange solution.

No v3 threshold tuning on OAKBAT is justified.  A future absolute-reference
experiment would require a new data contract: one continuous static stream
with sufficient navigation warm-up, pre-onset clean reference, controlled
multipath and carry-off onset, and receiver output that preserves consistent
unwrapped code phase.  That is a separate experiment, not a repair to report
inside the current WCL results.

## Retained data

The OS5/OS6 receiver outputs were stored on the HDD under
`/home/ubuntu/hdd_data/oakbat/analysis/static_reference_cgc_v2_oakbat_held/`
and use about 0.5 GB.  The original 9.6-GB-per-recording IQ files were read
only and their hashes remained unchanged.  No SSD cleanup or unrelated user
file modification was performed.

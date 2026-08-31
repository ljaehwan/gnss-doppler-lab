# FGI-SpoofRepo TGD prompt-local observability audit v1

Date: 2026-08-31

## Result

The post-hoc exploratory mechanistic signature is
`OBSERVABILITY_LOSS_SUPPORTED`. This does not change the frozen detector
decision, which remains `NOT_SUPPORTED`. It identifies why the detector missed
the stable captured interval.

| Stable-post quantity, `[160,230)` s | Median result |
|---|---:|
| Matched one-second bins | 70 |
| NMEA PVT displacement | 69.91 m |
| Pseudorange-fitted displacement | 71.37 m |
| Pseudorange-to-NMEA direction cosine | 0.9993 |
| Pseudorange-to-NMEA vector error | 3.25 m |
| Prompt-local nine-tap fitted displacement | 57.05 m |
| Nine-tap-to-NMEA direction cosine | 0.0052 |
| Nine-tap-to-NMEA vector error | 93.91 m |
| Nine-tap/pseudorange vector-error ratio | 28.89 |
| Frozen persistent spoof alarms | 0 of 70 |

All eight configured exploratory signatures passed. The pseudorange vector
pointed in the same three-dimensional ECEF direction as the receiver PVT in
all 70 stable-post bins at a cosine threshold of 0.8. Only 9 of 70 local-tap
vectors met that threshold, and their median direction cosine was nearly zero.
The local fit also retained a high median clock-centered residual (`0.689`),
whereas the differential pseudorange fit fell to `0.0246`.

## Easy interpretation

The spoofing signal really moved the receiver solution. When the receiver's
absolute satellite code phases are retained, the delays across PRNs explain
the same approximately 70 m false displacement as the NMEA position.

The prompt-local nine taps answer a different question: what does the
correlation shape look like in a narrow window around the peak that the
tracking loop currently follows? After takeover, the loop places Prompt on the
spoof peak. The old absolute offset is therefore absorbed into the tracking
state. A local peak may still appear well centered, and its per-satellite
signed shape offsets no longer form the stable 3-D displacement that moved the
PVT.

In short, the absolute code phase remembers the pull-off; the recentered local
shape usually does not.

## Baseline and sensitivity

GNSS-SDR changes its common pseudorange clock offset inside bins 40--42. Bin 43
is the first complete settled bin, so the primary clean PRN trends use
`[43,120)` s. Changing the clean start while preserving the 120 s endpoint
gave:

| Clean start | Post median direction cosine | Post median vector error |
|---:|---:|---:|
| 43 s | 0.9993 | 3.25 m |
| 50 s | 0.9976 | 5.85 m |
| 60 s | 0.9833 | 13.21 m |
| 70 s | 0.9557 | 20.98 m |
| 80 s | 0.7093 | 67.62 m |

The direction conclusion is strong for the 43--70 s starts. The short
80--120 s baseline extrapolates unmodelled PRN curvature poorly and is reported
as a limitation rather than omitted.

## Claim boundary and next method

NMEA PVT and pseudorange come from the same GNSS-SDR receiver, so their
agreement is not independent validation. The receive-time orbit subtraction
also omits precise transmit-time, Sagnac, satellite-clock, atmospheric, and
relativistic corrections; the clean-only PRN trends make this a differential
mechanism audit, not a precise navigation solution.

The result supports a narrower, defensible statement: prompt-local CGC can be
informative while authentic and spoof peaks remain locally resolvable, but it
is not invariant after full tracking recentering. The next detector must add
an invariant history reference--for example pre-capture code-phase memory or
multi-epoch code-Doppler consistency--and validate that extension on unopened
data. It must not be tuned and retested as a new detector on this opened TGD
recording.

## Frozen artifacts

- Config: `configs/experiments/fgi_spoofrepo_tgd_observability_audit_v1.json`
- Protocol: `docs/results/fgi_spoofrepo_tgd_observability_audit_protocol_v1.md`
- Runner: `scripts/audit_fgi_spoofrepo_tgd_observability.py`
- Unit tests: `tests/test_fgi_spoofrepo_tgd_observability.py`
- HDD result directory:
  `/home/ubuntu/hdd_data/fgi-spoofrepo/analysis/tgd_observability_audit_v1`
- `summary.json` SHA-256:
  `dfa4a09446b543a4df25f6e9071c12afca00423f0c43792c04bad79af4fef68e`
- `vector_comparison.csv` SHA-256:
  `d0c4afc294c7b972932e0dd433c7ef636aec0d4bc8b5ec2e0114dad4fc8d502d`
- `baseline_sensitivity.csv` SHA-256:
  `ef97f892770d54a344442c32c56061bbed2569a5ad7d4995482f3ca0c6d1ca71`

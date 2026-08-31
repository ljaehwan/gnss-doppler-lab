# FGI-SpoofRepo TGD CGC detection v1.1: negative result

Date: 2026-08-31

## Terminal decision

The preregistered one-shot result is `NOT_SUPPORTED`.  The frozen detector
produced no persistent false alarms in the clean interval, but also produced no
persistent detections in the stable post-onset interval.  No threshold,
interval, PRN, persistence rule, template, endpoint, or gate was changed after
score access, and this release must not be rerun.

| Endpoint | Frozen requirement | Result | Gate |
|---|---:|---:|---|
| Clean geometry bins | at least 60 | 80 | pass |
| Stable-post geometry bins | at least 60 | 70 | pass |
| Clean persistent alarm rate | at most 5% | 0.0% | pass |
| Stable-post persistent detection rate | at least 80% | 0.0% | fail |
| Median partial-F direction | post below clean | 0.471 below 0.555 | pass |
| Serial-bin AUC | descriptive only | 0.554 | -- |

The clean region had three raw alarms in 80 bins (3.75%) and the stable-post
region had six raw alarms in 70 bins (8.57%).  Neither region ever satisfied
the frozen causal 3-of-5 persistence rule.  There was no persistent alarm at
or after the documented 138 s onset.

## Input and execution validity

The official 9,961,930,752-byte `TGD_L1_E1.dat` file was stored on the HDD and
matched SHA-256
`10aad73665db7c5e530d9ef1d3b2fdb57bab0a7b9b19177a0128867fbad2606b`.
GNSS-SDR produced complex nine-tap tracking at 0.125-chip spacing.  The final
score-free identity-aware preflight retained twelve `SV_health == 0` GPS PRNs;
both primary intervals supplied at least eleven per geometry bin.

The first v1 detector release stopped before score access because the generic
OAKBAT preflight treated FGI's first positive observables `RX_time` as the RF
start.  Its pinned state remains `score_accessed=false`.  V1.1 separated that
480024.02-s observables reference from the true 480006.0-s RF-start reference,
without changing any detector or endpoint rule.

## Physical diagnosis

The attack did affect navigation.  Relative to the fixed clean NMEA position,
the stable-post PVT displacement had a median of 69.85 m and a 90th percentile
of 73.91 m.  Therefore the negative detector result is not explained by an
inactive spoofing signal.

The signed-delay observations did not become strongly common-geometric:

| Region | Median absolute delay | 90th-percentile absolute delay | Median geometry residual | Median fitted displacement |
|---|---:|---:|---:|---:|
| Clean | 0.163 chip | 0.275 chip | 0.778 | 0.207 chip |
| Stable post | 0.125 chip | 0.250 chip | 0.742 | 0.194 chip |

No delay estimate reached the 0.45-chip aperture edge.  Thus finite aperture
clipping is not the immediate cause.  The stronger interpretation is that,
after spoof takeover, the receiver tracking loop recenters its prompt on the
captured spoof peak.  A local prompt-referenced nine-tap profile then loses the
absolute common pseudorange displacement; the authentic peak may be too weak
or absent from the local aperture.  The frozen synthetic template estimator
therefore observes local profile distortion rather than the approximately
70-m PVT displacement.

## Consequence for the paper

This recording cannot be cited as successful external spoof detection.  It is
evidence against a broad claim that prompt-referenced CGC persists after full
takeover.  The defensible scope is narrower: CGC may characterize resolvable
dual-peak carry-off transitions, but a stable captured receiver requires an
additional invariant reference, such as pre-capture code-phase memory,
multi-epoch code-Doppler integration, or a wider correlator bank.  Any such
extension is a new preregistered method and must not be tuned or retested on
this opened TGD outcome.

## Frozen artifacts

- Config: `configs/experiments/fgi_spoofrepo_tgd_cgc_detection_v1.json`
- Protocol: `docs/results/fgi_spoofrepo_tgd_cgc_detection_protocol_v1.md`
- Runner: `scripts/run_fgi_spoofrepo_tgd_cgc_detection.py`
- Full HDD result:
  `/home/ubuntu/hdd_data/fgi-spoofrepo/analysis/tgd_cgc_detection_v1_1/summary.json`
- Result CSVs: the adjacent `delay_estimates.csv` and `geometry_scores.csv`


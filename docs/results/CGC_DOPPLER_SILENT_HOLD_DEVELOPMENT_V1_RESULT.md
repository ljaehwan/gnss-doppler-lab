# CGC Doppler-Silent Hold Development V1 Result

## Status

**DEVELOPMENT MECHANISM CONSISTENT.**

All six development mechanism checks passed on the five previously exposed
carrier-coupled receiver-RF pairs.  This result justifies implementing and
freezing a new fresh-static confirmation.  It is not a held-out result, an
exact reproduction of Tu et al., or evidence of superiority to all Doppler
detectors.

The audit reused retained RF and receiver outputs.  It generated no new attack
IQ and did not change the frozen CGC estimator, threshold, minimum support, or
persistence rule.

## Main result

The ordinary physically coherent carry-off produced the intended temporal
hand-off:

1. During the 5--10 s pull-off, every pair supplied all five one-second bins
   with at least five PRNs whose oracle primary--secondary Doppler separation
   exceeded the 3 Hz Tu-style input threshold.
2. During the 12--30 s stationary false-position hold, the maximum separation
   was only 0.053--0.070 Hz, so no hold bin supplied five PRNs at or above
   3 Hz.
3. The signed code displacement remained 38.6--64.3 m in median magnitude and
   stayed within the half-chip correlator aperture.
4. CGC remained persistently active throughout all 18 hold seconds in every
   pair, with a median hold raw-alarm rate of 0.944 across pairs.
5. No persistent CGC alarm occurred in the pre-attack interval.

| Pair | Pull-off Tu-available bins | Hold max Doppler difference | Hold median code offset | Hold raw CGC rate | Continuous Doppler-silent CGC interval |
|---|---:|---:|---:|---:|---:|
| Tokyo | 5/5 | 0.0528 Hz | 57.57 m | 1.000 | 18 s |
| London | 5/5 | 0.0702 Hz | 64.32 m | 0.944 | 18 s |
| Sao Paulo | 5/5 | 0.0606 Hz | 62.42 m | 1.000 | 18 s |
| Sydney | 5/5 | 0.0622 Hz | 38.60 m | 0.944 | 18 s |
| Nairobi | 5/5 | 0.0577 Hz | 50.96 m | 0.944 | 18 s |

This is the intended displacement-versus-velocity mechanism: after the false
receiver displacement stops changing, the differential carrier Doppler
collapses while the cross-satellite code-delay pattern remains.

## Actual-IQ spectral anchor audit

The fixed 20 ms actual-IQ audit used one baseline anchor, two pull-off anchors,
and three hold anchors for all receiver-tracked PRNs.  Four of 328 tracked
PRN-anchor rows were outside the preregistered receiver-centred +/-250 Hz
search.  They were retained with `search_contract_valid=false` and excluded
only from actual-IQ peak statistics; their oracle truth and CGC rows were not
excluded.

| Phase | Tracked PRN-anchor rows | IQ-evaluable rows | Oracle PRNs above 3 Hz | Any two blind peaks | Expected-frequency pair visible |
|---|---:|---:|---:|---:|---:|
| Baseline | 55 | 55 | 0 | 8 | 0 |
| Pull-off | 110 | 110 | 107 | 78 | 65 |
| Hold | 163 | 159 | 0 | 38 | 0 |

The strong pull-off increase is directionally consistent with dual-Doppler
observability.  However, the simple dominant-peak counter is not specific
enough to serve as a Tu baseline: sidelobes or noise created eight apparent
dual peaks in baseline and 38 during hold, even though no expected two-frequency
pair existed.  The truth-audited expected-frequency pair was visible for 65
pull-off rows and zero baseline or hold rows.

Therefore the blind spectral implementation is **not frozen for the future
confirmation**.  The next development step must implement primary-peak
cancellation or the FFT-CDS procedure described by Tu et al., and verify its
normal/hold false-peak behavior on these development data.  The final paper may
otherwise use only the more limited and clearly labelled oracle input-
availability comparison.

## Development checks

All returned `true`:

- all pairs have at least three Tu-available pull-off bins;
- all pairs have a maximum hold Doppler separation below 0.5 Hz;
- all pairs keep at least 25 m median code offset;
- all pairs remain inside the half-chip aperture;
- at least four pairs have a ten-second Doppler-silent CGC complement interval
  (all five actually have 18 s);
- zero persistent pre-attack CGC alarms.

## Provenance and artifacts

- Source summary SHA-256:
  `f57162a2f504f6a9d224889956805e69d4ce3bdf3d19386e506bef7fc70813f7`
- Development summary:
  `/home/ubuntu/hdd_data/cgc_doppler_silent_hold_development_v1/summary.json`
  (`sha256=2c44ff2c97c8a155f839d2383c2b624c6e180f5ef1d48ca2d55c58db0adfab44`)
- Full one-second timeline:
  `/home/ubuntu/hdd_data/cgc_doppler_silent_hold_development_v1/timeline.csv`
  (`sha256=3676c3ba092c116b05c15789125949922f67ee7d0287ad66466c789983d89f07`)
- Actual-IQ per-PRN anchors:
  `/home/ubuntu/hdd_data/cgc_doppler_silent_hold_development_v1/iq_anchor_metrics.csv`
  (`sha256=24c8951ad735ddfe1d2783201ca36c84dfb21fd762555a81703472fda3b745c3`)
- Vector figure:
  `/home/ubuntu/hdd_data/cgc_doppler_silent_hold_development_v1/doppler_silent_hold_timeline.svg`
  (`sha256=f03aabafc7045a031864525011146c3d070ad9c6fadc496bb52d585940aeabf8`)

The development output occupies approximately 236 KiB; the retained source RF
remains unchanged on the HDD dataset mount.

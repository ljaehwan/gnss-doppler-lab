# CGC Code/Carrier Fresh Static V1 Result

## Primary decision

**NOT SUPPORTED** under the preregistered release gates.

The ordinary carrier-coupled control produced persistent CGC detections in all
five fresh static geometries.  The authentic-Doppler-locked intervention
produced at least one persistent detection in four of five geometries, but the
detections were not sufficiently prompt or stable to support the stronger
claim that code-domain CGC is robust to removal of spoof-carrier Doppler
evidence.

This is the single released evaluation.  No pair was substituted and no
threshold, persistence rule, or gate was changed after metrics were exposed.

## Frozen design

- Freeze commits:
  - candidate pool and support protocol: `1f786f6b9230a671eb6b6fa3b39f29f02c988d3d`
  - selected release, runner, and gates: `d473aaf`
- Candidate selection used only a one-second line-of-sight support probe.  No
  25 MHz selected-pair RF or CGC score was accessed before the final release.
- Five previously unused exact position--UTC geometries were selected:
  Tokyo, London, Sao Paulo, Sydney, and Nairobi.
- Each geometry used a matched pair with byte-identical pre-attack RF:
  1. carrier-coupled code carry-off;
  2. the same code carry-off with spoof-carrier range and range rate locked to
     the authentic signal.
- Attack onset: 5 s; power ramp end: 10 s; primary hold interval: 12--30 s.
- Detector: complex 9-tap signed-delay estimator, clock-centered Partial-F
  test with `p_F <= 0.06028418845288192`, at least eight PRNs, and 3-of-5
  one-second persistence.

## Aggregate results

| Metric | Carrier-coupled | Authentic-Doppler-locked |
|---|---:|---:|
| Persistent detection pairs | 5/5 | 4/5 |
| Median hold raw-alarm rate | 0.944 | 0.333 |
| Median latency from onset | 4 s | 15.5 s among detected pairs |
| Persistent pre-attack alarms | 0 | 0 |

All five code/carrier truth audits passed.  In the Doppler-locked condition,
the spoof code trajectory was identical to the carrier-coupled condition while
the spoof carrier range and range rate were exactly equal to the authentic
component.  The held code--carrier separation had geometry-dependent median
magnitudes of approximately 41--68 m and maxima of approximately 88--100 m.

The following preregistered gates failed:

- median locked hold raw-alarm rate: 0.333, below the required 0.5;
- median locked detection latency: 15.5 s, above the allowed 10 s.

The pair-count, truth-invariant, 5/5 carrier-coupled detection, at-least-4/5
locked detection, zero persistent pre-attack alarm, and carrier-coupled hold
rate gates passed.

## Per-geometry result

| Geometry (ENU target) | Startup PRNs | Coupled latency / hold rate | Locked latency / hold rate | Locked median hold `p_F` |
|---|---:|---:|---:|---:|
| Tokyo (+100, 0, 0) m | 11 | 4 s / 1.000 | 18 s / 0.389 | 0.1484 |
| London (0, +100, 0) m | 14 | 3 s / 0.944 | 13 s / 0.333 | 0.2046 |
| Sao Paulo (-80, +60, 0) m | 14 | 3 s / 1.000 | 6 s / 0.889 | 0.0165 |
| Sydney (+60, -80, 0) m | 12 | 5 s / 0.944 | no detection / 0.167 | 0.1504 |
| Nairobi (+60, 0, +80) m | 13 | 5 s / 0.944 | 23 s / 0.278 | 0.1600 |

Only Sao Paulo showed stable locked-condition geometry evidence throughout the
hold interval.  In the other four geometries, the locked median `p_F` remained
above the frozen alarm threshold, so the few alarms were intermittent or late.

## Interpretation and paper boundary

This result supports the narrower observation that CGC is highly effective for
the tested conventional carrier-coupled carry-off signals, with zero persistent
pre-attack alarms.  It does **not** support claiming that the present complex
9-tap estimator reliably preserves CGC when the secondary code is pulled off
while its carrier Doppler remains authentic.

The result also isolates a useful failure mode: removing differential carrier
motion changes the observability of the two-component correlation profile even
though the code-delay geometry is unchanged in truth.  The present paper must
therefore avoid using the Seoul development pilot as evidence of general
Doppler independence.  A follow-up experiment should vary relative carrier
phase and carrier-frequency offset explicitly, without changing this released
result, to determine whether the loss is caused by coherent phase-dependent
delay observability or by the current signed-delay estimator.

## Artifacts

- Primary summary:
  `/home/ubuntu/hdd_data/cgc_code_carrier_fresh_static_v1/summary.json`
  (`sha256=f57162a2f504f6a9d224889956805e69d4ce3bdf3d19386e506bef7fc70813f7`)
- Delay estimates:
  `/home/ubuntu/hdd_data/cgc_code_carrier_fresh_static_v1/analysis/delay_estimates.csv`
  (`sha256=85b6fcd9beddbcae749c6630a8510c3ed1d69b02e631188c814c50f98d92942f`)
- Geometry scores:
  `/home/ubuntu/hdd_data/cgc_code_carrier_fresh_static_v1/analysis/geometry_scores.csv`
  (`sha256=1c156c4057dac98d4d29c494bec69aaabbfb7d214f1aed5c64e54c40f7459e6a`)
- Retained campaign size: approximately 36 GB on the HDD dataset mount.

The claim boundary remains: fresh static simulated receiver-RF evidence only;
this is not field validation, universal spoof detection, or evidence that all
Doppler-based detectors fail.

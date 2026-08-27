# CGC real-IQ detection v1 result

## Decision

`REAL_SPOOF_DETECTION_NOT_SUPPORTED`. Seven of eight frozen primary gates
passed. DS7 produced a real persistent spoof alarm with zero persistent alarms
in the pooled negative data, but the first persistent alarm arrived 119 seconds
after the nominal onset, exceeding the preregistered 60-second maximum.

No threshold, interval, persistence rule, source, gate, or candidate component
was changed after release.

## Primary result

The cleanStatic calibration lower-5% residual threshold was `0.330236`. A
persistent alarm required three threshold crossings within five consecutive
one-second bins.

| Data region | Bins | Raw alarms | Persistent alarms | Median residual |
|---|---:|---:|---:|---:|
| cleanStatic calibration | 90 | 5 (5.56%) | 0 | 0.6613 |
| cleanDynamic locked normal | 30 | 2 (6.67%) | 0 | 0.5866 |
| DS7 stable pre | 60 | 7 (11.67%) | 0 | 0.6680 |
| DS7 stable post | 161 | 52 (32.30%) | 47 (29.19%) | 0.4655 |

The pooled primary negative set (`cleanDynamic + DS7 stable pre`) had 0/90
persistent alarms. Its seven correlation-distortion-enriched bins also had zero
persistent alarms. DS7 serial-bin AUC was 0.6543. The first raw DS7 threshold
crossing was available at 119 seconds, but crossings were initially isolated;
the first frozen three-of-five persistent alarm was not available until 219
seconds.

## Secondary replication

| Scenario | Pre persistent alarms | Post persistent alarms | First persistent end | Delay from nominal onset | Serial-bin AUC |
|---|---:|---:|---:|---:|---:|
| DS1 | 0/60 | 14/352 | 173 s | 73 s | 0.6552 |
| DS2 | 0/60 | 34/348 | 174 s | 74 s | 0.7117 |
| DS3 | 0/60 | 22/348 | 135 s | 35 s | 0.6344 |

These are replication diagnostics because their threshold-free direction was
known before this experiment. DS7 was the primary outcome-unseen CGC attack.

## Interpretation

The real-IQ result supports the physical direction and specificity of the
clock-centered geometry statistic: all four attacks had lower post-onset median
residuals, each eventually triggered persistent alarms, and no persistent alarm
occurred in the primary negative pool or its high-distortion subset.

It does not support the frozen operational detector. A low normal-only absolute
threshold plus a three-of-five rule is too intermittent and slow for the subtle
carrier-aligned DS7 attack. The next detector experiment should freeze a
normal-calibrated sequential evidence accumulator and test it on a new untouched
attack recording; this result must not be retuned or reclassified.

TEXBAT still lacks ground-truth reflector/path annotations. The high-distortion
normal subset is multipath-enriched, not a labeled real-multipath class, and the
result is replay-IQ rather than live-field false-alarm validation.

The immutable raw summary is `artifacts/cgc_real_detection_v1/summary.json`,
SHA-256 `d7fa0024628a231dae2ec1da2f2bae6d4b19fb3fdf97886913d9520349aa0a63`.

# Frozen CGC TEXBAT external protocol v1

## Question and boundary

Apply the already frozen complex-template clock-centered CGC candidate to preserved real TEXBAT DS1–DS3 complex-nine-tap epochs. Test only whether the residual changes in the preregistered spoof-consistent direction from stable pre-onset to stable post-onset data.

TEXBAT DS1–DS3 do not provide a labeled independent-multipath class. Therefore this experiment cannot test real multipath-versus-spoof classification, estimate a field false-alarm rate, or fit an absolute alarm threshold. It is complementary external change evidence for the physical mechanism.

## Frozen inputs and support

The NPZ exports preserve raw receiver epochs as complex I/Q in nine tap positions plus PRN, sample count, time, channel, segment, and C/N0 identity arrays. Their export manifests and SHA-256 values are pinned in the config. No smoothing or model inference is stored in these inputs.

Before any CGC score access, input-only preflight established 11 healthy tracked PRNs per recording. Every stable-pre bin has at least eight PRNs (60 bins per recording), and at least 348 stable-post bins per recording have eight PRNs. The fixed receiver position is the median of 700 checksum-valid pre-onset NMEA GGA samples per recording. These support observations may not be used to alter the intervals or score.

## Frozen transformation

Use the same template bank and nearest-template signed delay estimator as the simulation candidate. For each recording, estimate every epoch in chunks of 50,000 without changing the estimator. Within each one-second bin and PRN, take the median delay. Use only explicitly healthy broadcast ephemerides. Compute receiver-to-satellite ECEF LOS at the bin-center TOW from the pinned ephemeris and fixed pre-onset NMEA receiver position.

For bins with at least eight PRNs, fit LOS plus a common clock intercept and calculate

`R_c = SSE_full / sum(w_i (delay_i - weighted_mean(delay))^2)`.

No C/N0 weights are used, matching the frozen candidate. The detection score remains `-R_c`.

## Frozen intervals and endpoint

- Stable pre: bin start in `[30,90)` seconds.
- Transition exclusion: `[90,110)` seconds.
- Stable post: bin start at or after 110 seconds.

For each DS recording, the primary change is `median R_c(pre) - median R_c(post)`. Positive means the post-onset recording has the higher spoof score and is the only preregistered direction. One scenario-level change is the primary unit; serial-bin AUC is descriptive only.

Report `DIRECTIONALLY_CONSISTENT` only if all three scenario changes are positive, all three recordings complete, every stable-pre region supplies at least 50 geometry bins, every stable-post region supplies at least 300 geometry bins, and every included geometry bin has at least eight PRNs. Otherwise report `NOT_DIRECTIONALLY_CONSISTENT`. No exclusion, tuning, interval change, calibration, or retest is allowed after result access.

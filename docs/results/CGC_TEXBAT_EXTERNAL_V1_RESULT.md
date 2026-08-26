# CGC TEXBAT external v1 result

## Decision

`DIRECTIONALLY_CONSISTENT`. All three frozen TEXBAT scenarios changed in the preregistered direction, and every support gate passed. No threshold, calibration, interval change, scenario exclusion, tuning, or retest was used.

| Scenario | Stable-pre bins | Stable-post bins | Min PRNs | Pre median residual | Post median residual | Pre − post | Serial-bin AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| DS1 | 60 | 352 | 10 | 0.6491 | 0.5161 | 0.1330 | 0.6552 |
| DS2 | 60 | 348 | 10 | 0.6914 | 0.4922 | 0.1992 | 0.7117 |
| DS3 | 60 | 348 | 8 | 0.7067 | 0.5773 | 0.1294 | 0.6344 |

The median scenario-level residual decrease is 0.1330. Because the detection score is the negative residual, every recording has a higher spoof score in the stable post-onset region than in the stable pre-onset region.

## Interpretation

The effect is moderate rather than perfect on real replay data: descriptive bin AUC ranges from 0.634 to 0.712. Nevertheless, the same fixed physical candidate moves in the predicted direction in three independent TEXBAT spoof recordings, after using real complex nine-tap receiver epochs, broadcast ephemerides, and a pre-onset NMEA position.

Together with the fresh simulation result, this establishes two different pieces of evidence:

1. Fresh controlled receiver-RF simulation separates independent multipath from carry-off spoofing on 3/3 held-out pairs with pair-block AUC 1.0.
2. Real TEXBAT spoof recordings reproduce the predicted pre/post geometry-residual decrease on 3/3 scenarios.

## Limitation

TEXBAT DS1–DS3 contain spoof pre/post regions but no labeled independent-multipath class. This result therefore cannot be reported as real-data multipath-versus-spoof classification accuracy, a false-alarm rate, or validation of an absolute detector threshold. It is external directional validation of the proposed physical mechanism.

The immutable raw summary is `artifacts/cgc_texbat_external_v1/summary.json`, SHA-256 `22c4f3c06b953b43fb1a34dee9692d0e322243b2d8e760ec7e7c4c38ae6bed2b`.

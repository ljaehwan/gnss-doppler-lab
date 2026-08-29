# TUNI GPS SS-29 support preflight v1.1 result

## Outcome

`INSUFFICIENT_SUPPORT`

SS-29 is scientifically well matched to the intended multipath-versus-partial-
spoofing question, but it cannot be evaluated by the frozen `N>=8` CGC model.
The 10-second prefix yielded six distinct tracked PRNs and at most five eligible
PRNs in any one-second bin. No primary bin reached eight PRNs.

| Quantity | Result |
|---|---:|
| Raw IQ prefix accessed | 2,000,000,000 bytes (first 10 seconds) |
| Analysis interval | `[5,10)` s |
| Discovered PRNs | G01, G02, G16, G18, G21, G27 |
| Maximum eligible PRNs per bin | 5 |
| `N>=8` primary bins | 0 of 5 |
| G01 eligible bins | 5 |
| G02 eligible bins | 5 |
| G21 eligible bins | 5 |
| G32 eligible bins | 0 |

The eligible PRN sets were `{1,2,16,21}` in bins 5–7 and
`{1,2,16,18,21}` in bins 8–9. Epoch density itself was sufficient (up to 1,000
epochs per PRN/bin); the failure is simultaneous satellite and target support,
not sparse telemetry.

## Invalid v1 output and correction

The initial v1 postprocessor divided `PRN_start_sample_count` by the 50 MHz RF
input rate instead of the receiver's 5 MHz internal tracking rate. This
compressed the time axis by ten, falsely found no rows in `[5,10)`, and emitted
an invalid zero-support result. Before correction, that output and the complete
31-file MAT tree were hash-pinned.

Released v1.1 commit `232e87ec89b95eeb9de81d07a98a2b777b3f85e6`
changed only the timebase metadata. It did not download or reprocess IQ and did
not change any support rule. The corrected external summary SHA-256 is
`68c1c7ef4847b7d2c3861825f8bb930738c805fea0f30532893e053318f9ee73`.

## Scientific interpretation

No delay template, geometry statistic, partial-F value, alarm, or detector
score was computed. Therefore this result says nothing about whether the model
would detect SS-29 if it had enough satellites. It only proves that SS-29 is
outside the model's frozen support domain.

The full 29,999,832,000-byte SS-29 recording should not be downloaded for this
experiment. A new confirmatory dataset is still required, with at least eight
simultaneously sustained GPS L1 C/A PRNs and all attacked PRNs tracked.

# CGC PRN-support sweep v1 result

## Decision

The pre-registered overall decision is
`FINITE_SUPPORT_MECHANISM_NOT_SUPPORTED` because one of four gates failed.
The direct overfitting hypotheses both passed, but the frozen partial-F
correction narrowly missed the required specificity pass rate at seven PRNs.

| PRNs | Legacy pre-alarm median (range) | Partial-F pre-alarm median (range) | Partial-F specificity trials | DS7 detection trials | Median delay |
|---:|---:|---:|---:|---:|---:|
| 7 | 20.00% (0--43.33%) | 0% (0--6.67%) | 58/64 = 90.63% | 59/64 = 92.19% | 124.0 s |
| 8 | 8.33% (0--16.67%) | 0% (0--3.33%) | 64/64 = 100% | 64/64 = 100% | 124.5 s |
| 9 | 0% (0--5.00%) | 0% | 64/64 = 100% | 64/64 = 100% | 119.0 s |
| 10 | 0% | 0% | 64/64 = 100% | 64/64 = 100% | 119.0 s |

The median unadjusted pre-attack residual rises monotonically with support:
0.4583, 0.5298, 0.5982, and 0.6642 for N = 7, 8, 9, and 10. The Spearman
correlation between PRN count and residual is 1.0. Meanwhile the legacy false-
alarm correlation with PRN count is -0.9487. These results directly support
the physical explanation that the four-parameter direction-plus-clock model
can fit unrelated delay errors too easily at low support.

Partial-F removes most of this support dependence: its median persistent false-
alarm rate is zero at every N. However, six of the 64 seven-PRN trials contain
four persistent false-alarm bins out of the 60-bin stable-pre interval, giving
6.67% rather than the frozen maximum 5%. Therefore the 95%-of-trials
specificity gate passes in only 90.63% of trials and cannot be rescued by
changing a threshold after the result.

## Claim-safe interpretation

This audit confirms the finite-support overfitting mechanism. It does not show
that partial-F completely solves seven-PRN specificity. The paper-safe claim is
that degrees-of-freedom normalization sharply reduces the low-support bias and
passes at N = 8--10, while a residual seven-PRN boundary remains.

The correction also retains at least 92.19% DS7 trial-level detection across
all tested support sizes, but detection is late: median delay is 119--124.5 s.
Fast detection is not supported by this experiment.

N = 11 has only three stable-pre and 44 stable-post bins, and N = 12 has none.
They are excluded for insufficient support rather than reported as favorable
small-sample results.

## Provenance

- Pre-registered release commit:
  `ad6b7e5fcc632bfeee7325dbd9234bb3c82cf074`.
- Config SHA-256:
  `0d631b1566ad499d1e49bcbb9349e76a173351552bfd46d0da2a6b80cd6c3cd0`.
- Trial metrics SHA-256:
  `54b2e8127303b2eef8a2aac360e6dcee9b3d58fd2a4369af26c38c526f09e2c4`.
- Support curve SHA-256:
  `b10161da6ca078483b413bdfee1f09cc3b2d69c76125dbe694ea8a39bbcdae20`.
- Summary SHA-256:
  `66b43604ef829bc3b793fd55669ca68fb5344499bfd4b5b4bfc1b666c61dad22`.

## Next experiment

Do not tune another seven-PRN threshold on DS7 or GNSS-OpenIF S1. The next
valid evidence is frozen transfer to the already downloaded TUNI GPS C-5 and
SS-17/18/20 recordings. Any condition-aware or robust-regression extension
must be developed on calibration data and judged on a new held-out recording.

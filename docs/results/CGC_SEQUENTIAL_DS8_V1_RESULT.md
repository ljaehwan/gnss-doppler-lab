# CGC sequential DS8 v1 result

## Decision

`SEQUENTIAL_DS8_NOT_SUPPORTED`. Eight of nine frozen gates passed. The
outcome-unseen DS8 recording produced a real Page-CUSUM spoof alarm with no
alarm in either normal trace or in DS8 before its official onset. The first
post-onset alarm was available at 201 s, however, which is 91 s after the
official 110 s onset and exceeds the preregistered 60 s maximum.

No score, normal interval, Page-CUSUM constant, threshold, input, time role, or
gate was changed after release commit `eeaf383`.

## Frozen result

The normal baseline was the 90 cleanStatic bins in `[330,420)`. Its median
clock-centered residual was `0.661295` and its robust MAD scale was `0.239878`.
The frozen sequential rule was

`C_t = max(0, C_(t-1) + clip((0.661295-r_t)/0.239878,-3,3) - 0.5)`

with an alarm at `C_t >= 5`.

| Data region | Bins | Sequential alarms | Maximum CUSUM | Median residual |
|---|---:|---:|---:|---:|
| cleanStatic baseline | 90 | 0 | 2.2002 | 0.6613 |
| cleanDynamic locked normal | 30 | 0 | 3.8417 | 0.5866 |
| DS8 pre-onset | 79 | 0 | 1.9892 | 0.6749 |
| DS8 post-onset | 361 | 187 | 56.8006 | 0.5859 |
| DS8 stable post (`>=120 s`) | 351 | 187 | 56.8006 | 0.5828 |

The descriptive DS8 pre-versus-stable-post serial-bin AUC was `0.5904`. There
were at least nine healthy PRNs before onset and ten after onset.

## Why the timely-detection gate failed

This run did not merely miss an early residual shift because of a high decision
threshold. The physical residual did not move in the spoof direction during
most of the first 80 s after onset:

| Bin-end interval | Median residual | Maximum CUSUM | Alarms |
|---|---:|---:|---:|
| `[110,130)` | 0.7101 | 0.5517 | 0 |
| `[130,150)` | 0.7738 | 1.2851 | 0 |
| `[150,170)` | 0.6283 | 0.7974 | 0 |
| `[170,190)` | 0.7314 | 0.7117 | 0 |
| `[190,210)` | 0.5503 | 5.4739 | 1 |
| `[210,230)` | 0.2839 | 18.3998 | 10 |
| `[230,270)` | 0.2905 | 56.8006 | 40 |

The first large coherent lower-residual excursion appeared around 190--210 s,
and the first threshold crossing followed at 201 s. Therefore a detector based
only on this physical statistic cannot reliably claim onset-time detection for
DS8 from the current result.

## Interpretation

The independent DS8 test strengthens two narrower findings:

1. The clock-centered cross-satellite score is specific in these replay-IQ
   traces: no sequential alarm occurred in 120 normal bins or 79 DS8 pre-onset
   bins.
2. Once the DS8 carry-off produced a strong coherent correlator deformation,
   the statistic detected it repeatedly: 187 of 351 stable-post bins were in
   alarm.

It does not validate the sequential detector as a prompt operational alarm.
Together with the delayed DS7 result, it motivates a new physical question:
what counterfeit displacement, authentic-to-counterfeit power ratio, or
correlator-lobe separation is required before cross-satellite delay geometry is
observable? That is a post-result hypothesis and must be tested separately; it
is not a reclassification of this failed gate.

TEXBAT still has no reflector/path ground truth, so this result does not by
itself establish strict real-multipath-versus-spoof classification. It is also
replay-IQ rather than a live-field false-alarm experiment.

The immutable raw summary is
`artifacts/cgc_sequential_ds8_v1/summary.json`, SHA-256
`d7eacede40f236fb371c3b521d93a2daae0b4ade1e54820c1660b282468b90e0`.

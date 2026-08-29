# CGC PRN-support sweep protocol v1

## Purpose

This protocol tests the finite-support mechanism behind the observed CGC
false alarms. The question is whether the unadjusted clock-centered geometry
residual becomes spuriously small when only a few satellites constrain the
four-parameter direction-plus-clock model, and whether the already frozen
partial-F score corrects that dependence.

This is a mechanism audit of previously sealed TEXBAT/DS7 delay estimates. It
does not constitute a new independent RF dataset.

## Frozen inputs

- Frozen CGC result:
  artifacts/cgc_real_detection_v1/summary.json, SHA-256
  d7fa0024628a231dae2ec1da2f2bae6d4b19fb3fdf97886913d9520349aa0a63.
- Frozen delay estimates SHA-256:
  9d0ac4a7a6f9b4c59367862366d638309c8f3dcb1effa28b9e2409b6f5700840.
- Frozen seven-PRN partial-F audit SHA-256:
  c07dee4ecc82e9d0ce98c98d7e8ba9b00ba6ff86400406e2cbb86821708c07d4.
- Legacy residual alarm threshold: 0.33023636533817136.
- Partial-F tail-score threshold: 0.06028418845288192.
- Persistence: three positive bins in a causal five-bin window.

No threshold, interval, support size, persistence rule, or trial count may be
changed after release.

## Physical models

For per-satellite delay estimates y and line-of-sight vectors u, compare

\[
H_0: y_i=c+e_i
\]

with

\[
H_1: y_i=-u_i^\mathsf{T}d+c+e_i.
\]

The alternative uses four parameters: three components of common displacement
d and one clock term c. Its residual ratio is

\[
r=\mathrm{SSE}_1/\mathrm{SSE}_0.
\]

The support-normalized score is

\[
F=\frac{(\mathrm{SSE}_0-\mathrm{SSE}_1)/3}
        {\mathrm{SSE}_1/(N-4)}, \qquad
p=P(F_{3,N-4}\ge F_{\mathrm{observed}}).
\]

The empirically calibrated p value is used as a support-normalized ranking
score, not as an exact probability under real correlated receiver errors.

## Frozen sampling

Evaluate exact support sizes N = 7, 8, 9, and 10. For each N, run 64
deterministic trials. Within every scenario/bin, rank PRNs by

SHA256(scenario:trial:PRN)

and select the first N. Because N is not included in the rank, the subsets are
nested: the seven-PRN set is contained in the eight-PRN set, and so on.

Every trial must contain the same 60 DS7 stable-pre bins from receiver seconds
30 through 89 and the same 161 stable-post bins from seconds 110 through 270.
The attack onset for causal delay reporting is second 100.

Preflight support inspection found only three stable-pre and 44 stable-post
bins with eleven PRNs; twelve-PRN support is absent. N = 11 and N = 12 are
therefore reported as insufficient support and cannot enter the primary
comparison.

Every selected geometry must have full rank four. Record the design-matrix
condition number as a descriptive geometry-quality covariate.

## Pre-registered decisions

For each N, report legacy and partial-F raw and persistent alarm rates in the
stable-pre negative interval, persistent DS7 detection rate, and causal
detection delay.

The finite-support mechanism is marked SUPPORTED only if all gates pass:

1. at every N, at least 95% of the 64 trials have partial-F stable-pre
   persistent false-alarm rate no greater than 5%;
2. at every N, partial-F persistent DS7 detection occurs in at least 80% of
   trials;
3. the median legacy stable-pre persistent false-alarm rate at N = 7 is greater
   than at N = 10; and
4. the median unadjusted stable-pre residual at N = 7 is smaller than at
   N = 10.

Detection delay is reported without a new gate because the earlier seven-PRN
audit already exposed a sensitivity-delay tradeoff. No outcome-driven rescue
threshold is allowed.

## Execution integrity

The runner refuses to execute from a dirty worktree and records the release
commit. It verifies all frozen input hashes before reading scores and writes a
trial table, a four-row support curve, and a hash-bound summary.

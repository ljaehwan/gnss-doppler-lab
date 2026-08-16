# TRACE Stage-0 R1 native-cadence result

Configuration frozen before this TRACE-R1 evaluation.

## Outcome

Route: **B**. Final verdict: `NEEDS_TRACE_SPECIFIC_RECEIVER_DUMP`.

The retained post-synchronization rows are about 20 ms apart, but the actual
receiver loop and NCO update every 1 ms. A retained row's `code_freq_chips` and
`carrier_doppler_hz` apply only to the next 1 ms input buffer; the next retained
complex correlator follows 19 unobserved loop updates. Therefore the required
row-t action to row-(t+1) correlator mapping did not hold.

All four core scenarios have audited post-onset 20 ms row support, but it is not
causal action support and was not scored:

| Scenario | >=4-PRN post-onset blocks | Maximum PRNs/block |
|---|---:|---:|
| DS3 | 679 | 11 |
| DS7 | 322 | 11 |
| OS3 | 720 | 11 |
| OS4 | 720 | 11 |


Clean holdout/external FPR, Full/A1/A2/B0 comparison, action shuffle, physical
controls, and detection delays are unavailable. No model, threshold, attack
score, ROC/PR result, or performance claim was produced.

Prompt referencing removes common carrier phase, so full Doppler phase rotation
was not applied to normalized taps. Doing so would double-apply a removed global
phase and still would not recover the omitted actions.

The TRACE hypothesis remains worth pursuing because this result is an input
observability failure, not evidence for or against action equivariance. The one
recommended next action is to generate authenticated native-1ms receiver dumps
with complex nine taps, applied next-buffer code/carrier actions, sample stamps,
C/N0, lock, integration interval, and loop-boundary flags, then rerun the frozen
R1 protocol from Phase A.

## Commits and verification

- Preregistration freeze: `c4099cc837190c67c1bbc97781858594a95c4983`
- Result commit: `c282bd278760aefe29e4275e15cf90e74f4a78e4`
- Fresh-clone result verifier: PASS
- Final verification tip: the commit containing this report; its exact SHA is
  recorded in the task-level final self-report because a commit cannot contain
  its own content-derived SHA.

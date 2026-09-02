# Temporal/Observability-Stabilized CGC Development Result V1

## Status

**PROMISING; REQUIRES A NEW UNTOUCHED RECEIVER-RF CAMPAIGN.**

This is a development result, not a replacement for the released CGC result.
The same five exact-Doppler-lock pairs, five synthetic multipath controls, one
Tokyo phase sweep, and previously accessed GNSS-OpenIF S1/S2 recordings were
reused to design and diagnose the candidate.

## Question

The frozen CGC becomes weak when authentic and counterfeit carriers have the
same Doppler and nearly the same carrier phase.  In that state, several
two-path dictionary delays have similar cost and the per-second signed-delay
estimate can jump between nearby templates.  Can a causal physical continuity
rule stabilize the signed-delay sensor without changing the nine-tap bank,
Partial-F threshold, or LOS geometry model?

## Candidate logic

For each PRN, retain the latest five one-second signed-delay estimates and use
the causal median

\[
\widetilde{\delta}_i(t)=\operatorname{median}\{\widehat{\delta}_i(s):
t-4\le s\le t\}.
\]

The ordinary clock-centered CGC and support-normalized Partial-F statistic are
then computed from \(\widetilde{\delta}_i(t)\).  No future sample is used.

A second, physically distinct condition is needed.  Geometry consistency is a
cause test, not proof that a second component is observable.  Define

\[
A_\delta(t)=\sqrt{\frac{1}{N}\sum_{i=1}^{N}
\left(\widetilde{\delta}_i(t)-\overline{\widetilde{\delta}}(t)\right)^2}.
\]

The development candidate for the next fresh test is therefore:

- \(A_\delta<0.10\) chip: **abstain / secondary delay not yet observable**;
- \(A_\delta\ge0.10\) chip and \(p_F\le0.06028418845\): **spoof-like cause**;
- \(A_\delta\ge0.10\) chip and \(p_F>0.06028418845\): **multipath-like cause**;
- issue a persistent label after three positives among the latest five bins.

The 0.10-chip value is four steps of the frozen 0.025-chip delay dictionary.
It was nevertheless selected after inspecting reused outcomes, so it is not a
validated or deployable threshold.  It must be frozen before a new campaign.

## What remains unchanged

- complex nine-tap aperture and prompt-phase rotation;
- frozen deterministic two-path dictionary;
- signed delay rather than absolute delay;
- all eligible PRNs, with at least eight PRNs;
- clock-only versus LOS-plus-clock nested model;
- Partial-F threshold \(p_F\le0.06028418845\);
- three-of-five final persistence.

This is not a threshold-only patch.  The median stabilizes the physical
observable; the amplitude condition prevents a cause classifier from forcing
a spoof/multipath label when the secondary delay is below its useful support.

## Reused development results

The primary comparison contains 90 held exact-lock spoof bins and 149
synthetic independent-multipath bins.

| Metric | Frozen per-bin CGC (W=1) | Five-bin median only | Five-bin median + observability abstention |
|---|---:|---:|---:|
| Partial-F/gated AUC | 0.8383 | 0.9482 | 0.9787 |
| Exact-lock hold raw alarm rate | 41.11% | 55.56% | 55.56% |
| Synthetic multipath raw alarm rate | 2.68% | 2.01% | 0.00% |
| Exact-lock pre-attack persistent count | 0 | 1 | 0 |
| Median truth-direction \(R^2\) | 0.4342 | 0.5220 | 0.5220 |

The median alone is not acceptable as a final modification: it smooths random
normal delays too, and introduced a pre-attack persistent false alarm in the
Nairobi pair.  The observability abstention removed that event without reducing
held-interval raw sensitivity in these reused pairs.

### Five exact-lock pairs at W=5

| Pair | Hold raw rate | Median \(p_F\) | Median-only latency | Joint-candidate latency |
|---|---:|---:|---:|---:|
| Tokyo | 55.56% | 0.0483 | 9 s | 9 s |
| London | 61.11% | 0.0332 | 3 s | 10 s |
| Sao Paulo | 100.00% | 0.0010 | 8 s | 8 s |
| Sydney | 33.33% | 0.1291 | 8 s | 15 s |
| Nairobi | 27.78% | 0.1740 | 0 s (pre-event history artifact) | 8 s |

The amplitude condition makes early labels more conservative.  Its latency
penalty is expected because it explicitly waits until cross-satellite delay
spread is observable.

### Carrier-phase intervention

The W=5 phase values below were computed after the five-bin window was chosen.
The baseline is the previously sealed single-Tokyo-geometry phase audit.

| Global relative phase | Baseline median \(p_F\) | W=5 median \(p_F\) | Baseline latency | W=5 latency | Baseline/W=5 truth \(R^2\) |
|---:|---:|---:|---:|---:|---:|
| 0 deg | 0.1484 | 0.0483 | 18 s | 9 s | 0.336 / 0.474 |
| 45 deg | 0.00386 | 0.000681 | 6 s | 8 s | 0.746 / 0.892 |
| 90 deg | 0.000417 | 0.000129 | 5 s | 8 s | 0.889 / 0.932 |
| 135 deg | 0.0000232 | 0.0000169 | 2 s | 4 s | 0.950 / 0.962 |
| 180 deg | 0.0340 | 0.00282 | 14 s | 5 s | 0.716 / 0.854 |

Temporal stabilization helps the ambiguous 0-degree and cancellation-prone
180-degree cases most.  It adds two to three seconds in already easy phase
conditions, which is the expected robustness/latency trade-off.

## Benign compatibility checks

These are post-hoc compatibility checks, not new external validation.

| Control | Frozen persistent result | W=5 median only | W=5 + abstention |
|---|---:|---:|---:|
| Five earlier train-normal streams | 1/5 pairs alarmed | 2/5 pairs alarmed | 0/5 pairs alarmed |
| Five synthetic multipath streams | 0.00% bins | 0.67% bins | 0.00% bins |
| GNSS-OpenIF S1 real multipath | 2.38% bins | 0.00% bins | 0.00% bins |
| GNSS-OpenIF S2 real multipath | 3.41% bins | 3.41% bins | 0.00% bins |

The real recordings have no machine-readable per-epoch multipath truth, and
both were accessed before this candidate existed.  They show compatibility,
not confirmatory specificity.

## Rejected alternatives

- Expanding the authentic-center template grid degraded the 0-degree result
  (median \(p_F=0.345\)) and was rejected.
- Folding signed delay to absolute delay discarded useful direction
  information (development AUC 0.737) and was rejected.
- Leave-one-satellite-out predictive geometry raised AUC only from 0.838 to
  0.848; the complexity was not justified.
- Selecting a geometry-favorable delay candidate and scoring it on the same
  PRNs was rejected as circular selective overfit.

## Interpretation

The evidence supports a narrow conclusion: temporal continuity can repair part
of the per-second template ambiguity produced by a Doppler-locked, phase-aligned
mixture.  It does not remove the finite-aperture onset limit, guarantee every
carry-off direction, or turn CGC into a stage-one anomaly detector.

The abstention state is essential.  Below observable delay spread the correct
output is “insufficient secondary-delay evidence,” not “multipath” and not
“spoof.”

## Required next experiment

Before modifying the paper's primary method or claiming improvement:

1. freeze W=5, \(A_\delta=0.10\) chip, the original Partial-F threshold, and
   three-of-five persistence;
2. generate untouched static receiver-RF pairs with new location/time/LOS
   geometries and independently varied per-PRN relative carrier phases;
3. include normal, independent multipath, Doppler-coupled carry-off, and exact
   Doppler-locked carry-off under matched RF/front-end conditions;
4. report abstention rate, spoof/multipath conditional accuracy, persistent
   false alarms, detection probability, and latency separately;
5. do not retune after accessing the new outcomes.

Only that test can decide whether this candidate replaces the current frozen
CGC in the WCL manuscript.

## Reproduction

```bash
.venv/bin/pytest -q tests/test_temporal_cgc.py
.venv/bin/python scripts/evaluate_cgc_temporal_stabilization_dev.py
.venv/bin/python scripts/audit_cgc_temporal_benign_controls_dev.py
.venv/bin/python scripts/audit_cgc_temporal_openif_controls_dev.py
```

Machine-readable outputs are under
`artifacts/cgc_temporal_stabilization_dev_v1/`.

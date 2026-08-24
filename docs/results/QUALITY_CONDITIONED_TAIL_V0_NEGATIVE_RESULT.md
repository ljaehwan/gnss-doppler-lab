# Quality-Conditioned Tail v0 Negative Result

Date fixed: 2026-08-25

## Status

The experiment completed on frozen TEXBAT and OAKBAT PRN-local GRU scores, but
the tested quality state did **not** produce a useful detector improvement.  It
must not replace the frozen clean-calibrated binomial-tail baseline.

The valid conclusion is narrow: causal age since the start of a contiguous
*available score sequence* is not a sufficient proxy for receiver tracking
quality in these frozen score contracts.

## Hypothesis and detector

The experiment tested whether normal-only thresholds conditioned on tracking
maturity could discount acquisition/reacquisition anomalies without weakening
multi-satellite spoofing evidence.

For each PRN-local GRU score `e(i,t)`, the score sequence is split at timestamp
gaps greater than 0.75 s.  Causal continuity age is assigned to one of three
fixed states:

```text
age < 5 s
5 s <= age < 20 s
age >= 20 s
```

Normal-only q50/q70/q80 node thresholds are estimated independently in each
state.  At each receiver event, the number of PRNs exceeding each applicable
threshold is converted to exact binomial-tail surprise, the three surprises are
maximized, and the frozen causal EWMA recurrence is applied:

```text
state[t] = 0.75 * state[t-1] + 0.25 * surprise[t]
```

The matched control uses one global set of q50/q70/q80 node thresholds.  Both
detectors receive exactly the same frozen PRN-local scores, event grouping,
q99 event calibration, onset guards, and timing contract.

Important terminology limitation: score-continuity age begins after the GRU
warm-up and is inferred from score timestamps.  It is not hardware tracking
lock age.  The frozen score CSVs do not preserve `segment_index`,
`window_index`, lock indicators, or acquisition state across all TEXBAT
scenarios.

## Frozen campaigns

### TEXBAT pooled clean calibration

- Calibration: cleanStatic + cleanDynamic only.
- Evaluation: DS1-DS4; attack score CSVs were not opened before calibration was
  fixed.
- Clean results are calibration resubstitution checks, not held-clean evidence.
- The matched global implementation exactly recovered the frozen node
  thresholds and event threshold `4.169877716047041`.

| Scenario | Detector | Pre-FPR | Post-TPR | First score delay |
|---|---:|---:|---:|---:|
| DS1 | global / quality | 0 / 0 | 0.9573 / 0.9573 | 25.04 / 25.04 s |
| DS2 | global / quality | 0 / 0 | 1.0000 / 1.0000 | 9.37 / 9.87 s |
| DS3 | global / quality | 0 / 0 | 0.9741 / 0.9741 | 19.45 / 19.45 s |
| DS4 | global / quality | 0 / 0 | 0.7714 / 0.7714 | 14.04 / 14.04 s |

Quality conditioning produced no detection-rate improvement and delayed DS2 by
0.5 s.  The cleanDynamic resubstitution false-positive rate was 0.02176 for
both detectors.

### TEXBAT cleanStatic-to-cleanDynamic transfer

- Calibration: cleanStatic only.
- OOD normal control: cleanDynamic, never used for calibration.
- Evaluation: DS1-DS4 with the same cleanStatic-only thresholds.

| Metric | Global | Quality-conditioned |
|---|---:|---:|
| cleanDynamic OOD FPR | 0.9278 | 0.9003 |
| DS2 pre-FPR | 0.0710 | 0.0296 |
| DS3 pre-FPR | 0.0059 | 0.0000 |
| DS4 pre-FPR | 0.0119 | 0.0000 |

The lower attack-prefix false-positive rates do not rescue the detector:
90.03% of cleanDynamic windows still alarm.  The dominant failure is the
cleanStatic-to-cleanDynamic score-domain shift, not short-lived tracking-age
contamination.  DS3 post-TPR also decreased from 0.9841 to 0.9798 and its
three-consecutive delay increased by 4 s.

### OAKBAT calibration-to-held transfer

- Calibration: the frozen cleanStatic calibration partition only.
- Negative control: the disjoint held-clean partition.
- Evaluation: OS1-OS4 with onset 120 s and no scenario-prefix calibration.

| Scenario | Detector | Pre-FPR | Post-TPR |
|---|---:|---:|---:|
| OS1 | global / quality | 0.1106 / 0.1106 | 0.0287 / 0.0272 |
| OS2 | global / quality | 0.0337 / 0.0337 | 1.0000 / 1.0000 |
| OS3 | global / quality | 0.1250 / 0.1298 | 0.7550 / 0.7364 |
| OS4 | global / quality | 0.0433 / 0.0433 | 1.0000 / 1.0000 |

Both detectors produced zero held-clean flags, but the proposed variant did not
improve any attack scenario and slightly worsened OS1 and OS3.

## Why the hypothesis failed

1. Most attack-onset PRNs are already in the mature `age >= 20 s` state, where
   conditioned thresholds are close to the global thresholds.
2. Continuity age cannot identify a false lock that persists without a missing
   score or timestamp gap.
3. The TEXBAT static-to-dynamic problem is a broad score-distribution shift,
   not an isolated acquisition transient.
4. The portable frozen score contract lacks the receiver-state metadata needed
   to test the stronger quality-conditioned hypothesis across all scenarios.

## Research decision

Do not tune age cutoffs on DS/OS labels and do not present this variant as an
improvement.  More neural architectures or attack-specific thresholds are not
justified by this result.

The quality-aware direction should be reconsidered only after the scorer
preserves real receiver metadata (`channel`, `segment_index`, `window_index`,
`epoch_count`, explicit lock/reacquisition state) and an independent clean
corpus supplies enough genuine reacquisition/false-lock examples for
calibration and held-clean evaluation.

## Reproduction

```bash
.venv/bin/python scripts/eval_quality_conditioned_tail.py \
  --manifest configs/experiments/quality_conditioned_tail_v0.json \
  --out-dir artifacts/quality_conditioned_tail_v0

.venv/bin/pytest -q tests/test_quality_conditioned_tail.py
```

Machine-readable frozen summaries are under
`artifacts/quality_conditioned_tail_v0/`.

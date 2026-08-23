# TEXBAT-first spoofing-model design audit

## Outcome

This read-only design audit ends with
`NO_CREDIBLE_NEW_MODEL_AFTER_FAILURE_AUDIT`. It does not select or authorize
a detector. Two physically distinct residual hypotheses were scored. The
code/carrier/clock-closure candidate scored 73/100 and the NAV-content
coherence candidate scored 61/100; neither meets every mandatory requirement
and the frozen 75/100 selection threshold.

No detector was implemented or trained, no attack evaluation was run, and no
raw IQ payload was opened, stated, hashed, mapped, or sampled in this audit.
Tuni2025 SS-3, SS-5, SS-11, SS-12, and SS-13 remain unopened external
validation material. The audit inspected only Git objects, committed compact
artifacts, source code, and public literature.

## Scope and method

Base commit: `461eb4dc7bb794e719295daf028f6811658ba37f`.

The inventory covers 33 detector or detector-baseline records across 96 remote
research refs. Each row cross-checks source code with a preregistration or
configuration and a retained terminal verdict or metric where one exists.
Infrastructure-only TRACE/R2C and synthetic-simulator branches were indexed
but were not counted as models. Early code-only concepts are retained and
labeled as such rather than being assigned a terminal scientific verdict.
Of 88 SHA/path evidence links, 79 belong to remote-preserved objects and are
resolved by the fresh-clone verifier. Nine links belong to four explicitly
labeled local-only historical branches; their full SHA/path provenance is
preserved as a source-repository snapshot because unrelated local refs are
not transferred when this audit branch is pushed.

The central distinction is between:

1. implementation/provenance failures, which do not disprove a physical
   hypothesis;
2. dataset-limited results, which cannot support generalization;
3. controlled physical no-go results, in which preregistered positive,
   negative, destruction, dominance, or external-FPR gates failed.

The machine-readable evidence ledger is
`prior_experiment_inventory.json`; the flattened review table is
`prior_experiment_inventory.csv`. `failure_mechanism_map.md` removes
name-only duplicates, while `observable_coverage_matrix.csv` classifies the
21 requested observable families.

## TEXBAT task boundary

The prospective task is a causal receiver-observable detector. Attack truth,
scenario ID, filename, official onset, and PRN identity are not features.
Official onset is used only after decisions are frozen to calculate clean
false-alarm rate, first post-onset alarm delay, attack detection probability,
and recording-safe block-bootstrap confidence intervals.

TEXBAT DS1-DS4 and DS7-DS8 have already been exposed in this repository and
must not be described as blind. DS5 and DS6 have no committed payload-use
evidence. This audit itself read zero TEXBAT payload bytes.

## Candidate decision

`candidate_1.md` specifies a causal code/carrier/clock-closure monitor. It
is falsifiable and physically plausible, but closely overlaps established
clock-state, code/carrier, and Doppler-consistency literature. Its WCL novelty
risk and signal-specific Galileo transfer reduce it below the selection bar.

`candidate_2.md` specifies a causal NAV-content coherence monitor. It is
independent from the repository's failed analog-observable models, but a
coherent replay can preserve its content, so it fails physical
identifiability for the TEXBAT primary task.

No third candidate survived de-duplication. Repackaging correlation-peak
morphology, common-emitter structure, tracking innovations, or raw-IQ texture
inside another encoder would repeat already tested physical quantities.

## Reproduction

```bash
python3 scripts/verify_texbat_first_spoofing_model_design_audit.py
python3 -m unittest tests.test_texbat_first_spoofing_model_design_audit
```

The verifier checks evidence-object syntax and availability where present,
inventory/CSV agreement, terminal-verdict semantics, score arithmetic,
causality and preservation contracts, documentation consistency, and every
artifact hash. It performs no dataset access.

# B0-CS Stage-0 Static Revalidation

## Scope and freeze boundary

B0-CS asks whether the shared PRN-local B0 predictor can retain its normal
peak-prediction signal while replacing the receiver-level independent-PRN
binomial tail with dependence-aware set aggregation, clean receiver-block
calibration, and sequential evidence.  It does not add a geometry, relation,
CAF, IQ-power, or learned receiver classifier.

The machine-readable preregistration is
`artifacts/b0_cs_stage0_static/preregistration.json`.  Attack IQ, attack node
tables, and attack score outputs must not be opened by the B0-CS stage runner
until it has verified the preregistration hash and a completed clean-only freeze
record.  Public scenario descriptions and code/document surveys are not attack
outcomes.  No attack label or result may select a feature, bin, block, betting
function, threshold, predictor checkpoint, or control.

## Required pre-implementation survey

### Historical B0 and binomial-tail receiver gate

`scripts/train_prn_node_gru.py` defines a shared `PrnLocalGRU`: a two-layer
128-dimensional feed-forward encoder, one unidirectional GRU with hidden size
128, and a 128-dimensional head.  The frozen tap-only setting uses 12 causal
epochs, dropout 0.05, AdamW with learning rate 0.001 and weight decay 0.0001,
batch size 256, seed 11, and the nine prompt-relative Method-A tap magnitudes in
the order `E4,E3,E2,E,P,L,L2,L3,L4`.  PRN identity is grouping metadata, not an
input.  The historical trainer used a PRN holdout when only one run was present;
that is not the chronological paper split required here.

`scripts/score_texbat_prn_node_gru.py` resets causal history per run/PRN, scores
the standardized next-window error by scalar RMSE, and historically exposes
max/mean/median/top-three receiver summaries.  Its frozen score timestamp is
the 1 s window start and becomes available at window end.

`scripts/eval_btail_support_gate.py` and
`docs/BINOMIAL_TAIL_GATE_BASELINE.md` define Historical-B0's exact receiver
gate.  CleanStatic plus cleanDynamic supplied q50/q70/q80 per-PRN thresholds,
the gate assumes nominal independent Bernoulli exceedances, takes the maximum
negative-log exact binomial tail, and applies
`state[t]=0.75*state[t-1]+0.25*raw[t]`.  Its q99 threshold is also from the two
clean recordings.  The tracked checkpoint SHA-256 is
`f171bf0b2084e617c15ab6af72ef930539a4b8fddb120b5aa8f43a6339c96a6b`.
The related tests pin the checkpoint/features/calibration bytes, exact tail,
event grouping, causal recurrence, run reset, timing, and clean/onset metric
semantics.

Historical-B0 is reproduced only as the repository reference.  It is not the
Paper-B0 result because its predictor split is by PRN and its detector and
threshold used cleanDynamic.

### CMTE, CMTE-A2, and earlier sequential conformal code

The comparison implementations live in clean, read-only research worktrees,
not at this branch's base commit:

- CMTE commit `d19115e48f8859330ae05bbab59c0ffd5f3d2004` implements signed
  nine-dimensional standardized innovations, shrinkage Mahalanobis
  nonconformity, finite-sample conformal ranks, a fixed mixture of power
  e-values, PRN-set summaries, and experimental sequential capitals/e-CUSUM.
- CMTE-A2 commit `1537c958f7e82a83a32cd40a7d5c22a1328b3cf8` uses the same signed
  multivariate innovation and shrinkage Mahalanobis score, but its primary
  receiver score is the non-sequential mean `-log(p)` with a clean threshold.
  Its preregistration expressly forbids mixture e-values and sequential scores.
- The older hybrid-conformal commit
  `f361690026db74f5ef4e6d34d082e67c2a7bd59f` supplies tie-conservative
  upper-tail conformal ranks, permutation-invariant masked set summaries, and a
  causal leaky CUSUM.  It is graph/relational work and is a survey reference,
  not a component imported by B0-CS.

B0-CS differs from CMTE-A2 in five material ways.  It uses the existing scalar
B0 RMSE rather than a signed multivariate innovation or covariance model.  It
converts each scalar residual to an e-value and uses the arithmetic mean across
the currently tracked PRN set, with no PRN identity stratum.  It treats
cross-PRN dependence separately from temporal dependence.  It calibrates
receiver-level non-overlapping block maxima and then applies the prespecified
e-CUSUM.  Finally, it conditions calibration only on lagged C/N0 and tracked
count nuisance context learned from clean data.

### Splits, timing, and provenance

The historical B0 trainer's PRN holdout and CMTE's fixed time roles show why a
new split audit is necessary.  B0-CS uses a chronological 50/15/20/15 split of
cleanStatic target epochs with three 6 s guards and resets causal histories at
every role, physical recording, segment, channel, and cadence gap.  Target
epochs, source sample counters, and derived byte intervals are audited for
disjointness.  Calibration and holdout never enter predictor fitting; holdout
is opened only after the clean checkpoint, calibrator, and thresholds freeze.

The authenticated clean source is
`/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-clean-graph-input-v2/exports/cleanStatic.npz`
with SHA-256
`fcd1d378c28e79fe4a550b65fc1208cde3c8fb334db11406a07fed4d90fba237`.
Its manifest reports raw causal epochs, complex nine-tap IQ, `cn0_db_hz`,
sample counters, PRN/channel/segment identities, 25 MHz sampling, and no attack
input access.  The downstream CMTE-A2 node conversion is reconstructed
equivalence and explicitly not confirmatory-eligible; B0-CS therefore rebuilds
its own nodes from the authenticated NPZ and preserves sample/C/N0 lineage.

Official timeline values are metadata, never fit inputs.  The frozen mapping is
DS1 125.0 s, DS2 110.1 s, DS3 118.9 s with pull-off 195.0 s, DS4 113.8 s with
pull-off 225.0 s, and DS7/DS8 injection 110.0 s with time-push 150.0 s.  DS4 is
`LIMITED` when 225 s is absent.  DS7 and DS8 form one family and are never
counted as independent confirmations.

## B0-CS formulas and validity boundary

For PRN `i`, Paper-B0 supplies
`a_i(t)=sqrt(mean_j((x_ij(t)-xhat_ij(t))^2))`.  The calibration tail rank is
`p_i(t)=(1+# {a_cal >= a_i(t)})/(M+1)`, so its minimum is `1/(M+1)`.  The fixed
bet is `e_i(t)=0.5*p_i(t)^(-0.5)`.  Epochs with fewer than four valid PRNs are
suppressed.  Otherwise `E(t)=mean_i(e_i(t))` and
`S_set(t)=log(E(t)+1e-12)`.

The arithmetic mean preserves the e-value expectation bound without requiring
PRN independence when the component evidence is valid.  It does not eliminate
finite-sample concerns from time dependence, estimated nuisance strata, or
nonstationarity.  B0-CS therefore makes no distribution-free claim under
arbitrary temporal dependence and is labeled `EMPIRICALLY_BLOCK_CALIBRATED`.

For non-overlapping receiver blocks, `B_b=max_t S_set(t)`.  Clean calibration
gives the same inclusive upper-tail block p-value and fixed power e-value.  The
sequential recurrence is `C_b=max(1,C_(b-1))*e_b`, starts at one, resets per
physical recording, and alarms at `C_b>=100`.  Conditional e-value assumptions
are not asserted unless separately verified; empirical block FPR and normal
average run length are primary validity diagnostics.

## Claim limits

TEXBAT is developmental.  Deployment-level FPR requires a source-distinct
static normal recording without recalibration.  Missing lineage, C/N0,
sample-overlap mappings, scenario duration, raw-IQ recalculation, or statistical
support produces a structured `UNAVAILABLE` or `LIMITED` record, never an
imputed result.  Attack outcomes cannot trigger retuning or a replacement
model.

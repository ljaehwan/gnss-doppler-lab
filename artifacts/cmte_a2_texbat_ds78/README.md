# CMTE-A2 TEXBAT DS7/DS8 validation

## Final status

**PRIMARY INVALID / NO-GO.**

The preregistered one-shot result is preserved exactly, but post-run validation found that channel-specific floating availability timestamps were grouped by exact equality. Consequently, every stored “epoch” contained exactly one PRN (`tracked_prn_count = 1`) in DS1, DS2, DS3, DS4, DS7, and DS8. The preregistered primary score therefore did **not** implement the intended multi-PRN epoch mean `mean_i[-log(p_i)]`. Primary metrics and moving-block confidence intervals are retained for audit only and must not be used as confirmatory evidence.

No post-hoc threshold change, aggregation change, or holdout rerun was used to replace the primary result.

## Frozen chronology

- Base branch commit: `d19115e48f8859330ae05bbab59c0ffd5f3d2004`
- Preregistration commit: `e7cb2e5822923a129d72c475706f87721ddd8104`
- Frozen execution/source commit: `71d00f310b6152868b2e02df2ca955cfecd43eb3`
- Result-bearing artifact commit: `92f78f4c49bee98ce840e84c74f8387042b401cb`
- Trust-anchor SHA-256: `bc9f76fc26fb48b0e974f626cf386758a375eadb254966ad216502db01b21430`
- One-shot ledger status: `completed`
- Result checksum-manifest SHA-256 recorded by ledger: `dbcfde6f7f607652ccc526a869b59074d327aa3f4ff1ec85b623b543a8f7699d`

DS7/DS8 scoring occurred only after the preregistration, code/state/input freeze, and pre-campaign test attestation.

## Chronological B0 evidence

A new shared PRN-local GRU was trained with the preregistered fixed settings:

- 9 prompt-relative magnitude taps: `E4,E3,E2,E,P,L,L2,L3,L4`
- causal history: 12 windows
- hidden size: 128; dropout: 0.05
- AdamW, learning rate 0.001, weight decay 0.0001
- 25 epochs; no early stopping; seed 11
- no PRN identity feature; shared weights

Role audit:

- B0/scaler/model-selection prefix: window end `<=239.84 s`
- conformal calibration: approximately `250.04–289.84 s`, `Q_cal n=669`
- threshold calibration: approximately `300.04–329.x s`, 460 residual rows
- independent clean test: approximately `340.04–480.12 s`, 2,714 residual rows
- role overlap: false
- attack scenarios excluded from every fitting step
- history reset at role/split/recording/segment/channel/cadence-gap boundaries

The deterministic rerun reproduced the same checkpoint:

`44bc85320e4fdf6ffdff3b4c12941a1f90d39832d4b13707ecb4a0317f936fa0`

Historical B0 gate semantics were checked against the existing evaluator and golden DS1 events; maximum absolute error was `3.55e-15`, with identical strict alarms.

## Frozen CMTE-A2 definition

For signed 9D residual `r_i(t)=x_i(t)-x_hat_i(t)`:

```text
q_i = (r_i-mu)^T Sigma_reg^-1 (r_i-mu)
p_i = (1 + count(Q_cal >= q_i)) / (|Q_cal| + 1)
S_A2(t) = mean_i[-log(p_i(t))]
```

- inclusive ties and finite-sample `+1`
- shrinkage covariance with epsilon regularization
- primary operating point: q99.5, NumPy `higher`
- frozen threshold: `4.204692619390966`
- no e-value, e-CUSUM, restart capital, sequential Full score, online attack normalization, or attack-based tuning

## Producer provenance

Primary common converter was used for cleanStatic, DS1–DS3, DS7, and newly prepared DS8:

- magnitude: `hypot(I,Q)`
- prompt-relative normalization, then 1.0 s window mean
- stride: 0.5 s
- converter SHA-256: `5e6db2ef2a07b01a5753d2ac3729df0da47c95821ce61fc4118b634efb671f5a`
- wrapper SHA-256: `a5318f9871b66f1d8273b66ce93e4f907c53dad980169a11d81559974bab6ef2`

DS8 was regenerated from canonical raw IQ with the pinned receiver/exporter:

- raw SHA-256: `1614d8de6fc8ebc3429def6e9505050c08a3ee8da69c11ecc27a98305f735d78`
- complex NPZ rows: 464,692; shape `[N,9,2]`
- complex NPZ SHA-256: `d1973fa150b7b4e7359df4827f36ce60289f206e9db11c1ac2bc1fd33a0df533`
- node rows: 8,645

DS4 used a verified historical node artifact and is explicitly a mixed-producer development sensitivity, not primary same-producer evidence.

## Sealed primary metrics — audit only

These values are retained exactly but are invalid as multi-PRN CMTE-A2 evidence because every aggregated row had `N=1`.

### Development/reproduction

- DS1: AUC 0.9479, PR-AUC 0.9906, stable-pre FPR 1.9685%, post detection 50.42%, first delay 5.12 s
- DS2: AUC 0.9792, PR-AUC 0.9961, stable-pre FPR 0.3282%, post detection 36.54%, first delay 15.04 s
- DS3: AUC 0.9822, PR-AUC 0.9981, stable-pre FPR 0.4405%, post detection 57.56%, first delay 34.13 s
- DS4 sensitivity: AUC 0.7230, PR-AUC 0.6512, stable-pre FPR 0%, post detection 2.74%, first delay 14.46 s

### Confirmatory holdout

- DS7: AUC 0.8273, PR-AUC 0.9314, stable-pre FPR 0.7979%, post detection 44.16%, persistent detection 58.22%, first delay 77.43 s
- DS8: AUC 0.9300, PR-AUC 0.9882, stable-pre FPR 0.0915%, post detection 74.81%, persistent detection 84.09%, first delay 12.49 s

Independent clean FPR for the sealed primary CMTE-A2 score was **1.5844%**, above the preregistered required ceiling of 1.5%.

### B0-Exact holdout comparator

- clean FPR: 0.7001%
- DS7: stable-pre FPR 2.9255%, post detection 60.26%, persistent 79.05%, delay 22.79 s
- DS8: stable-pre FPR 0.0915%, post detection 81.71%, persistent 91.73%, delay 16.49 s

Even before invalidation, CMTE-A2 did not meet the preregistered similar-clean-FPR improvement criterion against B0-Exact.

## Bootstrap confidence intervals

The sealed campaign generated 2,000-replicate, >=10 s moving-block intervals. Examples:

- DS7 ROC-AUC: `[0.7656, 0.9373]`
- DS7 post detection: `[0.3213, 0.6200]`
- DS7 persistent detection: `[0.3912, 0.7044]`
- DS8 ROC-AUC: `[0.9235, 0.9652]`
- DS8 post detection: `[0.7421, 0.8328]`
- DS8 persistent detection: `[0.7948, 0.8789]`

However, because pseudo-epochs were formed per PRN rather than per physical epoch, these CIs are also **invalid for the intended multi-PRN detector** and are audit-only.

## Success criteria

1. Independent clean FPR <=1.5%: **FAIL** (`1.5844%`)
2. DS7 and DS8 stable-pre FPR <=5%: sealed numbers pass, but primary invalid
3. No confirmatory stable-pre FPR >=20%: sealed numbers pass, but primary invalid
4. Improve over B0-Exact at similar clean FPR: **FAIL**
5. No catastrophic degradation on the other holdout: **FAIL under preregistered combined decision**
6. PRN-count diagnostic: files exist, but all epoch counts were `N=1`; intended variable-cardinality multi-PRN behavior was not validated

## Claims

Possible claims:

- chronological B0 training/calibration separation and deterministic checkpoint reproduction were implemented and audited;
- DS8 same-producer raw-IQ preparation was completed with pinned hashes;
- preregistration, frozen trust anchor, and one-shot holdout chronology were enforced;
- a concrete timestamp-aggregation failure mode was identified and preserved transparently.

Not permissible:

- that the sealed primary validates CMTE-A2 multi-PRN epoch evidence;
- that DS7/DS8 confirmatory performance is established;
- that moving-block CIs describe the intended detector;
- that CMTE-A2 improves B0-Exact;
- that target clean FPR control was achieved;
- that DS7/DS8 were project-wide pristine holdouts.

## Final decision

**PRIMARY INVALID / NO-GO.** A corrected 0.5 s physical-epoch aggregation would change both aggregation and threshold calibration after holdout exposure. It was therefore not substituted for the preregistered primary in this result package.

# Earlier and non-WCL-CGC research lines

These lines remain valuable but are outside the current WCL CGC manuscript.
Their scores, data splits, learned checkpoints, and thresholds must not be mixed
with the final CGC endpoint.

## Learned peak-shape and support detectors

- B0 normal-only PRN-local GRU and exact binomial-tail support gate:
  [`../../docs/BINOMIAL_TAIL_GATE_BASELINE.md`](../../docs/BINOMIAL_TAIL_GATE_BASELINE.md)
- graph, relational, multi-view, and other learned peak-shape experiments;
- the uncommitted `WCL_THREE_CONDITION_AUDIT_V1` and associated GRU/OpenIF/tap
  scripts remain in a review queue, not in the CGC paper.

These approaches learn or score channel anomalies. CGC instead receives a
first-stage alert and classifies whether signed delays obey common LOS geometry.

## Raw-IQ and receiver-fingerprint work

- M1 raw-IQ noise/fingerprint continuity:
  [`../../docs/RAW_IQ_NOISE_CONTINUITY_SECOND_METHOD.md`](../../docs/RAW_IQ_NOISE_CONTINUITY_SECOND_METHOD.md)
- receiver-state tail and quality-conditioned tail models;
- peak-floor temporal and contrastive models.

## Doppler and static-reference geometry

- early carrier-Doppler anomaly studies;
- static pre-attack pseudorange-reference CGC;
- OAKBAT-held reference experiments;
- Tu-style Doppler observability comparisons.

They may motivate the current work, but they use a different observable or
trusted reference and are not the pre-resolved complex-correlator CGC endpoint.

## GCMR and alternative geometry models

Files prefixed by `gcmr_` and related TEXBAT experiments belong to a separate
geometry/model line. They are not the clock-centered Partial-F classifier used
by the WCL manuscript.

## Superseded CGC development

Earlier RF challenge, locked-test, transfer-sweep, observability-anchor,
real-detection, code/carrier pilot, and sequential TEXBAT experiments led to the
final design. See
[`../wcl-cgc/experiments/`](../wcl-cgc/experiments/) for their scientific role.

## Known pre-existing test debt

The tracked Peak-Floor CPC scripts/tests reference an untracked and historically
missing `scripts/train_peak_floor_temporal_autoencoder.py`. The full suite
therefore currently reports six legacy failures. The WCL CGC focused suite and
manifest audit pass. This debt predates the CGC classification and must be fixed
in a separate legacy-maintenance change rather than silently recreated during
paper promotion.

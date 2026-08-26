# CGC RF locked test v1: terminal input-support failure

Date: 2026-08-26
Status: **NOT SUPPORTED**

## Outcome

The sealed held-out test did not reach CGC performance analysis.

Test pair 012 produced only three authentic startup line-of-sight GPS PRNs.
The protocol required at least eight PRNs in every pair. The locked runner
therefore stopped before generating pair-012 multipath RF and before computing
or emitting any pair-level CGC separation or AUC.

Under the preregistered all-gates-must-pass rule, the final decision is
NOT_SUPPORTED.

This is not evidence that the clock-centered CGC score reversed direction or
failed on an evaluated test pair. No test-pair CGC performance metric was
emitted. It is evidence that this particular held-out campaign could not
support the required three-pair physical comparison.

## Frozen release provenance

The test was released only after the protocol and runner were committed.

- protocol/config commit:
  a5eb634f1f0f0d693caac5b5710b6b0315853641
- runner commit:
  acc8dac0284e7e6a147ffe09bee36b289b82387a
- config SHA-256:
  f3792b00993d40b028218ae0024d855a294f6e71cc0f12caab5c268bd5404c95
- protocol SHA-256:
  be93580ecd60d90f8758ea833be0cff2ca4a11aeb609d678b5388976cd3ae655
- runner SHA-256:
  3a6e29d45425168395e64bf6c479de2c8b4351037efce7805f86d8db952736d8

The release guard verified that all three files were tracked, committed, and
clean before source generation began.

## Execution trace

All three frozen normal/spoof source pairs were generated under the pinned
signal law. Their authentic startup LOS support was then inspected by the
same multipath receiver path used in train replication.

| Pair | Motion | Startup LOS PRNs | Source complete | Receiver runtime complete |
|---|---|---:|---:|---:|
| 010 | Static | 13 | yes | yes |
| 011 | Straight | 12 | yes | yes |
| 012 | Parallel sweep | **3** | yes | no |

Pairs 010 and 011 completed multipath/spoof GNSS-SDR processing. Pair 012
stopped at the frozen support check with:

    ValueError: too few startup LOS PRNs for pv1-pair-012

The pair-012 authentic simulator itself exited normally and its startup table
contained PRNs 08, 21, and 26. Therefore this was not a process crash. The
generated pair simply lacked the preregistered satellite support required for
an eight-PRN geometry fit.

No CGC analysis was run on pairs 010 or 011 after the stop.

## Gate decision

| Gate | Required | Observed | Result |
|---|---:|---:|---:|
| Complete receiver pairs | 3 | 2 | **FAIL** |
| Minimum startup LOS PRNs per pair | 8 | 3 | **FAIL** |
| Positive centered separations | 3/3 | not evaluated | NOT EVALUATED |
| Pair-block AUC | at least 0.80 | not evaluated | NOT EVALUATED |
| Centered improvement over legacy | at least 2/3 | not evaluated | NOT EVALUATED |
| Comparison bins per scenario/pair | at least 5 | not evaluated | NOT EVALUATED |

Every gate was required to pass. The two observed failures are sufficient for
the terminal NOT_SUPPORTED decision.

## Why the run was not resumed

The minimum eight-PRN requirement was frozen before test access. Lowering it,
moving the 012 time or location, replacing the navigation input, excluding
012, or completing only 010--011 would change the test after observing a
held-out input.

The protocol permits no post-release tuning or replacement test. This was an
input-support gate failure, not an independently verifiable infrastructure
failure. Consequently no resume, repair run, or substitute performance result
is authorized.

Pairs 010--012 are now a spent test partition. They may not be presented later
as untouched evidence.

## Scientific interpretation

The earlier train result remains a positive train-only result: the
clock-centered residual ranked coherent carry-off spoofing above the selected
satellite-specific multipath family on pairs 002--006.

The final independent claim, however, is not established. The current evidence
supports neither a held-out multipath-versus-spoof performance number nor an
operational detector claim.

This outcome exposes a campaign-design issue that must be solved independently
of the score: satellite/navigation support must be verified before assigning
fresh pairs to a locked split. Such verification should be a data-availability
constraint, not a model-dependent filter.

A defensible next campaign should:

1. define a new set of times, locations, motions, and navigation inputs;
2. verify adequate LOS support before random/fixed split assignment;
3. preregister a fresh train/test roster and all gates;
4. carry the CGC candidate forward unchanged if the aim is an independent
   retry; and
5. never use 010--012 as untouched validation evidence.

## Artifacts

Terminal artifact:

    artifacts/cgc_rf_locked_test_v1/terminal_failure.json

SHA-256:

    444c13a9f8991c90c7cc66a0ad7c6d8316dc9a1dca3fe2c7ecee987bc4d042ba

Source-generation summary SHA-256:

    71eda16d37939aeb2f506fd0f8f465e203cb4d998ad19ca6eaa471c7d7de67ef

Release-state SHA-256:

    0ad10e51be77419874ca14e3e585f5f4a341a47f9183130d21f792d74b5fca15

SSD archive:

    /home/ubuntu/ssd_data/gnss-early-detection/artifacts/cgc-rf-locked-test-v1-terminal

The archive contains 179 regular files totaling 33,241,987,312 bytes. Full
checksum dry-runs reported no differences, and the SSD terminal artifact hash
matches the local hash.

## Claim boundary

This is the terminal outcome of one sealed simulated receiver-RF release. It
does not provide a held-out CGC separation, pair-block AUC, transferable alarm
threshold, field false-alarm estimate, TEXBAT result, operational result, or
WCL generalization claim.

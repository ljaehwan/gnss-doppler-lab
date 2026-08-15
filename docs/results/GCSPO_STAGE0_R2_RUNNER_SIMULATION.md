# GCSPO Stage-0 R2 runner simulation

This branch contains a runner adapter for the frozen GCSPO Stage-0 static TEXBAT
physics-hypothesis test.  The adapter changes execution granularity and records
exact-support mismatches; it does not change features, normal modelling, state
equations, scores, thresholds, regularization, timelines, or frozen randomness.

Large outputs are written only to:

`/home/ubuntu/ssd_data/gnss-early-detection/artifacts/gcspo_stage0_r2_runner_simulation/`

## Reproduction

Each scientific phase is submitted separately from the task worktree using:

```bash
/home/ubuntu/projects/gnss-doppler-lab/.venv/bin/python \
  /home/ubuntu/projects/gnss-doppler-runner-minimal-v0/scripts/research_run.py \
  --repo /home/ubuntu/projects/gnss-doppler-gcspo-stage0-r2-runner-simulation \
  submit --name <required-run-name> -- \
  /home/ubuntu/projects/gnss-doppler-lab/.venv/bin/python \
  scripts/run_gcspo_stage0_r2.py --phase <phase>
```

The authoritative run IDs, commands, terminal states, logs, and phase artifacts
are recorded in `run_inventory.json` under the SSD result root.  Failed attempts
remain in its `attempts` array and in the append-only runner registry.

## Source limitations

DS3 and DS7 have byte-inventoried receiver tracking and geometry inputs.  DS4
has no authenticated receiver-level signed tracking/geometry bundle (legacy
derived score files are not inputs).  DS8 has receiver tracking but lacks the
authenticated observables required for geometry, so geometry is not estimated
or generated.  The final detector and physical-hypothesis verdicts therefore
remain distinct and may be incomplete even when the executable core scenarios
finish successfully.

## Implementation-only repairs

- The frozen config is restored byte-for-byte after proving semantic JSON
  equality; the first failed attempt exposed whitespace-only serialization drift.
- A5 uses one CUDA process because PyTorch forbids CUDA reinitialization in
  forked children.  Completed clean outputs are preserved across the retry.
- B0/Full comparisons retain only complete identical epoch+PRN+window support,
  while every Full-only, B0-only, and support-mismatch event is inventoried.

## Result

All seven required latest runner phases succeeded.  Failed/retried attempts are
preserved in the runner registry and `run_inventory.json`.

- Verdict: `EVALUATION_INCOMPLETE_SOURCE_OR_SUPPORT`; neural Stage-1 is not allowed.
- cleanStatic Full FPR: q99 0.07173, q99.5 0.06751 (both recomputed).
- DS3 Full: ROC-AUC 0.8268, pre-onset FPR 0.1489, attack detection 0.6220.
- DS7 Full: ROC-AUC 0.5991, replay excluded from external FPR, attack detection 0.2943.
- B0 exact support: DS3 643/907 Full windows; DS7 396/533.  DS3 transition is
  explicitly `UNAVAILABLE_ON_COMMON_SUPPORT`.
- Full does not significantly beat A2 or A5 in the paired 10-second bootstrap.
- LOS shuffle reduces Full score in DS3 and DS7, but the frozen primary temporal
  desynchronization control does not.
- Physical controls fail: empirical noise, one-PRN disturbance, and PRN drop each
  reach 17 consecutive alarms and 0.7647 maximum persistent-alarm ratio.
- The shared-state magnitude changes are small and inconsistent across onset and
  pull-off phases.

This frozen detector is not worth continuing as the current paper model.  The
LOS-shuffle geometry diagnostic may be retained only for a separately
preregistered physics study after authenticated DS4/DS8 inputs are available.

## Delivery status

The initial result was committed locally as `2fe4c4d3f72ad4d071287b741f92bcb8d31e37c0`.
The initial push attempt failed because the environment had no GitHub HTTPS
credentials and no `gh` command; that historical blocker remains recorded in
`infrastructure_blocker.json` under the SSD artifact root.

Later delivery succeeded after GitHub HTTPS credentials were configured for the
research delivery environment.  The current remote branch is
`research/gcspo-stage0-r2-runner-simulation`; before adding the evidence bundle,
local and remote both pointed at
`2912d0f657151d84131eb0f31b5a6b613dee610f` with ahead/behind `0/0`.

Git delivery status is tracked separately from the scientific verdict in
`artifacts/gcspo_stage0_r2_runner_simulation_evidence/delivery_status.json`.
The evidence bundle commit SHA is reported after commit creation; the scientific
results and detector verdict above are not changed by delivery status updates.

## Verifiable evidence bundle

A compact evidence bundle is stored under
`artifacts/gcspo_stage0_r2_runner_simulation_evidence/`.  It preserves the SSD
manifest identity, copies or deterministically gzips small/essential evidence
files, includes seven-phase runner terminal evidence, records provenance caveats,
and provides `scripts/verify_gcspo_r2_evidence.py` plus pytest coverage for
independent recomputation from CSV/JSON evidence.  The bundle verification
judgement is `EVIDENCE_VERIFIED`; this means the evidence is internally
verifiable, not that the detector is a GO.

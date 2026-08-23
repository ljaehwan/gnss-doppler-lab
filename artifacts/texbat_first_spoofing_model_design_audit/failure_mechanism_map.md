# Failure-mechanism and de-duplication map

This map is based on the code/artifact evidence listed in
`prior_experiment_inventory.json`, not on model names.

| Physical family | Models and baselines | Terminal evidence | Mechanism-level conclusion |
|---|---|---|---|
| Prompt-normalized correlation morphology and prediction residuals | B0, B0-CS, CMTE-A2, CLIF-IP, AMCF, CRISP, conditional GRU, AI morphology quorum | B0 remains developmental; B0-CS pivoted; CMTE was primary-invalid; CLIF-IP/AMCF/CRISP failed controlled gates | A new encoder or sequential wrapper does not create a new observable. Reusing these taps has high repeat risk. |
| Sparse or decomposed CAF/peak structure | MOSAIC, PG-SCC, ACAF-NF, MIRAGE, peak-floor DIC/PFCP | MOSAIC, PG-SCC and MIRAGE are no-go; ACAF-NF is source/support-inconclusive | Sparse coordinates, multipath decomposition, and multiscale peak geometry did not establish incremental physical identifiability. |
| Tracking-loop response and innovations | GCMR, GCMR-PI, GCSPO, MCTD, CRID, TRCD, DA-PFRT | MCTD and CRID controls failed; GCSPO/GCMR-PI were incomplete; TRCD/DA-PFRT are code-only | Reweighting DLL/PLL/Doppler residuals or changing the temporal learner would repeat the same physical channel. |
| Common origin, common emitter, shared onset | CORA, CINDER, Q-COMET, BITPROBE, M1 | CORA/CINDER/Q-COMET physical gates failed; BITPROBE inference was inconclusive and relaxed gates found no robust signal; M1 remained shortcut-limited | Single-front-end RF fingerprints and cross-PRN synchrony are confounded by receiver gain, propagation, and coherent source construction. |
| PRN-panel aggregation | fixed9 baselines, PG-SCC, Q-SET, AMCF | PG-SCC showed no sparse-panel advantage; Q-SET stopped at receiver feasibility; AMCF failed | Dynamic masking is necessary engineering, not a spoofing observable or independent scientific hypothesis. |
| Power/lock/reacquisition | M1, NC-TOPI, Q-SET receiver feasibility | NC-TOPI was IQ-shortcut dominated; Q-SET lacked sufficient clean receiver support | These variables are nuisance/audit channels and cannot be promoted to a primary score. |
| Code/carrier/Doppler/PVT closure | GCMR/GCSPO partially; Candidate 1 directly | Repository evidence is incomplete for full clock closure, but direct published methods exist | Physically plausible and not a repository duplicate at equation level, yet too close to prior art for the required WCL novelty. |
| NAV content coherence | BITPROBE used bit edges; no committed content detector | Candidate 2 remains unvalidated | Independent observable, but coherent replay preserves parity/TOW/ephemeris; it cannot identify the primary threat class. |

## Final causal chain

The repository has already exercised the main single-receiver analog
observable families: correlation shape, complex phase/amplitude, delay and
Doppler, loop response, common-origin structure, temporal onset, RF texture,
and PRN aggregation. Their controlled failures cannot be repaired by naming a
new neural architecture.

Only solution-level closure and decoded-message consistency remain materially
different. The former fails the novelty/transfer score and the latter fails
physical identifiability. Therefore the correct output is no model selection,
not a third renamed peak or residual detector.

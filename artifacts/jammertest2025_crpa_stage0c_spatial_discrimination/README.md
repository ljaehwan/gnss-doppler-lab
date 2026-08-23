# Jammertest 2025 CRPA Stage-0C label-only design freeze

Status: `LABEL_ONLY_DESIGN_FREEZE_PRE_FEATURE`.

This commit freezes the Area-1 30/40 dBm primary three-fold blocked design, one-block test guards, deterministic SHA-256 balancing, sensitivity designs, numerical invariance grid, M0–M3 features, classifier, threshold, bootstrap, and verdict gates before any IQ feature is opened. Primary and sensitivity A are label-feasible. Sensitivity B is label-infeasible because the 25 dBm negative cell cannot retain block-disjoint train and test support after the required guard.

No IQ bytes or Stage-0B class-conditional spatial results were used to select this design. The execution script remains a fail-closed skeleton until this commit is pushed.

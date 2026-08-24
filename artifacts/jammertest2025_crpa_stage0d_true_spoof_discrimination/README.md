# Jammertest 2025 CRPA Stage-0D true-Spoof discrimination

Status: `LABEL_ONLY_DESIGN_FREEZE_PRE_IQ`.

This first commit freezes the exact Area 1, 40 dBm, Spoof-versus-Prn five-fold complete-OOF design, same-class guard semantics, received-power matching algorithm, M0/M1/M2/M2R/M3 definitions, classifier, controls, seed, and verdict set before opening IQ data. Execution remains fail-closed until this commit is pushed.

The explicit block lists contain 5 Spoof and 6 Prn blocks (11 class blocks). They are treated as authoritative over the later prose reference to 10 blocks.

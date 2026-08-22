# BITPROBE-GNSS Stage-0A clean NAV-edge operator audit

This artifact evaluates only cleanStatic NAV-bit edge operator identifiability.
It does not evaluate attacks, validate a spoofing detector, or localize an
observed common operator to a transmitter. For one receiver,
h * sum(s_i) = sum(h * s_i); receiver-side common/nonlinear effects remain
a source-localization confound.

The method, edge support, nuisance suite, synthetic controls, statistics, and
gates were preregistered and pushed before any clean raw edge extraction.
Large edge tensors remain on the bound SSD output root; Git contains compact
summaries and hashes only.

## Final disposition

The locked analysis completed with full support and initially returned BITPROBE_STAGE0A_EDGE_OPERATOR_NOT_IDENTIFIABLE. Post-result contract review found that the frozen bootstrap mixed the preregistered chronological halves and that the PRN-permutation comparison statistic was not shape-identical to the observed statistic. The original output is preserved, no executable or gate was changed, and the terminal verdict is therefore fail-closed INCONCLUSIVE_BITPROBE_STAGE0A_EXECUTION_OR_PROVENANCE. Stage-0B is not authorized.

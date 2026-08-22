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

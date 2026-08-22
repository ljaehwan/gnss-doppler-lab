# BITPROBE-GNSS Stage-0A R0a inference-contract repair

This is an inference implementation repair only. It does not reopen raw IQ,
TRACE, or attack data and does not regenerate edges. The edge estimator,
features, normalization, point statistics, nuisance suite, synthetic controls,
and every scientific gate remain frozen.

The repair isolates chronological halves before block bootstrap and applies a
shape-identical exact second-half PRN-label permutation. Synthetic level
detectability is not authoritative; the reproduced synthetic result measures
common-versus-separate geometry only. A single receiver cannot localize a
common operator to a transmitter.

Stage-0B authorization is determined only by the terminal R0a verdict after
the pushed execution freeze is replayed against the bound frozen tensor.

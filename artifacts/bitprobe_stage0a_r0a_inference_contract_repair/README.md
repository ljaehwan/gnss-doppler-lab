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

## Final disposition

The two permitted inference repairs reproduced twice byte-identically from the frozen tensor. Point estimates, support, nuisance values, synthetic values, PRNs, edge counts, and source binding were unchanged. A newly confirmed mismatch remains in the frozen flip-specificity test: the preregistration specifies contiguous 10-edge blocks, while the frozen code shuffles individual pooled edges and repartitions them equally. R0a does not repair that third issue. The terminal substantive verdict is therefore INCONCLUSIVE_BITPROBE_STAGE0A_R0A_INFERENCE_REPAIR, repair_status=FAIL, and Stage-0B is not authorized.

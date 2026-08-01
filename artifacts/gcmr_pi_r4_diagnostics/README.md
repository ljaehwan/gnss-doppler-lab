# r4 frozen diagnostics

Read-only analysis of frozen r3; no retraining or threshold tuning on attacks. Warmup uses 30–110 s pre, excludes 110–130 s, and uses >=130 s post. Full uses frozen thresholds unchanged. RelationOnly is A3+A4 (normal-calibrated components); EnergyOnly is A2.

Limitations: attack-recording steady-state FPR is reported, not a clean false-alarm guarantee. Energy dependence and relation-only incremental information are descriptive. Destruction is a cache-observation structural proxy because raw GRU residuals are not retained; it is not claimable as a successful causal intervention. 3T is frozen evidence; 9T requires a frozen result.

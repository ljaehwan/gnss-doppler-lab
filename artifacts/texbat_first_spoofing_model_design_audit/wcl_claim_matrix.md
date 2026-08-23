# WCL claim matrix

| Candidate | Minimum defensible one-paper claim | Evidence needed before a claim | Blocking issue in this audit |
|---|---|---|---|
| Causal Code-Carrier-Clock Closure Monitor | A causal, normal-calibrated closure statistic links receiver clock bias, clock drift, and code/carrier increments across a dynamic PRN mask and transfers by signal-specific canonicalization | source-distinct clean FPR; preregistered physical controls; development/confirmatory TEXBAT order; block-bootstrap delay and detection intervals; frozen GPS-to-Galileo canonicalizer | Direct code/carrier, Doppler, and clock-state prior art leaves only a weak composition claim; score 73/100 |
| Causal NAV-Content Coherence Monitor | A causal multi-PRN decoded-content validity process detects message-incoherent spoofing without analog shortcuts | clean decoder validity, synthetic content mutations, coherent-replay boundary control, causal TEXBAT metrics | Coherent replay can preserve all proposed content, so the observable is not identifiable for the primary threat; score 61/100 |

No model is selected. Consequently there is no defensible minimum WCL claim,
no implementation plan, and no permission to open Tuni attack data. A future
reopening would need genuinely new hardware information, an authenticated
signal, an external sensor, or a physical observable that is not a
re-encoding of the failed single-receiver channels.

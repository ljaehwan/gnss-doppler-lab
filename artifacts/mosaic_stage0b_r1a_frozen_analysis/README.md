# MOSAIC Stage-0B R1a frozen scientific finalization

This directory is the committed analysis freeze for `FROZEN_POLICY_COMPLETION`. Scientific recalculation must occur only after this freeze, its analysis/verifier code, and its tests are committed and pushed.

The existing R1 compact bundle is not itself a scientific verdict: `control_metrics.csv` and `bootstrap_intervals.csv` are empty, plots are placeholders, single/four `status=PASS` denotes receiver execution rather than physics recovery, the finalizer returns fixed `INCONCLUSIVE_PREREG_GATE_UNDERSPECIFIED` after 72 cases, and external `single_gate.json` is absent from the compact artifact.

No IQ injection or receiver replay is authorized by this analysis.

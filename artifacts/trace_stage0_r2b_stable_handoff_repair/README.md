# TRACE Stage-0 R2b stable handoff repair

R2b diagnosed the R2a support collapse as a deferred-handoff code-phase state mismatch. R2a moved a first-tracking state by up to about 30 seconds while resetting the code NCO and residual phase, producing 7.76--54.40 chips of predicted unmodeled drift. R2b instead restores every preserved target-aligned action field at its exact raw sample and excludes a PRN only when no row reaches the frozen guard.

The repair restored cleanStatic from 79 quality-filtered common causal pairs and zero stable >=4-PRN epochs to 148,967 common quality pairs and 14,977 stable >=4-PRN epochs. DS3 and OS3 native cadence/support smokes passed. Source, causal, raw-timeline, common-row exactness, and TRACE score tolerance gates passed; all 149,979 common physical rows were bit-exact and common score error was zero.

Phase A nevertheless failed closed because repA consistently contained one additional terminal quality pair. Whole-replay canonical hashes and block keys therefore differed, so threshold-crossing/alarm identity did not pass. Final verdict: `INCONCLUSIVE_RECEIVER_REPRODUCIBILITY`, failure label `TERMINAL_ROW_SET_NONREPRODUCIBLE`. Phase B is `NOT_AUTHORIZED`; attack performance, normal FPR, controls, AUROC/pAUC, and alarm delay are unavailable and were not computed.

Large receiver builds and native dumps remain under `/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2b-stable-handoff-repair/`. This Git artifact tree contains only manifests, hashes, gate evidence, and unavailable placeholders.

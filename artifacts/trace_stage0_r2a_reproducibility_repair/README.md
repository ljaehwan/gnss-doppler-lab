# TRACE Stage-0 R2a reproducibility repair

This artifact tree preserves the R2 failure, performs the post-hoc canonical PRN/raw-sample audit, freezes the receiver handoff repair, and gates Phase B on Phase A semantic reproduction. Large native dumps remain under the R2a SSD root.

Final status: `INCONCLUSIVE_RECEIVER_REPRODUCIBILITY` with failure label `INSUFFICIENT_MULTI_PRN_SUPPORT`. The R2 rep1/rep2 audit classified a mixed acquisition-start/channel-scheduling shift (Case B) and true carrier/tracking-state difference on identical canonical keys (Case C). The repaired cleanStatic rep3/rep4 source and causal gates passed, and 164,925 common canonical physical rows matched bit-for-bit, but only 79 quality-filtered causal pairs and zero common >=4-PRN quality epochs survived versus the preregistered minimum of 1,000. TRACE score, threshold/alarm reproduction, Phase B attack metrics, and normal FPR are therefore unavailable. Phase B was not authorized or run.

Large replay outputs and per-replay manifests are under `/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2a-reproducibility-repair/`; durable commands, stdout/stderr hashes, exit codes, and run IDs are indexed by `runner_runs.json`.

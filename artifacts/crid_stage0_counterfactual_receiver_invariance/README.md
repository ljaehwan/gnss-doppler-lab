# CRID-GNSS Stage-0

CRID configuration frozen before this CRID evaluation; TEXBAT/OAKBAT were previously inspected by the broader project.

Final verdict: `INCONCLUSIVE_RECEIVER_REPLAY_OR_ALIGNMENT`.

The verified receiver is GNSS-SDR 0.0.19 TRACE-R2c (SHA-256 `2f6e8e969e525bb48b4d94f016af8fd24f433b0be26b51837f316f60a6b911e0`). C0/C1/C2/C3 respectively use (spacing chip, DLL Hz, PLL Hz) = (0.125,1.5,20), (0.10,1.5,20), (0.125,0.5,5), and (0.125,5,25). All use 1 ms coherent integration, order 3, the same fixed handoff and channel/PRN map, and sequential execution.

Two TEX C0 repeats were byte-identical for all 10 native TRACE dumps. Clean four-configuration alignment used absolute raw endpoints with ±1 receiver code-NCO sample tolerance and zero fitted group delay. TEX provided 13,925 and OAK 14,894 epochs with at least four common PRNs. Clean-only q99 holdout FPR was 1.27% (TEX) and 0.73% (OAK), both within the 2% gate.

The first frozen positive raw-IQ control (TEX, 0.15 chip, -3 dB) consumed the complete 4,500,000,000-byte IQ stream and wrote ten native C0 dumps, but the R2c receiver remained sleeping after EOF with no further I/O for more than five minutes. It was interrupted. Therefore C1-C3 control comparability, OAK physical controls, attack replay, ablations, shortcuts, bootstrap, and scenario detection gates were not evaluated. No attack payload was opened, and this engineering failure is not interpreted as a physical NO-GO.

No neural model was implemented. The sole recommended next action is a separately frozen terminal-drain repair for physical-control IQ replay while retaining this CRID score and gate definition.

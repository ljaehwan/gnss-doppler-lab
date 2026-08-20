# CRID-GNSS Stage-0 R1 terminal-drain repair

Engineering-only repair of the frozen CRID receiver replay lifecycle. CRID science settings, handoff, PRN map, equations, gates, and the prior artifact are unchanged. No attack payload was accessed.

Root cause: R2c entered `flowgraph_->wait()` after exact finite-source EOF while all workers were quiescent. The wrapper now accepts natural EOS first, otherwise sends graceful SIGINT only after exact byte consumption and five seconds of ten-dump stability. SIGTERM/SIGKILL are never accepted as success.

Final verdict: `TERMINAL_DRAIN_REPAIR_PASS`.

Status: `READY_FOR_FROZEN_CRID_RESUME`. No attack evaluation was started.

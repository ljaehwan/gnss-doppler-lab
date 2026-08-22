# Q-SET-GNSS Stage-0A R1a Galileo C-1 terminal-drain repair

Final verdict: `READY_FOR_TUNI2025_C3_CLEAN_DOWNLOAD`.

R1 remains preserved as historical inconclusive evidence. R1a changed only process supervision after bounded-source EOF and complete-record parsing of native dumps. The receiver configuration, decoder, windows, and success gates were unchanged.

- Actual format: `>i2 interleaved I,Q`
- Acquired Galileo E1 PRNs: 8
- PRNs continuously tracked for at least 10 s: 7
- Dynamic observed panel: [2, 10, 11, 14, 25, 29, 30, 36]
- Common two-window panel: [2, 11, 14, 30, 36]
- Telemetry-sync auxiliary PRNs: [2, 10, 14, 30, 36]
- New R1a C-1 payload bytes read: 42012414912
- C-3/attack payload bytes read: 0/0

This step performed no Q-SET training, threshold calibration, attack scoring, or detection claim. Even a passing verdict authorizes only a future C-3 clean download; attack data remains unauthorized.

# Q-SET-GNSS Stage-0A R1 Galileo C-1 receiver preflight

Final verdict: `INCONCLUSIVE_RECEIVER_FEASIBILITY`.

The official raw identity and frozen bounded format test passed, reproducing `>i2` big-endian signed-int16 interleaved I,Q. The first 30-second prefix decoded exactly, but GNSS-SDR did not terminal-drain after sample EOF: tracking bytes stopped while the acquisition queue repeatedly reassigned the same Galileo PRNs. The run was stopped fail-closed, the first output was preserved, and the second time window was not started. No partial support result is used as a success claim.

No Q-SET model training, threshold calibration, attack scoring, C-3 access/download, or attack-data access occurred. C-3 clean download and all attack work remain unauthorized.

- Freeze SHA: `de44b6d448b9395a3c1a7199637b0df32cf51077`
- C-1 payload bytes read: `66,012,246,912`
- C-3/attack payload bytes read: `0/0`
- Acquired PRN count: not adjudicated
- PRNs tracked >=10 s: not adjudicated

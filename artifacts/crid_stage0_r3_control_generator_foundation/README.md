# CRID-GNSS Stage-0 R3 control generator foundation

Design freeze: `6fd4de6c1fd6bc9a1db1375b10d8a9227f50763b`.

Generated and replayed 18 positive and 15 negative clean-only controls per domain without opening attack data or computing CRID scores. Final verdict: `INCONCLUSIVE_CONTROL_PROVENANCE`. The raw-IQ controls, truth sidecars, and replay evidence live under `/home/ubuntu/ssd_data/gnss-early-detection/artifacts/crid-stage0-r3-control-generator-foundation`; compact committed files bind their hashes and validation results.

The initial 12-second-only OAK smoke was diagnostic and excluded because handoff time origin was absent. Final smoke uses an exact 45-second clean prefix with only the R0c window replaced.

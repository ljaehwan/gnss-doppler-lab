# CGC real detection v1 input-support failure

The first frozen release stopped at DS7 before its CGC outcome was calculated.
DS7 contains the six required arrays (`complex_iq`, `sample_count`, `time_s`,
`prn`, `channel`, and `segment_index`) but not the optional `cn0_db_hz`
diagnostic included by the newer TEXBAT exporter. The runner had incorrectly
required exact equality with the newer seven-array schema.

The preserved failed release state has `metrics_emitted=false`. The repair only
makes `cn0_db_hz` optional. No score, threshold, interval, persistence rule,
gate, or source changed before the repaired release.

# SPLITCLOCK-GNSS Stage-0A clean identifiability

Verdict: `INCONCLUSIVE_SPLITCLOCK_EXECUTION_OR_PROVENANCE`.

The design freeze was committed and pushed before any clean raw read. Both allowed clean files and the frozen V3 receiver outputs passed integrity and panel-support checks. The pre-score sign/unit gate then failed: native RINEX L1B increments use `+lambda*delta(L)` while the frozen contract specified the opposite sign; the frozen 8 ms tracking-cadence field also conflicts with the verified 4 ms native TRACE cadence. Therefore no feature, score, threshold, synthetic-on-clean control, or attack freeze was executed. Attack and Jammertest raw access remained zero.

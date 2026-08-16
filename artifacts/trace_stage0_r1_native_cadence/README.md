# TRACE Stage-0 R1 native-cadence preregistration

Configuration frozen before this TRACE-R1 evaluation.

Phase A selected Route B before attack scoring. The retained rows transition
from about 1 ms to about 20 ms spacing, but GNSS-SDR continues correlating and
updating the code/carrier NCO every 1 ms. `log_data()` is gated by the GPS
navigation-bit counter after synchronization, so a retained row action governs
only the next 1 ms buffer and cannot map to the next retained row after 19
unobserved updates.

No model has been fit and no attack score has been computed. If an authenticated
native-1ms TRACE receiver dump cannot be generated within task scope, the frozen
verdict is `NEEDS_TRACE_SPECIFIC_RECEIVER_DUMP` with no performance claim.

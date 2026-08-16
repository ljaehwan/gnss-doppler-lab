# TRACE Stage-0 R1 native-cadence protocol

TRACE-R1 first audits receiver cadence from consecutive sample stamps and then
authenticates the receiver's correlation, discriminator, filter, NCO update,
dump, and consume order. Attack scoring is forbidden until that gate passes.

The retained GPS L1 C/A products use the receiver default one-symbol tracking
loop. After navigation-bit synchronization, the source still correlates and
updates the NCO every 1 ms, but writes a tracking row only when the 20-symbol
data-bit counter wraps. Thus a retained row action applies to the next 1 ms
buffer, not to the next retained correlator about 20 ms later. Nineteen
intermediate actions are absent, so the retained 20 ms row-to-row mapping is
physically invalid.

Prompt-referenced complex normalization removes common gain, carrier phase, and
navigation-bit sign. Consequently TRACE-R1 does not multiply normalized taps by
`exp(-j*2*pi*carrier_doppler_hz*dt)`. Full Doppler rotation would double-apply a
removed common phase. Carrier information is admissible only as separately
verified signed residual/action context.

Route B requires a new receiver dump with native 1 ms complex nine taps,
receiver-applied next-buffer code/carrier actions, sample stamp, integration and
loop-boundary flags, C/N0, lock, and authenticated IQ/config/source hashes. The
current task is fail-closed until such a dump exists; no attack metrics or
performance claims are produced.

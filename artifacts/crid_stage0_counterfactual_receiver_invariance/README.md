# CRID-GNSS Stage-0

CRID configuration frozen before this CRID evaluation; TEXBAT/OAKBAT were previously inspected by the broader project.

This directory binds the frozen receiver configurations, same-IQ replay
contract, clean-only causal model, physical-control grid, thresholds, gates,
and independent verifier. Attack payload access is forbidden until the freeze
commit has been pushed and its remote SHA recorded.

The freeze uses GNSS-SDR 0.0.19 TRACE-R2c, fixed pre-attack handoffs and
channel/PRN maps, and sequential replay. A 45-second TEX C0 benchmark peaked
at 662400 KiB RSS. Two sequential C0 runs produced byte-identical native
TRACE dumps for all ten fixed channels.

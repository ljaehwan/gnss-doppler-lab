# Jammertest 2025 CRPA Stage-0 metadata-only feasibility audit

Final verdict: `INCONCLUSIVE_SCHEMA_REQUIRES_ONE_BOUNDED_H5_SAMPLE`. No raw IQ/HDF5/LFS payload was downloaded or opened, and no model, training, or score was run.

The release contains a 2×2 CRPA and a reader that exposes four complex channels, but it does **not** directly bind channel order, synchronization, relative phase preservation, calibration, geometry, or orientation. The CRPA is a single 1,398,308,992-byte NPY LFS object, not the split HDF5 series described generically in the README. Public CRPA split rows contain type/power/bandwidth and area-by-filename, but no clean label, band, VGA, time/day, transmitter, recording group, or position.

Consequently, neither a leakage-safe balanced subset nor a power/VGA-matched spatial spoofing experiment can be authorized. The exact next payload object is documented only as a bounded follow-up target; this audit does not authorize downloading it. The literature audit also finds direct prior blind multi-antenna snapshot/eigen/ML work, so novelty would need the narrower destruction-controlled cross-domain field claim after provenance closure.

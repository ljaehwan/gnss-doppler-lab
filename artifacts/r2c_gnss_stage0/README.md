# R2C-GNSS Stage-0 artifact

Verdict: `DATA_INVALID`. No real TEXBAT attack epoch was evaluated. The required receiver-produced complex nine-tap vectors, corresponding raw IQ reconstruction inputs, and time-aligned LOS geometry are absent. Existing nine-tap products are magnitude-only and prohibited as primary evidence.

Synthetic control outputs validate software mechanics only; they do not establish real-world attack performance. All unavailable tables carry explicit status rows. See `docs/R2C_GNSS_STAGE0.md` for equations, scope, ablations, and limitations. The retry used sandbox bypass after the first run failed at Bubblewrap loopback setup; this is recorded in `provenance.json`.

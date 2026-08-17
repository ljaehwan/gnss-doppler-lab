# MOSAIC Stage-0A R1 raw-IQ recorrelation

Final verdict: `STAGE0A_RAW_ALIGNMENT_PASS`. This means only that real cleanStatic raw IQ can be reconstructed into the receiver's native complex nine taps under the source-frozen convention; it is not Stage-0B GO.

The old navigation-bit blocker is not required for exact 1 ms Stage-0A tap-shape alignment because the bit is a common ±1 complex scalar removed by the independently fitted per-epoch amplitude. Stage-0B still requires decoded/validated navigation-bit provenance to synthesize continuous raw IQ.

The carrier sign, code-delay sign, tap coordinates, sample format, and current-interval state timing were fixed from the patched receiver source and receiver configs before looking at alignment metrics. No attack recording was read, no alternative convention was selected by outcome, and no threshold was changed.
The reported delay center error cross-checks the native residual-chip field against the independently dumped residual-sample field. The Doppler center error cross-checks native Doppler against the independently dumped phase-step field. Complex cosine and magnitude Spearman compare the native and directly raw-reconstructed nine taps after one global complex-amplitude fit per epoch.


- OAKBAT.cleanStatic: PASS, 235 valid, 0 rejected, gate pass fraction 1.000000.
- TEXBAT.cleanStatic: PASS, 235 valid, 0 rejected, gate pass fraction 1.000000.

No target PRN failed the frozen gate.

Next action: Obtain a decoded and independently validated navigation-bit sequence in a separate Stage-0B provenance task.

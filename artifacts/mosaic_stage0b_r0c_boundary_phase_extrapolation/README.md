# MOSAIC Stage-0B R0c boundary phase extrapolation

Verdict: **BOUNDARY_PHASE_EXTRAPOLATION_PASS_WITH_SCOPE_LIMITATION**.

The modulo-20 phase was selected independently for each PRN from only the chronological first half of post-sync receiver `data_symbol_boundary==1` rows. The chronological second half was held out. Prompt sign, parity, preamble, and TOW were not phase-selection inputs and were checked only after the phase was frozen.

- OAKBAT.cleanStatic PRN 10: phi=19, fit 86/86, holdout 86/86
- OAKBAT.cleanStatic PRN 11: phi=12, fit 69/69, holdout 70/70
- OAKBAT.cleanStatic PRN 21: phi=5, fit 91/91, holdout 92/92
- OAKBAT.cleanStatic PRN 24: phi=2, fit 69/69, holdout 70/70
- OAKBAT.cleanStatic PRN 27: phi=14, fit 69/69, holdout 70/70
- TEXBAT.cleanStatic PRN 3: phi=0, fit 87/87, holdout 87/87
- TEXBAT.cleanStatic PRN 13: phi=19, fit 87/87, holdout 87/87
- TEXBAT.cleanStatic PRN 16: phi=4, fit 87/87, holdout 87/87
- TEXBAT.cleanStatic PRN 19: phi=4, fit 87/87, holdout 87/87
- TEXBAT.cleanStatic PRN 30: phi=11, fit 87/87, holdout 87/87

All holdout flags match their fitted phase. For every PRN, the complete TRACE sequence from the first corrected pre-sync bit through the last direct flag has no missing or duplicate 1-ms row, channel/PRN/session change, reset/reacquisition, file boundary, unexplained NCO sample gap, or raw-IQ bounds failure. The sole state evolution is normal pull-in state 2 to tracking state 4, with no regression.

The corrected mapping is an exact field-for-field preservation of R0b's `+1 TRACE epoch` mapping plus phase-audit columns. Its endpoints remain copied from actual TRACE rows; no nominal `20 ms * fs` endpoint construction was used. Corrected ends match the extrapolated phase for 6000/6000 bits.

Post-selection validation passes: corrected Prompt 6000/6000, observed Prompt transition 754/754, parity 200/200, preamble 20/20, and TOW continuity 10/10. Frozen hashes are unchanged.

- OAK authorized interval: `[150275296, 210202273)`
- TEX authorized interval: `[817815304, 1117517038)`

Stage-0B injection is authorized only inside those intervals, but no injection was executed in R0c. Attack data, synthetic injection, model training, and detector experiments were not run. The scope remains one approximately 12-second interval per dataset; distant-interval validation was not performed.

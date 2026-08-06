# ACAF-NF Stage-0 static

This is physical feasibility only, not a neural network or active policy. It directly rereads native 25 Msps signed-int16 interleaved I/Q and calculates 1-ms CAF using a locally generated GPS C/A replica on code -1..+1 chips/.125 and Doppler -250..+250 Hz/50.

The pre-evaluation schedule is cleanStatic 30/180/330/420 s and each attack 10/60/candidate-2/candidate+2 s. Candidates are unasserted metadata: DS1 125, DS2 110.1, DS3 118.9, DS4 113.8, DS7/8 110. CleanStatic only has chronological fit/threshold/held-clean roles. No attacks fit. Historic tracker data are permitted only after raw hash proof for centers, never historic CAF/taps. Missing exact DS7/8 temporal nonoverlap fails closed INCONCLUSIVE. B0 is NOT_DIRECTLY_COMPARABLE with no unsafe CSV reuse.

Controls include two direct-C/A sources, gain .5/.8/1.2/2, phase, AWGN, fixed/random/shuffled query order and fixed EPL/fixed9/rawpower/dense K=3/5/9/16. Compact strict JSON, SHA-256 checksums, and a plot are under `artifacts/acaf_nf_stage0_static`.

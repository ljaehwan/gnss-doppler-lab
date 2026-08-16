# Q-COMET Stage-0 static result

Final verdict: `NO_GO_SHARED_ONSET_HYPOTHESIS`. Freeze commit: `dd1746e63498fa4c611876bd4eb8f5242df211b4`. Every GO criterion was required; the exact failure reasons and criterion values are in `final_verdict.json`. Neural Stage-1 was not implemented.

## Inputs, provenance, and time

The scored fields were receiver-relative sample count/time, PRN support, and genuine complex I/Q at E4/E3/E2/E/P/L/L2/L3/L4 (0.125-chip spacing). Preserved audit-only receiver fields include carrier Doppler, code frequency/error, carrier discriminator, C/N0, and carrier lock; C/N0, lock, Prompt amplitude, and IQ power were never direct scores. `source_binding.json` binds raw IQ, receiver binary/config/source, and MAT/NPZ lineage. `timeline_inventory.json` binds official receiver-relative onsets. DS4 ends near 128.2 s and is transition-only. DS7/DS8 are one family; `ds78_byte_identity_audit.json` prevents byte-identical pre-110 s data from being counted as independent normal confirmation.

## Hypotheses and quotient

H0 is `c[i,t] = f_theta(c[i,t-1:t-L]) + epsilon`, fitted only to cleanStatic, with calibration-A Ledoit-Wolf covariance. The innovation is `r = Sigma^(-1/2)(c-c_hat)` and `v = (I-J(J'J)^(-1)J')r`. The observed-local-peak Jacobian removes gain, common carrier/navigation-bit phase, local code-delay recentering, and tap-dependent Doppler phase; it preserves the remaining signed complex deformation directions. H1 uses free PRN-specific coefficients on frozen step/ramp/transient bases and `S_t = max_k sum_i log((1-pi)+pi*B_i(k,t)) - 0.5 log(L+1)` over a causal 10 s window. Full does not impose a common deformation direction.

## Normal reliability and performance

The chronological cleanStatic split is train 20–140 s, calibration-A 150–210 s, calibration-B 220–340 s, and holdout 350–470 s, with 10 s guards and byte-disjoint ranges. Calibration-A selected ridge-VAR lag 2 and fit shrinkage covariance; calibration-B alone set q99/q99.5 thresholds. Clean holdout FPR was 0.00%.

TEXBAT Full — DS3: pAUC=0.555, pre-FPR=0.000, delay=49.6 s, onset=198.5 s; DS4: pAUC=0.500, pre-FPR=0.000, delay=no alarm, onset=1.0 s; DS7: pAUC=0.714, pre-FPR=0.000, delay=92.5 s, onset=270.0 s; DS8: pAUC=0.869, pre-FPR=0.000, delay=92.5 s, onset=400.5 s. DS3/DS4/DS7-8 therefore did not provide plausible common-onset recovery, and worst pre-onset FPR was 0.00%.

OAKBAT frozen confirmation — OS1: pAUC=0.499, pre-FPR=0.004, delay=2.0 s, onset=120.5 s; OS2: pAUC=0.858, pre-FPR=0.004, delay=2.0 s, onset=142.5 s; OS3: pAUC=0.501, pre-FPR=0.000, delay=23.5 s, onset=142.5 s; OS4: pAUC=0.973, pre-FPR=0.008, delay=0.5 s, onset=142.5 s. The structure remained frozen and only the normal predictor/covariance/threshold were refit on OAKBAT cleanStatic. OS-family onset behavior was mixed and fewer than two scenarios had onset error within 10 s.

Paper-B0 A0 is `UNAVAILABLE_WITH_REASON`: no authenticated exact common-support adapter/checkpoint was bound at freeze, so historical CSV performance was not copied. `ablation_metrics.csv` contains A1–A7, Full, EPL3, and No-quotient on common receiver units/support. Full exceeded both A3 and A5 in the required two families (DS3, DS7-8). No core recording remained unavailable: DS4 was source-bound from raw IQ, and OAKBAT was regenerated from raw after an initial normal-clean adapter failure revealed that legacy MATs lacked genuine complex outer-tap I/Q.

Relation destruction — DS3: drop=35.7%, 95% CI delta=[-1.48,2.75]; DS4: drop=0.0%, 95% CI delta=[0,0]; DS7: drop=72.6%, 95% CI delta=[2.81,162]; DS8: drop=22.8%, 95% CI delta=[-26.6,145]. Only families with at least 30% score drop and a strictly positive paired 10 s block-bootstrap CI qualify; fewer than two qualified. The same bootstrap also reports pAUC delta, alarm-delay change, and persistent-detection delta.

Correlator-domain controls covered gain 0.5/0.75/1.25/2, carrier phase, navigation-bit sign, code recentering, Doppler, empirical clean noise 0.5/1/2, C/N0 metadata, clock-like drift, one-PRN disturbance, PRN drop/add, independent multipath, timestamp gap, reacquisition, and exact-aligned counterfeit. Sustained alarms occurred for: gain_2.0, phase_0.5, phase_1.5, small_doppler_shift, receiver_clock_like_drift, exact_aligned_counterfeit_expected_undetectable. These are not raw-IQ physical proofs. The exact-aligned case documents the information-theoretically-undetectable boundary.

Official-versus-estimated onsets, participation, and per-PRN Bayes factors are in the CSVs and evidence-backed plots. Across the eight Full scenario rows, the maximum absolute Pearson correlation was 0.508 with total raw-residual energy and 0.480 with total Prompt power. Q-COMET targets the first receiver-correlator-visible common change, not necessarily transmitter RF turn-on.

Claimable contribution: an auditable, normal-only quotient/common-onset Stage-0 implementation plus reproducible negative-result evidence. Non-claimable: raw-RF immunity, universal spoof detection, absolute transmitter onset recovery, superiority to Paper-B0, or evidence for neural Stage-1.

Recommended next action: stop Q-COMET and retain this Stage-0 bundle as the negative-result record.

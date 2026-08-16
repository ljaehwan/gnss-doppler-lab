# TRACE Stage-0 static result

Configuration frozen before this TRACE evaluation.

## Outcome

Final verdict: `INCONCLUSIVE_INPUT_OR_ALIGNMENT`

Fail-closed reason: `TRACKER_ACTION_ALIGNMENT_UNRESOLVED`.

No attack correlator scores, ROC/PR metrics, delays, FPRs, or comparative
detection claims were computed. The result is inconclusive because the retained
receiver products cannot instantiate the preregistered native 1 ms action-to-next
correlator sequence on frozen OAKBAT confirmation.

## Inputs and alignment

The actual common TEXBAT/OAKBAT receiver fields are complex I/Q for
`E4,E3,E2,E,P,L,L2,L3,L4`, complex `Prompt_I/Q`, `code_error_chips`,
`code_error_filt_chips`, `carr_error_hz`, `carr_error_filt_hz`,
`carrier_doppler_hz`, `code_freq_chips`, residual code phase `aux1`,
`PRN_start_sample_count`, `CN0_SNV_dB_Hz`, and `carrier_lock_test`. Raw IQ,
receiver config, manifest, and every source MAT are SHA-256-bound in
`source_binding.json`. OAKBAT genuinely has full complex nine-tap data; no real
tap/Prompt fabrication was used.

Receiver source ordering establishes that row `t` first correlates the current
interval, then updates DLL/PLL/NCO state and logs the action for the next
interval. The sample stamp is `nitems_read + current_prn_length`; the same
length is consumed afterward. This verifies emitted-row semantics, action sign,
and units. However, after navigation-bit synchronization the retained nine-tap
rows are predominantly 20 ms accumulated correlators. Frozen OS3 and OS4 have
zero native 1 ms post-onset row pairs and therefore zero blocks with the
required four native PRNs. That makes the required tracker action/native-next-
correlator alignment unavailable during confirmation.

The frozen analytic warp was

`W(x,u)_k = exp(-j phi_u) interp[x(t)](tau_k + delta_tau_u)`,

where `delta_tau_u = (code_freq - 1.023e6*(1+doppler/1.57542e9))*Delta t` and
`phi_u = 2*pi*doppler*Delta t`. It uses complex interpolation, no extrapolation
or zero padding, and a common seven-interior-tap covariance support. Prompt-
referenced complex normalization removes common gain, phase, and navigation-bit
sign as direct evidence.

## Planned fit and real-scenario results

The preregistered cleanStatic plan was a chronological train/calibration/
holdout split with guard intervals, normal-only shared ridge correction,
Ledoit-Wolf residual covariance, q99 calibration, median non-overlapping 0.5 s
pooling, at least four PRNs, and the actual three-consecutive-block alarm.
Because the input/alignment gate failed, the split, real normal model,
thresholds, and leakage audit are marked `UNAVAILABLE`; no substitute 20 ms
model was fit.

DS3/DS7 development, OS3/OS4 confirmation, and DS1/OS1 boundary results are all
`UNAVAILABLE`. A0/A1/A2/A3/A4 and B0 comparisons are likewise unavailable.
B0 was not populated from historical CSVs. Action shuffle and real gain/phase/
C/N0/noise/PRN controls are unavailable because fitting/scoring them would
constitute performance evaluation after a failed alignment gate.

## Synthetic physics

The independent complex-domain synthetic sweep passed: two-source median
residual was 0.000739 versus 0.000262 for single-source/nuisance controls; the
paired mean effect was 2.669 with bootstrap 95% CI [1.480, 3.722], Wilcoxon
one-sided p-value numerically 0, and resolvable-separation detection probability
73.13% at the control q99 threshold. It swept -3/0/+3 dB, 16 phases, 0--0.5 chip,
0--50 Hz, four PRNs, and eight reference epochs. Exactly 192 Prompt-null points
were quality-masked. This validates the synthetic complex-superposition sanity
check only and cannot rescue unavailable real alignment.

## Claims and failure interpretation

Claimable: the TRACE source contract, normalization, no-padding analytic warp,
shared normal-only linear implementation, focused invariant/causality tests,
input provenance inventory, cadence audit, and physically valid synthetic
two-source effect.

Not claimable: real spoofing detection performance; Full versus A1 or B0;
clean/external FPR; attack delay; OS3/OS4 confirmation; dynamic generalization;
exact injection detection; or a B0 replacement. TRACE-N is not designed or
implemented because the verdict is not GO.

One recommended next action: generate new TRACE-specific receiver dumps that
retain per-channel native 1 ms complex nine taps and the receiver-applied action
through OS3/OS4, with authenticated raw-IQ/config/source bindings, then rerun the
unchanged frozen protocol from the beginning.

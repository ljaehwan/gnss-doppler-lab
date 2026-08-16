# TRACE Stage-0 static protocol

TRACE is a normal-only action-conditioned predictor of subsequent complex
correlator evolution. Its claim scope is a static receiver under authentic plus
counterfeit coexistence and matched-power/coherent carry-off. It is not a claim
of universal spoofing detection, injection-instant localization, dynamic
generalization, or guaranteed seamless-takeover detection.

## Receiver fields and causal alignment

Both TEXBAT and OAKBAT use the same patched GNSS-SDR build and the same genuine
complex tap contract `E4,E3,E2,E,P,L,L2,L3,L4`. TRACE reads their I/Q components,
`Prompt_I/Q`, `code_error_chips`, `code_error_filt_chips`, `carr_error_hz`,
`carr_error_filt_hz`, `carrier_doppler_hz`, `code_freq_chips`, residual code
phase (`aux1`), `PRN_start_sample_count`, C/N0, and carrier lock. C/N0 and lock
are quality gates only. PRN is grouping metadata, never a model input.

The bound receiver source performs correlation first
(`dll_pll_veml_tracking.cc:1923`), then runs the DLL/PLL (`1964`), updates the
next-buffer NCO (`1965`), and logs it (`1968`). The dump stamp is
`nitems_read(0)+d_current_prn_length_samples` (`1484`), and that length is
consumed at `2199`. Therefore row `t` correlators describe the completed
interval, while row `t` NCO/action applies to the correlation that produces row
`t+1`. Consecutive rows must retain PRN and have a consumed interval within
0.9--1.1 ms. Source definitions fix offset, sign, and units before attack
evaluation; attack results cannot select them.

## Normalization and warp

For tap `k`, TRACE uses

`x_k(t) = c_k(t) conj(c_P(t)) / (|c_P(t)|^2 + epsilon)`.

This removes common gain, carrier phase, and navigation-bit sign as direct
evidence. Low Prompt power, C/N0 below 28 dB-Hz, or lock below 0.85 is masked.

The analytic warp is

`W(x,u)_k = exp(-j phi_u) interp[x(t)](tau_k + delta_tau_u)`

with

`delta_tau_u = (code_freq - 1.023e6*(1+doppler/1.57542e9))*Delta t`

and `phi_u = 2*pi*doppler*Delta t`. Complex real/imaginary interpolation is
used. Extrapolation and zero padding are forbidden. Because any nonzero shift
loses one aperture edge, the predeclared common covariance support is the seven
interior taps `E3..L3`; the model still consumes the full nine-tap vector. EPL
ablation A4 uses `E,P,L`.

## Frozen model and scoring

Full TRACE adds a shared PRN-local ridge correction to the analytic warp. Its
inputs are the current normalized nine-tap real/imaginary vector and current
action; there is no PRN identity and no future context. A1 is a direct
action-free shared ridge predictor. A0 is persistence, A2 warp-only, A3 is
action-scalar-only, and A4 is EPL TRACE. Coefficients are trained only on the
chronological cleanStatic training split. Ledoit-Wolf residual mean/covariance
and thresholds are estimated only from calibration. OAKBAT reuses frozen
predictor coefficients and re-estimates only its cleanStatic normal
mean/covariance/threshold.

The PRN-local score is signed-vector Mahalanobis energy. Native 1 ms scores are
pooled by a fixed median into non-overlapping 0.5 s blocks with at least four
PRNs. The primary threshold is clean calibration q99, and the alarm is the
actual third and subsequent consecutive above-threshold block, reset across
gaps. q99.5 and instantaneous results are secondary.

## Split, controls, and decision

cleanStatic is split 50/25/25 chronologically with 5 s guards. Raw sample and
byte overlap are audited. The synthetic sanity test adds authentic and spoof
components only in the complex domain while sweeping power, phase, delay, and
residual Doppler. Action shuffling is within PRN and 3 dB-Hz C/N0 bins.

Configuration frozen before this TRACE evaluation. Development roles are DS3
and DS7; OS3 and OS4 are frozen confirmation. DS1 and OS1 are boundary
diagnostics. B0 must be rerun on identical support or be explicitly unavailable;
historical result CSVs may not be copied. The only verdicts are
`GO_FOR_TRACE_STAGE1`, `NO_GO_ACTION_EQUIVARIANCE`, and
`INCONCLUSIVE_INPUT_OR_ALIGNMENT`. If tracker alignment is unresolved, the
mandatory reason is `TRACKER_ACTION_ALIGNMENT_UNRESOLVED` and performance is not
claimed.

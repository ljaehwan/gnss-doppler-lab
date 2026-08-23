# Candidate 1: Causal Code-Carrier-Clock Closure Monitor

## Core hypothesis

A spoofing pull-off that changes receiver time or range must break the causal
closure among pseudorange-derived clock bias, carrier-Doppler-derived clock
drift, and per-PRN code/carrier increments unless the adversary synthesizes
all of those observables coherently.

## Observable and nuisance model

At receiver epoch `k`, robustly solve the pseudorange linearization:

```text
(delta_x_rho_hat[k], b_rho_hat[k])
  = argmin_(delta_x,b) sum_(i in A[k]) w_rho[i,k] *
    psi(rho[i,k] - range_i(x[k]) - u[i,k]^T delta_x - c*b)
```

and the range-rate/Doppler linearization

```text
(delta_v_D_hat[k], bdot_D_hat[k])
  = argmin_(delta_v,bdot) sum_(i in A[k]) w_D[i,k] *
    psi(-lambda[i]*f_D[i,k] - range_rate_i(x[k],v[k])
        - u[i,k]^T delta_v - c*bdot)
```

The clock-closure residual is

```text
e_clk[k] = (((b_rho_hat[k] - b_rho_hat[k-1]) / delta_t)
            - bdot_D_hat[k]) / sigma_clk_hat[k]
```

With carrier phase sign fixed by the receiver convention, define the
per-PRN code/carrier increment closure

```text
e_cc[i,k] = (delta_rho[i,k] + lambda[i]*delta_phi[i,k]
             - g_hat[i,k] - c*d_hat[k]) / sigma_cc_hat[i,k]
```

where `g_hat[i,k]` removes satellite/receiver geometry and
`d_hat[k]` is a robust common receiver-clock term. The causal epoch
feature is

```text
z[k] = (abs(e_clk[k]),
        median_(i in A[k]) abs(e_cc[i,k]),
        MAD_(i in A[k]) e_cc[i,k])
```

PRN identity is not an input. `A[k]` is an availability mask, not a fixed
panel. C/N0 and lock only censor invalid measurements; they do not contribute
to the score.

Normal motion, oscillator drift, ephemeris error, multipath, cycle slips,
receiver resets, and reacquisition are nuisances. Geometry terms, robust
clock projection, heteroscedastic clean covariance, and explicit
cycle-slip/reacquisition censoring remove or expose them.

## Causal statistic and normal-only learning

Use a trailing 10 s window ending at `k`. Fit only a low-capacity clean
state-space covariance model, then map the robust Mahalanobis energy of
`z[k]` into a one-sided CUSUM. Calibration uses chronological source-distinct
clean blocks. The threshold is the clean calibration quantile of block-max
CUSUM at a preregistered false-alarm target; three consecutive causal
exceedances provide persistence. No attack label, onset, scenario, filename,
or PRN ID enters fitting or inference.

## Falsification controls

Positive controls, generated before attack access:

- inject a code/time ramp without the consistent carrier/Doppler clock drift;
- inject the same inconsistency into one and four PRNs;
- inject a clock-drift change without its pseudorange integral.

A fully coherent code/carrier/Doppler/time synthesis is a boundary control
that is expected not to alarm and must be disclosed as an undetectable class.

Negative controls:

- authentic receiver motion with the same acceleration envelope;
- oscillator drift and time-step perturbations;
- ephemeris issue changes;
- multipath, cycle slips, channel loss, and reacquisition with the censoring
  rule fixed in advance;
- receiver/session transfer with source-distinct normal recordings.

## TEXBAT and Tuni decision conditions

Before any attack access, a future study would have to freeze the following
TEXBAT success condition: source-distinct clean holdout FPR at most 0.01;
valid dynamic-panel support of at least four PRNs; every physical positive
control detected and every negative control within its clean bound; at least
0.80 attack detection probability with the recording-block-bootstrap 95%
lower confidence limit above 0.50; median first-alarm delay at most 10 s; and
no detection attributable to lock, C/N0, or PRN-count collapse. Scenario
families, not epochs, would be the bootstrap and confirmatory units. The
existing DS1-DS4 and DS7-DS8 exposure means this could not be called fully
blind.

Tuni2025 SS-3/SS-5 could be an external partial-PRN test only after a TEXBAT
model freeze. GPS L1 C/A and Galileo E1 require signal-specific wavelength,
observable-sign, and telemetry canonicalizers. The physical equations,
capacity, threshold rule, persistence, and gate could not change after Tuni
access.

## Prior art, novelty risk, and feasibility

Gao et al. directly study pseudorange/carrier consistency
(DOI 10.1109/TST.2013.6678905); Chu et al. and Zhou et al. use calculated or
positioning Doppler consistency (DOIs 10.33012/2017.15107 and
10.33012/2022.18251); recent clock-state verification uses pseudorange and
Doppler closure (DOI 10.3390/s26020397). The combination is physically
credible but the remaining WCL contribution risks being an engineering
composition rather than a novel detector.

Expected failures are coherent synthesis, authentic clock discontinuities,
unmodeled receiver-specific observable conventions, insufficient
carrier-phase continuity, and dataset leakage through shared sessions.
Computation is linear in available PRNs per epoch and feasible from receiver
observables if carrier phase and PVT clock states were retained.

## Frozen audit score

Physical identifiability 4/5; independence 4/5; clean-only falsifiability
5/5; TEXBAT onset fit 4/5; Tuni transfer 2/5; novelty 1/5; feasibility 4/5.
Weighted total: **73/100**. It fails the required 75/100 threshold and is not
selected.

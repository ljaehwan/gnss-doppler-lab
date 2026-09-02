# CGC exact-Doppler-lock weakening: root-cause audit v1

## Outcome

The weak exact-Doppler-lock result is **not explained by loss of receiver lock,
low C/N0, or a large nearest-template distance**.  The frozen simulator made a
more specific condition: separately generated authentic and counterfeit
components both initialized every carrier channel at zero phase, while the
locked counterfeit reused the authentic carrier trajectory.  Their relative
carrier phase was consequently fixed at zero.  This is a coherent in-phase
blind state for the present signed-delay sensor.

Across all five released static pairs, the exact-locked hold interval had less
prompt-aligned quadrature energy and much poorer agreement between the
estimated PRN-delay vector and the true code-delay geometry:

| Hold-bin median | Carrier-coupled | Exact Doppler-locked |
|---|---:|---:|
| Quadrature fraction | 0.0995 | 0.0439 |
| Absolute centered delay correlation | 0.9255 | 0.6589 |
| Delay-geometry agreement, correlation squared | 0.8565 | 0.4342 |
| Partial-F p-value | 0.000733 | 0.095646 |
| Nearest-template distance | 0.0486 | 0.0362 |
| C/N0 (dB-Hz) | 48.19 | 49.56 |

The lower template distance in the weaker condition is important.  The
profile is not outside the dictionary; instead, an almost-real in-phase
profile can be fitted closely by a wrong combination of authentic-center,
amplitude, and signed-delay nuisance parameters.  The per-PRN estimator is
therefore confident about a delay vector that has lost much of the true LOS
pattern.  The LOS-plus-clock model then reduces the clock-only error less,
which raises the Partial-F p-value above the frozen 0.060284 threshold.

## Receiver-in-loop phase intervention

To test causality, the Tokyo component pair, code trajectory, power envelope,
AWGN seed, frontend realization, gain, and GNSS-SDR configuration were held
fixed.  Only one global complex phase multiplying the locked counterfeit
component was changed.  The original 0-degree receiver result was reused; all
other IQ streams were generated independently under the same frozen frontend
reference.

| Spoof phase | Quadrature fraction | Delay R2 | Median pF | Hold raw alarm | Latency | Median C/N0 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 deg | 0.0501 | 0.3361 | 0.148351 | 38.9% | 18 s | 47.18 dB-Hz |
| 45 deg | 0.0690 | 0.7462 | 0.003857 | 77.8% | 6 s | 46.92 dB-Hz |
| 90 deg | 0.1374 | 0.8888 | 0.000417 | 100% | 5 s | 46.96 dB-Hz |
| 135 deg | 0.2958 | 0.9500 | 0.000023 | 100% | 2 s | 44.05 dB-Hz |
| 180 deg | 0.2159 | 0.7156 | 0.033987 | 61.1% | 14 s | 39.85 dB-Hz |

All four newly generated receivers acquired 11 PRNs.  The 45--135 degree
conditions recover despite comparable or lower C/N0 than at 0 degrees.  This
direct intervention supports relative carrier phase as the cause of the
original weakness.  The 180-degree result is not monotonic because destructive
combining reduces C/N0; it is a simultaneous cancellation stress case.

## Physical interpretation

After prompt-phase rotation, the sensor observes

\[
z_{i,k}=a_iR(\xi_k-\mu_i)
 + b_i e^{j\phi_i}R(\xi_k-\mu_i-\delta_i)+n_{i,k}.
\]

At relative phase \(\phi_i=0\), both paths lie on almost the same real axis.
The imaginary half of the complex nine-tap feature supplies little independent
information, and finite frontend filtering plus movement of the receiver's
tracking point makes \(\mu_i\), \(b_i\), and \(\delta_i\) more confounded.  At
nonzero phase, the delayed component contributes a distinct quadrature pattern;
the dictionary can recover signed delay more faithfully.  At phase near
\(\pi\), destructive cancellation introduces a different observability limit.

The causal chain supported by these diagnostics is therefore:

`zero relative phase -> little quadrature diversity -> ambiguous signed delay
per PRN -> weak cross-satellite LOS fit -> larger Partial-F p-value`.

## Claim boundary and next experiment

This result must not be rewritten as proof that all Doppler-locked attacks are
detected.  Conversely, the original zero-phase failure must not be generalized
to all Doppler-locked attacks.  This is one global-phase sweep on one reused
geometry, and the 0-degree state arose deterministically from this simulator's
zero phase initialization.  A phase-aware release should pre-register several
unseen geometries, power ratios, global phases, and independent per-PRN phases.

For the detector, the immediate safe improvement is an observability gate: low
quadrature diversity or strong destructive cancellation should produce
`insufficient two-path observability`, not a multipath label.  Any revised
delay estimator should then be developed without changing the released
five-pair result and validated on a fresh phase-stratified set.

## Reproduction

```bash
.venv/bin/python scripts/audit_cgc_locked_phase_root_cause.py
.venv/bin/python scripts/run_cgc_locked_phase_sweep_dev.py --phases-deg 0 45 90 135 180
.venv/bin/python scripts/render_cgc_locked_phase_sweep.py
```

Small canonical outputs are under
`artifacts/cgc_locked_phase_root_cause_v1/` and
`artifacts/cgc_locked_phase_sweep_dev_v1/`.  The 5.9-GiB RF and receiver bundle
is retained at `/home/ubuntu/hdd_data/cgc_locked_phase_sweep_dev_v1/`.

# CGC Three-Regime Research Direction V2

Status: **adopted direction; development evidence only**

This document supersedes the comparison scope in
`CGC_DOPPLER_SILENT_HOLD_EXPERIMENT_DESIGN_V1.md`. The V1 design applies only
to carrier-coupled, consistent-Doppler carry-off.

## Why this revision is necessary

The Tu et al. comparator applies to *consistent-Doppler* intermediate
spoofing, not to every code carry-off. A signal generator can move counterfeit
code phase while either:

1. changing the counterfeit carrier Doppler consistently with its code rate;
   or
2. keeping the counterfeit carrier Doppler locked to the authentic signal.

Tu et al. explicitly call the second case locked-Doppler spoofing and exclude
it from their dual-frequency-peak detector. They identify carrier/code
Doppler consistency as the relevant comparator for that transition. The
paper is <https://doi.org/10.1049/iet-rsn.2019.0366>.

Therefore the WCL study must not claim that code carry-off necessarily creates
a distinct carrier-Doppler peak. It must separate the physical regimes below.

## Physical model

Let the counterfeit-minus-authentic code delay for PRN `i` be

\[
\delta_i(t)=\tau_{s,i}(t)-\tau_{a,i}(t).
\]

Smooth carry-off requires a nonzero code-delay rate

\[
\dot\delta_i(t)\ne0.
\]

For GPS L1 C/A, a physically coupled carrier follows approximately

\[
\Delta f_{c,i}(t)=1540\,\dot\delta_i(t),
\]

up to the sign convention used for range and received Doppler. Define the
carrier-equivalent consistency mismatch

\[
q_i(t)=\Delta f_{c,i}(t)-1540\,\dot\delta_i(t).
\]

This gives the three regimes that matter:

| Regime | Code displacement | Code-delay rate | Carrier separation | Main observable |
|---|---:|---:|---:|---|
| Consistent-Doppler pull-off | growing | nonzero | nonzero and coupled | Tu dual peak; CGC as delay becomes observable |
| Locked-Doppler pull-off | growing | nonzero | near zero | carrier/code mismatch; CGC if signed delay is identifiable |
| Position hold | retained | near zero | near zero | accumulated code-delay geometry |

The position-hold carrier difference is not mathematically exactly zero in all
geometries because satellite LOS changes at two displaced positions. The
relevant question is whether it remains below the frequency-domain input
resolution while the code displacement remains inside the correlator aperture.

## Fair comparison contract

The comparison must include all three methods without stretching their stated
scope:

- **Tu-style dual-frequency input:** report whether at least five PRNs have a
  spoof/authentic carrier separation of at least 3 Hz. Until FFT-CDS is
  reproduced faithfully, this is an oracle input-availability result, not a
  claimed reproduction of the Tu detector.
- **Carrier/code consistency:** report `q_i` in the truth domain first. An
  operational receiver threshold must be calibrated on independent normal
  receiver data before it can be called a detector result.
- **CGC:** retain the frozen complex nine-tap signed-delay estimator,
  `p_F <= 0.06028418845288192`, at least eight PRNs, and 3-of-5 persistence.

The target claim is complementarity across observables, not universal
superiority over Doppler methods.

## Evidence already available

The released five-geometry fresh-static experiment already contains matched
carrier-coupled and authentic-Doppler-locked signals with identical false code
trajectories.

- Carrier-coupled CGC: persistent detection in 5/5 geometries and median hold
  raw-alarm rate 0.944.
- Locked-Doppler CGC: at least one persistent detection in 4/5 geometries, but
  median hold raw-alarm rate 0.333 and median latency 15.5 s.
- The locked-Doppler result failed its preregistered stability and latency
  gates. It cannot support a claim that the present CGC delay sensor is
  Doppler-independent.
- A separate development audit showed that the carrier-coupled position hold
  retained 38.6--64.3 m median code offset while its maximum spoof/authentic
  carrier separation fell to 0.053--0.070 Hz.

These facts establish a useful mechanism and a real failure boundary. They do
not yet establish the final locked-Doppler WCL claim.

## Next experiment before manuscript promotion

The next work is a development-only cause audit, followed by a new frozen
confirmation:

1. Sweep relative carrier phase at fixed code trajectory and power balance.
2. Sweep small carrier offsets around exact lock while preserving the same code
   trajectory.
3. Measure signed-delay error, sign accuracy, template distance, CGC `p_F`,
   Tu-style carrier support, and truth-domain `q_i` separately.
4. Determine whether the locked failure is caused by fundamental coherent
   two-component ambiguity, receiver prompt ownership, or the current
   nearest-template estimator.
5. Improve the delay sensor only on development geometries. Freeze the
   estimator and thresholds before generating a new support-selected static
   confirmation.

If exact carrier lock makes signed delay non-identifiable for some phase/power
states, that boundary must be reported rather than tuned away. A defensible
letter can claim the region in which CGC adds position-hold coverage and state
the locked coherent blind region explicitly.

## Safe WCL claim, conditional on the next gate

> Cross-satellite code-delay geometry provides a displacement-domain
> complement to frequency-domain relative-velocity and carrier/code-rate
> checks: it can retain spoof evidence after rate observables collapse, within
> a measured correlation-aperture and coherent-observability region.

Until the locked phase/offset audit and fresh confirmation pass, the manuscript
must not say that CGC reliably detects arbitrary Doppler-locked carry-off.

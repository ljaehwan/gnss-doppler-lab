# CGC Doppler-Silent Hold Experiment Design V1

Status: **design and development plan, not a preregistered result**

## 1. Research question

After a physically coherent position carry-off reaches its target and stops,
does a code-delay geometry detector retain spoof evidence after the
primary--secondary Doppler separation required by a dual-frequency-peak method
has disappeared?

The intended claim is temporal complementarity, not universal superiority:

> A dual-Doppler method observes relative motion during pull-off, whereas CGC
> can observe the accumulated cross-satellite code displacement during the
> subsequent zero-relative-velocity hold.

Tu et al. require a resolvable primary--secondary frequency-domain dual peak
and use at least five such satellites for the relative-velocity residual.  In
their TEXBAT study, a Doppler difference above approximately 3 Hz is the
dual-peak operating regime.  This experiment will therefore report both the
Tu-style input-availability condition and the CGC decision; it will not label a
custom spectral probe as an exact reproduction of Tu et al.

Reference: <https://doi.org/10.1049/iet-rsn.2019.0366>

## 2. Physical hypothesis

For satellite (i), let the counterfeit-minus-authentic range be

\[
\Delta\rho_i(t) = -\mathbf{u}_i^T\mathbf{d}(t) + c_\rho(t).
\]

The signed code delay is proportional to displacement,

\[
\delta_i(t) = \Delta\rho_i(t)/L_{\mathrm{chip}},
\]

whereas carrier-Doppler separation is proportional to range rate,

\[
\Delta f_i(t) = -\dot{\Delta\rho}_i(t)/\lambda_{L1}.
\]

During pull-off, \(\dot{\mathbf d}(t)\ne 0\), so both observables can be
present.  After the false displacement becomes constant,
\(\dot{\mathbf d}(t)=0\): \(\Delta f_i\) collapses toward zero while
\(\delta_i\) remains nonzero.  The experiment tests this position-versus-
velocity distinction using an ordinary code/carrier-coherent signal.  It does
not use the code/carrier-decoupled stress intervention.

## 3. Development audit on retained RF

Use only the `carrier-coupled` members of the already released five-geometry
campaign at
`/home/ubuntu/hdd_data/cgc_code_carrier_fresh_static_v1`.

These data are now development data because their CGC results have already
been exposed.  They may establish feasibility and freeze the spectral
measurement implementation, but they cannot be reported as a new untouched
confirmation.

### Fixed intervals

- authentic baseline: \(0 \le t < 5\) s;
- physical pull-off: \(5 \le t < 10\) s;
- settling guard: \(10 \le t < 12\) s;
- stationary false-position hold: \(12 \le t < 30\) s.

### Existing design-basis facts

Across Tokyo, London, Sao Paulo, Sydney, and Nairobi:

- pull-off median absolute Doppler separation: 42.2--51.0 Hz;
- pull-off maximum absolute Doppler separation: 138.0--157.2 Hz;
- duration with at least five PRNs above 3 Hz: 4.8--4.9 s;
- hold maximum absolute Doppler separation: 0.053--0.070 Hz;
- hold duration with at least five PRNs above 3 Hz: 0 s;
- hold median absolute code offset: 38.6--64.3 m;
- existing CGC median hold raw-alarm rate: 0.944, with persistent detection in
  all five pairs.

These numbers motivate the confirmatory gates below; they are not themselves a
fresh result.

### Time-resolved outputs

For every one-second bin and every tracked PRN, record the truth, tracking,
and CGC quantities below. The more expensive actual-IQ delay--Doppler audit
uses six anchors fixed before its first execution: 4.5, 7.5, 8.5, 12.5, 20.5,
and 28.5 s. These provide one baseline, two pull-off, and three hold views.

Record:

1. truth primary--secondary carrier-Doppler separation;
2. blind actual-IQ dual-peak count and estimated frequency separation;
3. the number of PRNs satisfying the Tu-style dual-peak input contract;
4. CGC PRN support, Partial-F \(p_F\), raw alarm, and persistent alarm;
5. carrier lock and C/N0, to exclude receiver loss of lock as the reason for
   spectral collapse.

The truth-based Tu-style availability result is deliberately oracle-favourable:
if even the known separation is below 3 Hz, a two-frequency-peak residual
cannot obtain the required observable.  The blind IQ result is reported as a
mechanism check and must not be tuned to make the comparator fail.

The development IQ parameters are inherited from the earlier Seoul mechanism
audit: 25 MHz source IQ resampled to 5 MHz, 20 ms coherent integration, 10 Hz
Doppler grid, a receiver-tracking-centred plus or minus 250 Hz search, and
dominant peaks with normalized height at least 0.5, prominence at least 0.1,
and spacing at least 60 Hz. A 25 Hz local probe audits expected frequencies
after the blind search; it does not choose the search centre.

## 4. Confirmatory fresh-static campaign

After the development audit freezes the IQ spectral implementation, create a
new candidate pool of ten exact position--UTC geometries in five geographic
slots.  Commit the pool, algorithms, intervals, and terminal gates before a
support probe.  Select the first candidate in each slot with at least ten
startup LOS PRNs using a one-second LOS-only preflight.  The preflight may not
generate selected 25 MHz attack RF or compute either detector score.

### Signal and receiver

- five selected static geometries, each previously unused at its exact
  position and UTC;
- 30 s GPS L1 C/A recording at 25 MHz;
- attack onset 5 s and physically coupled transition duration 5 s;
- one fixed 100 m three-dimensional target offset per slot, with directions
  declared in the pool before support selection;
- counterfeit power ramp from -30 to +3 dB over the same 5 s transition;
- counterfeit code and carrier both follow the same false trajectory;
- matched authentic component, RF frontend, noise law, and 11-channel complex
  9-tap GNSS-SDR receiver;
- frozen CGC threshold `p_F <= 0.06028418845288192`, at least eight PRNs, and
  3-of-5 one-second persistence.

The earlier independent-multipath campaigns remain the specificity control.
This experiment isolates temporal coverage and does not replace those
multipath results.

## 5. Preregistered primary endpoints

The future fresh release will return `SUPPORTED` only if all of the following
hold without pair substitution or retuning:

1. **Truth integrity:** code and carrier use the same false range and range
   rate in all five pairs; all five truth audits pass.
2. **Pull-off Doppler availability:** every pair contains at least three
   consecutive one-second pull-off bins with at least five PRNs whose oracle
   absolute Doppler separation is at least 3 Hz.
3. **Hold Doppler unavailability:** every hold bin has fewer than five PRNs at
   or above 3 Hz, and the maximum oracle hold separation is below 0.5 Hz.
4. **Persistent displacement:** every pair has at least 25 m median absolute
   code offset during hold and remains inside the signed-delay aperture.
5. **CGC hold detection:** at least four of five pairs have persistent CGC
   detection and the across-pair median hold raw-alarm rate is at least 0.80.
6. **Complement interval:** at least four of five pairs contain at least ten
   consecutive hold seconds in which Tu-style input support is below five PRNs
   while the CGC persistent alarm is true.
7. **Clean behavior:** the total number of persistent CGC alarms before 5 s is
   zero.
8. **Receiver continuity:** every evaluated complement interval retains the
   preregistered minimum tracking support; loss of receiver lock cannot be
   counted as Doppler-method unavailability.

Blind actual-IQ dual-peak recovery is a secondary endpoint until its parameters
are frozen from the development audit.  It may strengthen the mechanism but
cannot override a failed primary gate.

## 6. Main figure

Use one compact two-panel vector figure:

- top: time versus number of PRNs with resolvable dual-Doppler evidence, with
  the five-PRN requirement and pull-off/hold boundary marked;
- bottom: time versus CGC \(p_F\) and the persistent alarm, with the frozen
  threshold marked.

Shade baseline, pull-off, guard, and hold.  The intended visual is a hand-off:
dual-Doppler support exists during motion and collapses after stopping, while
CGC remains active because the code displacement persists.

## 7. Interpretation boundary

A passing result would support only the following statement:

> In the tested physically coherent static carry-off signals, CGC extends
> detection into a stationary false-position hold where the dual-Doppler input
> required by a relative-velocity residual is unavailable.

It would not establish that CGC is universally better than Tu et al., that all
Doppler detectors fail, or that CGC detects arbitrary code/carrier-decoupled
spoofing.  A failed gate will be reported as `NOT_SUPPORTED` without changing
the threshold, intervals, satellite support, or pair roster.

## 8. Execution order

1. Implement the read-only development timeline on the retained five RF pairs.
2. Freeze and test the blind spectral measurement using development data only.
3. Commit the fresh candidate pool, final runner, and all gates.
4. Run LOS-only candidate selection.
5. Commit the selected release and execute it once.
6. Produce the vector timeline and update both Korean and English manuscripts
   only after the single-release decision is known.

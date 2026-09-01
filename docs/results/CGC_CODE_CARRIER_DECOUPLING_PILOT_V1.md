# CGC code/carrier decoupling pilot v1

## Status and claim role

Completed on the development-only Seoul static geometry. This run verifies the
implementation and the intended physical contrast. It is not a fresh
multi-geometry confirmatory result and must not be reported as one.

## Controlled contrast

Both conditions contain the same authentic RF, false 100-m code trajectory,
power ramp, noise/frontend realization, 25-MHz IQ format, and 9-tap receiver.
Only the counterfeit carrier trajectory changes:

- `carrier-coupled`: false code and false carrier trajectory;
- `doppler-locked`: false code trajectory but authentic carrier trajectory.

Truth preflight passed all exact invariants. The two counterfeit conditions had
identical code range and code rate, while the locked counterfeit carrier range
and rate were identical to the authentic reference. During the selected
8.3--8.7-s moving interval, the coupled carrier separation was 103.4--129.5 Hz
(median 119.4 Hz); the locked separation was exactly 0 Hz.

## Actual-IQ carrier observability

Twenty-millisecond PRN correlation profiles were evaluated at 10-Hz spacing for
G12, G24, and G26 over five times (15 windows per condition). A normalized
height of 0.5 was frozen as the strong-peak threshold.

- Coupled control: both truth-predicted carrier frequencies exceeded the
  threshold in 15/15 windows; blind peak finding returned two dominant peaks in
  15/15 windows. Median counterfeit-frequency height was 0.976.
- Doppler-locked target: the authentic frequency exceeded the threshold in
  15/15 windows, while the counterfactual coupled-spoof frequency did so in
  0/15 windows. Its median height was 0.132.
- Blind peak finding returned one dominant peak in 13/15 locked windows. The two
  exceptions were nearby splits of the authentic main lobe, not peaks at the
  truth-predicted coupled-spoof frequency, and are therefore retained rather
  than discarded.

The GNSS-SDR tracking loop provides a consistent receiver-level diagnostic over
the same interval: median carrier-lock statistic increased from 0.445 in the
coupled control to 0.974 under Doppler lock, while median reported C/N0 changed
from 34.34 to 46.65 dB-Hz.

## Code-domain CGC outcome

The frozen 9-tap signed-delay estimator, minimum support of 8 PRNs (10 were
available), Partial-F threshold `p_F <= 0.06028418845288192`, and 3-of-5
persistence rule were reused without retuning.

- Carrier-coupled: first persistent CGC alarm at 8 s, 3 s after carry-off onset;
  hold-interval raw alarm rate 100%.
- Doppler-locked: first persistent CGC alarm at 12 s, 7 s after carry-off onset;
  hold-interval raw alarm rate 61.1%.
- Both conditions had zero raw and zero persistent alarms in the five
  pre-attack one-second bins.
- Development AUC against the already-inspected matched Seoul multipath control
  was 1.000 for both conditions. This AUC is diagnostic, not fresh evidence.

The supported conclusion is narrow: removing the distinct counterfeit carrier
Doppler peak did not remove code-domain CGC evidence in this static development
geometry. This does not show that every Doppler detector fails, and it does not
replace fresh multi-geometry validation.

## Reproduction and artifacts

Run from the repository root:

```bash
.venv/bin/python scripts/run_cgc_code_carrier_decoupling_pilot.py --phase all
.venv/bin/python scripts/audit_cgc_code_carrier_doppler_observability.py
```

Large IQ and receiver outputs are stored under
`/home/ubuntu/hdd_data/cgc_code_carrier_decoupling_pilot_v1`. The compact audit
summary, per-window CSV, and vector PDF/SVG are under its
`doppler_observability` subdirectory. The simulator is pinned to upstream
commit `28ca29a6719475195e3aabd5930c4ed02d67190f` plus the repository patch
`patches/gps-sdr-sim-code-carrier-decoupling-v1.patch`.

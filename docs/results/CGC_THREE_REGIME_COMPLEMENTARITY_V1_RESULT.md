# CGC Three-Regime Complementarity Audit V1 Result

## Status

**MECHANISM SUPPORTED; LOCKED-HOLD CGC ROBUSTNESS NOT SUPPORTED.**

This is a read-only development reanalysis of the already released five-pair
fresh-static campaign. It generated no new RF, changed no CGC threshold, and
does not convert previously exposed data into a new held-out result.

## Main result

The same false code trajectory was evaluated with either carrier-coupled
consistent Doppler or carrier Doppler locked to the authentic signal. The
comparison separates three physical regimes rather than treating all carry-off
signals as if they necessarily contained a distinct carrier-frequency peak.

| Regime | Tu input-available bins | Median code offset | Median carrier separation | Median carrier-equivalent code/carrier mismatch | Median pair CGC raw-alarm rate |
|---|---:|---:|---:|---:|---:|
| Consistent-Doppler pull-off | 100% | 28.80 m | 60.70 Hz | <0.001 Hz | 0.600 |
| Locked-Doppler pull-off | 0% | 28.80 m | 0 Hz | 60.70 Hz | 0.200 |
| Coupled position hold | 0% | 57.57 m | 0.0316 Hz | <0.001 Hz | 0.944 |
| Locked position hold | 0% | 57.57 m | 0 Hz | 0.0316 Hz | 0.333 |

Here, Tu input availability is an oracle contract: at least five PRNs have
absolute spoof/authentic carrier separation of at least 3 Hz. It is not an
implementation of Tu et al.'s complete FFT-CDS and relative-velocity detector.
The code/carrier mismatch is also a truth-domain physical quantity,

\[
q_i=\Delta f_{c,i}-1540\,\Delta f_{\mathrm{code},i},
\]

not an operational receiver detector with a calibrated threshold.

## Interpretation

The results support the intended observable hand-off:

1. During consistent-Doppler pull-off, the dual-carrier input used by Tu et al.
   is available and the code/carrier mismatch is essentially zero.
2. During locked-Doppler pull-off, the dual-carrier input disappears while the
   code/carrier mismatch becomes large.
3. After position hold, both rate-domain quantities collapse while a 57.57 m
   median code displacement remains.
4. The frozen CGC detector remains strong in the ordinary carrier-coupled hold,
   detecting all five pairs with a 0.944 median raw-alarm rate.

The present signed-delay estimator is not robust enough under exact
authentic-carrier lock. The locked-hold raw-alarm rate is only 0.333 and the
underlying released experiment failed its preregistered stability and latency
gates. Consequently, the WCL manuscript must not claim detection of arbitrary
Doppler-locked carry-off.

## Next gate

Use development geometries to sweep relative carrier phase and small
spoof/authentic carrier offsets while holding the code trajectory and power
constant. Record signed-delay error/sign accuracy, prompt ownership, template
distance, Partial-F score, Tu input support, and `q_i`. Only after the cause is
identified and the delay sensor is frozen may a new support-selected static
confirmation be generated.

## Reproduction

From the repository root:

```bash
.venv/bin/python scripts/audit_cgc_three_regime_complementarity.py
```

Outputs:

- `artifacts/cgc_three_regime_complementarity_v1/summary.json`
- `artifacts/cgc_three_regime_complementarity_v1/regime_summary.csv`
- `artifacts/cgc_three_regime_complementarity_v1/timeline.csv`
- `artifacts/cgc_three_regime_complementarity_v1/three_regime_timeline.svg`

The source campaign summary is
`/home/ubuntu/hdd_data/cgc_code_carrier_fresh_static_v1/summary.json`, SHA-256
`f57162a2f504f6a9d224889956805e69d4ce3bdf3d19386e506bef7fc70813f7`.

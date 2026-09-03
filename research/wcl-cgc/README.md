# WCL CGC research package

## Paper question

When a first-stage channel monitor reports a distorted correlation peak, can a
single-antenna receiver distinguish coordinated carry-off from independent
multipath by testing the cross-satellite signed-delay geometry?

The executable chain is:

```text
complex nine taps -> signed delay -> causal median -> observability
-> clock-only versus LOS-plus-clock fit -> Partial-F -> persistence
```

## Folder map

| Folder | Contents |
|---|---|
| [`figures/`](figures/) | The two manuscript figures, their generator, sources, and the developer-only formula audit |
| [`equations/`](equations/) | Every main paper equation mapped to executable source and unit tests |
| [`simulation/`](simulation/) | Final frozen receiver--RF simulation, conditions, positions, and commands |
| [`data/`](data/) | Committed summaries versus local HDD/SSD IQ and receiver artifacts |
| [`experiments/`](experiments/) | Paper-core, supporting, development-only, negative, and superseded experiments |
| [`reproduction/`](reproduction/) | Clean-clone environment, retained-output replay, and full-RF rebuild boundary |

## Authoritative endpoint

- frozen config:
  [`../../configs/experiments/cgc_temporal_final_static_v1.json`](../../configs/experiments/cgc_temporal_final_static_v1.json)
- runner:
  [`../../scripts/run_cgc_temporal_final_static_v1.py`](../../scripts/run_cgc_temporal_final_static_v1.py)
- compact result:
  [`../../docs/results/CGC_TEMPORAL_FINAL_STATIC_V1.md`](../../docs/results/CGC_TEMPORAL_FINAL_STATIC_V1.md)
- compact machine record:
  [`../../docs/results/cgc_temporal_final_static_v1_summary.json`](../../docs/results/cgc_temporal_final_static_v1_summary.json)
- full local artifacts: `/home/ubuntu/hdd_data/cgc_temporal_final_static_v1`
- sealed full summary SHA-256:
  `b154e411f4447fb534c0e10dee45f2f14f84b0e4e4eeedaa771cfefaf721045d`

## Result boundary

The final endpoint contains five outcome-unseen static geometries and four
conditions per geometry. It supports static simulated receiver--RF evidence,
not a universal field detector claim. JammerTest, TEXBAT, and GNSS-OpenIF are
reported as separate public transfer audits and are not pooled with the final
endpoint.

The main code/claim map remains
[`../../docs/WCL_CGC_CODE_MAP.md`](../../docs/WCL_CGC_CODE_MAP.md). The formula
accuracy numerical result is used as a same-stream validation; its diagnostic
figure stays in this development repository and is not a manuscript figure.

# CGC RF transfer sweep v1 result

## Decision

`EXPLORATORY MAP COMPLETED`. The full preregistered 18-cell grid completed
without condition, interval, receiver, or estimator substitution. The result
does not support one universal distance boundary. Instead, it reveals separate
receiver states for geometry onset, reliable discrimination, and metric
saturation.

The config, protocol, runner, distance grid, constant pull-off rate, and common
comparison interval were committed in `b90016b` before outcome access.

## Primary transfer curve

All conditions used a 20 m/s carry-off, the same 18--29 s comparison interval,
the same 0.125-chip nine-tap GNSS-SDR receiver, and the same independent
multipath control. Every AUC compared 12 spoof bins with 12 multipath bins; the
minimum spoof satellite support was 10 PRNs.

| Distance | Chips | AUC -6 dB | AUC +3 dB | Norm error -6/+3 | Edge fraction -6/+3 |
|---:|---:|---:|---:|---:|---:|
| 20 m | 0.068 | 0.201 | 0.049 | +248% / +205% | 0% / 0% |
| 40 m | 0.136 | 0.090 | 0.257 | +47% / +52% | 0% / 0% |
| 60 m | 0.205 | 0.549 | 0.472 | +15% / +19% | 0% / 0% |
| 80 m | 0.273 | 0.854 | 0.417 | +5.9% / -3.8% | 0% / 0% |
| 100 m | 0.341 | 1.000 | 0.986 | +1.9% / +2.8% | 0% / 0% |
| 120 m | 0.409 | 0.986 | 1.000 | +4.5% / +3.5% | 0% / 0% |
| 160 m | 0.546 | 1.000 | 1.000 | -3.8% / -1.6% | 7.6% / 16.7% |
| 200 m | 0.682 | 1.000 | 0.986 | -10.9% / -15.9% | 14.4% / 18.0% |
| 240 m | 0.819 | 1.000 | 1.000 | -11.9% / -18.0% | 22.7% / 32.6% |

The first tested AUC 0.8 crossing was 80 m at -6 dB and 100 m at +3 dB.
Both power regimes stayed above 0.8 through the largest tested distance. The
distance--AUC Spearman correlations were 0.896 and 0.877, respectively, but
neither full curve was strictly monotone.

## Receiver observability states

The joint AUC, recovered displacement, direction, and edge diagnostics support
four descriptive states in this geometry:

1. **Unresolved mixture, 20--40 m.** AUC stayed below 0.26, absolute direction
   cosine stayed below 0.64, and displacement magnitude was dominated by
   estimator ambiguity rather than physical separation.
2. **Geometry onset, about 60 m.** Direction cosine reached 0.802 and 0.810,
   while AUC remained only 0.549 and 0.472. Recovering the physical direction
   therefore begins before reliable multipath-versus-spoof ranking.
3. **Discriminable and metrically faithful, about 100--160 m.** Both powers
   reached AUC 0.986--1.000, direction cosine 0.963--0.994, and displacement
   norm error within roughly 5%. At 80 m only the -6 dB condition crossed 0.8,
   showing that power shifts entry into this state.
4. **Discriminable but metrically saturated, 200--240 m.** AUC remained
   0.986--1.000 and direction remained strong, but recovered displacement was
   11--18% too small. Per-PRN delay estimates at the +/-0.5-chip template edge
   increased to 14--33%.

These ranges are post-outcome descriptions of this exploratory geometry, not
new validated thresholds.

## Physical contribution

The key result is that **detection observability and metric observability are
not the same receiver property**. A finite correlator bank can retain enough
cross-satellite direction structure to distinguish coherent carry-off from
independent multipath even after individual delay estimates begin to saturate.
Consequently, a detector may continue to alarm correctly while its estimated
spoof displacement becomes biased low.

This explains why the earlier ideal-profile 60 m boundary did not transfer
directly to receiver RF. At 60 m the common geometry becomes visible, but it is
not yet consistently lower-residual than multipath. Power-dependent prompt
ownership shifts the reliable crossing, while finite tap support later limits
metric recovery without immediately destroying classification.

The resulting paper direction is a receiver-level CGC observability state
diagram, not a single distance threshold. The template-edge fraction also
emerges as a possible self-diagnostic: it warns that localization is becoming
biased even when the spoof classification score remains high.

## Integrity and retention

All 18 composed signals had byte-identical normal prefixes. Each cell supplied
12 eligible bins per class. Eighteen campaign-created composed IQ files and
nine distance-shared counterfeit components were hash-checked and removed
after their receiver outputs and scores were durable. The existing authentic,
normal, and multipath inputs were not removed. All 27 removed files remain
deterministically regenerable.

The immutable raw result is
`artifacts/cgc_rf_transfer_sweep_v1/summary.json`, SHA-256
`7f15b76b48e176f1c1b90915614b7dbc6b04ba0e735da6dbf8dd284177ddb7eb`.
The generated three-panel transfer plot is
`artifacts/cgc_rf_transfer_sweep_v1/transfer_curve.png`, SHA-256
`7bd778ef206847382a63a10ee6da72d09b08df2ad308d8082b8f139ce48bdae9`.

## Claim boundary and required validation

This remains one previously used static satellite geometry and an exploratory
map. The four-state description, crossing distances, and edge diagnostic are
not yet a WCL-level general claim.

The next confirmatory experiment must freeze representative cells from all
four states and test them on new LOS geometries. The algorithmic improvement
should then hold the DLL discriminator at 0.125 chip while adding nonuniform
multi-scale auxiliary taps. Its preregistered goal is to reduce edge saturation
and displacement bias without degrading the low-separation crossing or
multipath false ranking.

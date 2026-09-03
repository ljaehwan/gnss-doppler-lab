# Data and artifact locations

Large IQ and receiver outputs are intentionally excluded from Git. Committed
configs and summaries pin their path, byte count, and SHA-256. Missing local
data does not invalidate the committed result record, but it prevents a full
rerun until the dataset is restored.

## Primary and controlled simulation

| Role | Local path | Committed provenance |
|---|---|---|
| Final five-geometry receiver--RF test | `/home/ubuntu/hdd_data/cgc_temporal_final_static_v1` | [`cgc_temporal_final_static_v1_summary.json`](../../../docs/results/cgc_temporal_final_static_v1_summary.json) |
| Fresh three-pair mechanism test | `artifacts/cgc_rf_fresh_test_v2` | [`cgc_rf_fresh_test_v2_summary.json`](../../../docs/results/cgc_rf_fresh_test_v2_summary.json) |
| Fresh source generation | `artifacts/simulation_v5_fresh_test_generation_v2` | pinned by the fresh-test config and summary |
| Geometry/tap-aperture campaign | `artifacts/cgc_rf_ga_v1` | [`cgc_rf_geometry_aperture_validation_v1_summary.json`](../../../docs/results/cgc_rf_geometry_aperture_validation_v1_summary.json) |

The final full `summary.json` SHA-256 is
`b154e411f4447fb534c0e10dee45f2f14f84b0e4e4eeedaa771cfefaf721045d`.

## Public transfer datasets

| Dataset | Raw/derived local location | Paper role and config |
|---|---|---|
| FGI JammerTest JT23-17.1.6 | `/home/ubuntu/hdd_data/jammertest2023/JT23_17.1.6/JT23_17.1.6_L1_E1.iq` (72,110,000,000 bytes) | public coherent-spoof transfer; [`jammertest2023_jt17_cgc_v1.json`](../../../configs/experiments/jammertest2023_jt17_cgc_v1.json) |
| TEXBAT DS1--DS3 exports | `/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds123-graph-input/exports` | spoof pre/post direction and latency, no independent multipath; [`cgc_texbat_external_v1.json`](../../../configs/experiments/cgc_texbat_external_v1.json) |
| GNSS-OpenIF S1 raw IQ | `/home/ubuntu/ssd_data/gnss-datasets/gnss-openif/raw/S1_suburban_HK.bin` (10,200,547,328 bytes) | real multipath-rich negative control; [`gnss_openif_s1_real_multipath_v1.json`](../../../configs/experiments/gnss_openif_s1_real_multipath_v1.json) |
| GNSS-OpenIF S2 raw IQ | `/home/ubuntu/ssd_data/gnss-datasets/gnss-openif/raw/S2_urban_HK.bin` (11,593,056,256 bytes) | second recording-level multipath negative; [`gnss_openif_s2_real_multipath_v1.json`](../../../configs/experiments/gnss_openif_s2_real_multipath_v1.json) |

## Raw-data identity

- JammerTest IQ SHA-256:
  `4bd6e3963f0b3d6806670db5d1a653de05fd9fe602ec42b005eb6cc4d45931e3`
- OpenIF S1 IQ SHA-256:
  `79952be6e4b8c36d0dee2841d837c731499aef2c8cdcf42890c5bb066007ff91`
- OpenIF S2 IQ SHA-256:
  `7dfb51d0973b45b86441f1c50db95079bbf47459fe73fab3ca3d0b6bdcf3a8c5`
- TEXBAT derived export hashes are recorded scenario-by-scenario in the frozen
  external config.

## What is committed

Git contains:

- experiment configs and frozen protocols;
- compact Markdown/JSON/CSV result summaries;
- deterministic source, analysis, figure, and test code;
- hashes that bind the summaries to local artifacts.

Git does not contain multi-gigabyte IQ, transient receiver dumps, or generated
intermediate component signals. Do not replace a local file under the same name
without checking its pinned SHA-256.

## Storage distinction

The HDD holds the 72.1 GB JammerTest record and final receiver--RF campaign.
The SSD holds TEXBAT-derived inputs and GNSS-OpenIF records used for faster
receiver replay. Paths are machine-local; the frozen config, not this prose,
is authoritative if a storage migration is later recorded.

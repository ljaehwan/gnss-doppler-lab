# Experiment roles

## Data--scenario matrix

| Experiment | Data used | Scenarios/conditions | Use in paper |
|---|---|---|---|
| Matched-profile mechanism | Deterministic synthetic complex 9-tap profiles; 6 train geometries x 300 events = 1,800 paired events, plus held-out validation geometries | LOS-consistent carry-off delays versus a matched control that deranges the same profiles across PRNs | Complex signed-delay/PRN--LOS mechanism AUC |
| Fresh receiver--RF v2 | Three 30-s 25-MHz simulated pairs: Vancouver static, Busan straight motion, and Perth parallel sweep | Independent PRN multipath versus carrier-coupled carry-off; identical front end and receiver | Held-out mechanism AUC and Fig. 1 example |
| Final static v1 | Five 30-s 25-MHz simulated geometries: Fairbanks, Punta Arenas, Casablanca, Sapporo, and Prince George | Clean normal, independent PRN multipath, carrier-coupled 100-m carry-off, and authentic-Doppler-locked 100-m code carry-off | Primary Table I and AUC 0.9801 |
| Formula accuracy v1 | The unchanged final-static v1 truth and receiver outputs | Exact range truth, fixed-startup LOS approximation, raw delay, five-bin stabilized delay, and recovered 3-D displacement | Same-stream equation validation; no new performance dataset |
| Geometry/tap aperture v1 | Denver static, Seoul static, Tokyo straight, London circle, and Sydney sweep; two powers (-6/+3 dB) | Carry-off distances 40/60/100/240 m; central 3/5/7/9-tap subsets; matched independent multipath | Aperture, metric band, and saturation limitation |
| JammerTest transfer | Public JT23-17.1.6 raw L1/E1 IQ | Clean/pre-motion interval and independently fixed coherent-spoof route motion at 526 s | Public carry-off-consistent transfer and latency |
| TEXBAT transfer | Public DS1--DS3 complex 9-tap exports; earlier released DS7/DS8 audits retained separately | Official spoof pre/post regions; no labeled independent-multipath class | Directional change and latency only |
| GNSS-OpenIF negatives | Public S1 pedestrian-near-building and S2 dense-urban vehicle raw IQ | Real multipath-rich recordings without paired spoof labels | Recording-level persistent false-alarm checks |

Exact locations, byte counts, and raw-data SHA-256 values are in
[`../data/`](../data/). Frozen configs remain authoritative for every seed,
position, UTC, onset, interval, and threshold.

## Paper core

| Experiment | Purpose | Status |
|---|---|---|
| Correlator identifiability train/validation | Isolate complex signed-delay and PRN--LOS information | Controlled mechanism support |
| CGC RF fresh test v2 | Three held-out receiver--RF multipath/carry-off pairs | AUC 1.000 mechanism evidence |
| CGC temporal final-static v1 | Five unseen geometries and four frozen conditions | Primary endpoint; all eleven release gates passed |
| WCL figure build | Reproduce both vector manuscript figures from pinned summaries | Paper asset |

## Paper support

| Experiment | Purpose | Qualification |
|---|---|---|
| Formula accuracy v1 | Compare range law, delay estimates, and recovered direction with simulator truth | Same final streams; not an independent test |
| RF geometry/aperture v1 | 3/5/7/9 taps, geometry and distance | Some preregistered aggregate gates failed; 7 taps post-hoc |
| JammerTest JT23-17.1.6 | Public coherent-spoof transfer | One independently labeled recording |
| TEXBAT external v1 | Public spoof pre/post direction and latency | No independent-multipath class |
| GNSS-OpenIF S1/S2 | Real multipath-rich negative controls | Recording-level specificity, not paired class accuracy |

## Development only

- code/carrier decoupling pilot;
- exact-lock phase sweep and phase root-cause audit;
- five-bin temporal stabilization and observability-gate selection;
- Doppler-silent hold and three-regime research-direction audits.

These experiments may explain why the final rule was chosen, but their reused
scores are not final generalization estimates.

## Negative or failed

- train-normal universal detector-freeze audit;
- FGI-SpoofRepo TGD post-takeover CGC attempt;
- first CGC real-detection input-support failure;
- locked-hold robustness part of the three-regime audit;
- Casablanca final multipath hard negative;
- Tokyo/+3 dB aperture ranking reversal.

These results are retained to prevent overstating the claim. A failed gate must
not be rewritten as a pass.

## Superseded

Earlier locked, fresh-static, state-validation, transfer-sweep, observability,
real-detection, and sequential-DS8 campaigns are historical development. The
five-geometry final-static result and separate public audits replace their
performance role; their protocols and hashes remain useful provenance.

## Machine-readable list

Every canonical script/config/result path and replacement link is recorded in
[`../../../configs/paper/wcl_cgc_v1_manifest.json`](../../../configs/paper/wcl_cgc_v1_manifest.json).
The manifest deliberately lists failures and superseded work instead of
deleting it.

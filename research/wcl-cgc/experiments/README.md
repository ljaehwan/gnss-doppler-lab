# Experiment roles

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

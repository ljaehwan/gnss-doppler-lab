# CGC temporal observable final-static test v1

## Outcome

The preregistered release decision is **SUPPORTED**. All eleven frozen gates
passed without pair substitution, threshold changes, or a rerun after outcome
access. The release was committed as `e578a09` before any selected 25 MHz
stream or CGC score was generated.

The test used five untouched static position/UTC geometries and four
receiver-RF conditions per geometry: clean normal, independent PRN-specific
multipath, carrier-coupled carry-off, and authentic-Doppler-locked code
carry-off. Counterfeit PRNs had deterministic independent initial carrier
phases. Coupled and locked attacks shared the same phase seed within a pair.

## Main results

| Metric | Result | Frozen gate |
|---|---:|---:|
| Truth-invariant pairs | 5/5 | 5/5 |
| Supported pair-condition streams | 20/20 | 20/20 |
| Carrier-coupled persistent detections | 5/5 | at least 4/5 |
| Doppler-locked persistent detections | 5/5 | at least 4/5 |
| Normal pairs with a persistent alarm | 0/5 | at most 0/5 |
| Multipath pairs with a persistent alarm | 1/5 | at most 1/5 |
| Spoof pre-attack persistent alarms | 0 | at most 0 |
| Median locked hold raw-alarm rate | 1.000 | at least 0.50 |
| Median locked latency from 5 s onset | 8.0 s | at most 12 s |
| Locked-hold versus multipath AUC | 0.9801 | at least 0.90 |

The Partial-F-only AUC was 0.9791 and the observable-gated value was 0.9801.
Across 90 hold bins, raw alarm rates were 100% for coupled attacks and 97.78%
for Doppler-locked attacks. Both attack classes had a 94.44% persistent-alarm
bin rate. Across 130 benign evaluation bins, persistent-alarm rates were 0%
for normal and 11.54% for multipath.

## Pair-level result

| Geometry | LOS | Coupled | Locked | Locked latency | Normal persistent FA | Multipath persistent FA |
|---|---:|---:|---:|---:|---:|---:|
| Fairbanks | 13 | detected | detected | 8 s | no | no |
| Punta Arenas | 12 | detected | detected | 8 s | no | no |
| Casablanca | 11 | detected | detected | 8 s | no | **yes** |
| Sapporo | 13 | detected | detected | 8 s | no | no |
| Prince George | 13 | detected | detected | 8 s | no | no |

Casablanca is an important hard negative: its independent multipath happened
to align sufficiently well with the LOS geometry and produced a 65.38% raw
alarm rate plus persistence. This is retained rather than tuned away. The
result therefore supports a strong mechanism and ranking claim, but not
perfect spoof-versus-multipath separation or a universal detector claim.

## Integrity audit

- All five code/carrier truth audits passed. Coupled and locked code range/rate
  matched; locked carrier range/rate matched authentic; and the realized code
  carry-off was nonzero and near 100 m at maximum.
- Authentic components omitted the phase option. Coupled and locked components
  used the same pair-specific `-P` seed, different across pairs.
- All multipath echoes stayed within the frozen 0.12--0.45 chip delay range and
  used independent PRN delay, amplitude, and phase.
- Normal/coupled/locked RF prefixes were byte-identical through the 5 s onset
  in every pair.
- Twenty GNSS-SDR receiver manifests were produced. Intermediate component and
  RF IQ files were hash-checked and removed after receiver success; retained
  receiver, truth, log, and analysis data occupy about 1.5 GB on the HDD.
- The focused regression and physics audit suite passed 25 tests.

## Artifacts

The immutable full result is
`/home/ubuntu/hdd_data/cgc_temporal_final_static_v1/summary.json` (SHA-256
`b154e411f4447fb534c0e10dee45f2f14f84b0e4e4eeedaa771cfefaf721045d`).
The signed-delay, stabilized-delay, and geometry-score CSV hashes are recorded
in `docs/results/cgc_temporal_final_static_v1_summary.json`.

## Paper interpretation

This is the first untouched receiver-RF test of the development-selected W=5
observable CGC candidate. It provides useful WCL evidence that code-domain
cross-satellite geometry remains detectable when counterfeit carrier Doppler
is locked to authentic and counterfeit PRN carrier phases are independent.
The correct scope remains static simulated receiver-RF validation. The
Casablanca hard negative and the absence of field/moving validation must remain
explicit limitations.

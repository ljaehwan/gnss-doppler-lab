# ACAF-NF Stage-1 static feasibility

This additive Stage-1 is deliberately fail closed. Its authenticated preflight
must hash complete raw recordings and inventory tracker MAT files, but it must
not read attack IQ for CAF construction or compute any attack score, metric, or
plot when frozen R1.4 support is absent. The expected production conclusion is
`FOUNDATION_INVALID`: this is not a physics `NO_GO` and cannot support a
detection claim.

## Frozen contract

The signal is canonical GPS L1 C/A sampled at 25 MHz as signed int16
interleaved complex IQ at global raw offset zero. Each authenticated support is
exactly 25,000 samples. NCO and auxiliary state come from the previous tracker
row, remnant and carrier signs are -1, the replica runs forward, and Prompt is
current. The dense grid is delay -1..+1 chip in .125-chip steps and Doppler
-250..+250 Hz in 50-Hz steps. H1 excludes the center. A window contains exactly
20 causal, same-channel/same-PRN, consecutive 1 ms supports. Adjacent sample
deltas must be 24,999..25,001 with CN0 >= 28 and carrier lock >= .85. Decimated
~20 ms or ~500,000-sample rows are unavailable and are never interpolated.

`SignalSource.samples` is a count limit, never an offset. Skip, header, and
offset keys are forbidden and the first file sample is raw sample zero. Under
the receiver's `ishort` item semantics, cleanStatic's configured count is
exactly 24,008,196,096 scalar int16 items (`raw_size_bytes/2`), covering the
whole file; its complex-IQ sample count is `raw_size_bytes/4`. A configured
count of zero means all source items and is valid for the other scenarios.

For each complex CAF `C`, normalization is

`Y = C exp(-j angle(C[zero Doppler, zero delay])) / max(|C_center|, floor)`.

Here `[0,0]` denotes the zero-delay/zero-Doppler center, not the array corner.
The normalized center is deterministically `1+0j` for eligible epochs. The
center coordinate is excluded from variance estimation and both quasi-WLS
fits. The raw center complex value, magnitude, floor, and floor-use flag remain
explicit diagnostics; floor-applied epochs fail the fixed clean-only quality
gate and are ineligible.

## Normal-only model

Chronological, raw-disjoint cleanStatic train/calibration/holdout roles are the
only source for variance, model, penalty, threshold, and pooling selection.
Attack labels cannot enter fitting, calibration, or selection; an attack may
only contribute pre-onset FPR after the foundation passes.

With fixed diagonal variance `v` learned from normal train, complex diagonal
quasi-WLS (not covariance whitening) fits

`H0: Y = alpha T0 + epsilon`

`H1(delta): Y = alpha T0 + beta Tdelta + epsilon`, `delta != (0,0)`.

`alpha` and `beta` are unconstrained complex amplitudes. RSS is the sum of
`|residual|^2/v`. H1 selects the non-center delta with minimum RSS and therefore
must have `RSS1 <= RSS0`. The saved `raw_s2src = RSS0-RSS1` is diagnostic only.
The primary statistic is the full minimized delta search. Search multiplicity
is therefore learned empirically on untouched clean calibration data; the
frozen scalar penalty is zero and is identical at calibration and inference.
Training (or a separate selection split) chooses model and pooling. Calibration
only estimates the transform and q99/q99.5/target-1% thresholds; holdout is
evaluation-only.

Fixed comparison selectors are power-only, Prompt magnitude, complex EPL,
fixed nine complex delay taps, dense one-source residual, and dense two-source
score. B0 is `PROVISIONAL_UNAVAILABLE` unless its exact evaluator interface,
support, and threshold lineage can be authenticated. No comparison is invented.
Median, top-50% mean, and trimmed mean are permutation invariant; the primary
pool is selected on normal calibration stability/FPR only. PRN count and score
dominance are always diagnostics.

Raw-IQ controls cover gain, global phase, AWGN/noise floor, and amplitude.
Second-source fixtures are analytic same-PRN fractional-code-phase replicas
with residual Doppler, arbitrary phase, amplitude, and positive or negative
delay. Finite zero-padded shifts and additions at CAF peaks are prohibited.
Every physical control is finite and range checked. H1 templates are built by
re-correlating authenticated cleanStatic complex raw intervals, not accepted as
caller-supplied surfaces. Their digest covers the complex surface bytes,
coordinate, source role, construction method, and canonical deep-frozen JSON
provenance, including exact interval and grid hashes.

Bootstrap blocks are frozen at exactly 10 seconds and use an explicit finite
recording/phase origin: `floor((time-origin)/10)`. Supplied block IDs must equal
that derivation exactly, and the origin is recorded.

## Fail-closed artifact semantics

The producer rejects an existing destination, builds in a sibling staging
directory, and publishes with Linux `RENAME_NOREPLACE`. In `FOUNDATION_INVALID`, all science CSVs
are header-only; model, threshold, and bootstrap JSON are `NOT_EVALUATED`; and
`plots/` is empty with its reason recorded in execution validity. There are no
placeholder plots or stale metrics. Raw-byte reads are recorded as
`full_sha256_only`, and `no_attack_raw_scoring_performed=true`.

The standalone verifier does not import the producer or its verdict logic. Its
pinned declarative source-binding configuration freezes exact tracker MAT
filename/SHA inventories and exact manifest JSON pointers. It recomputes
recursive checksums, full raw hashes, strict MAT schemas/support/L20 counts,
timelines, empty-science semantics, R1.4 lineage, and the foundation conclusion.
It also independently reruns the exact focused Stage-1 and relevant Stage0-R1.4
pytest commands with one BLAS thread and compares commands, source HEAD,
dependency versions, exit status, and semantic collected/pass/fail counts to
the checksum-covered producer report. Only an external verifier
`--finalize` pass may write PASS and regenerate checksums; a second read-only
pass validates that closure. Finalization first copies pending input into a
private staging tree; both semantic passes operate on that copy before an
atomic no-replace rename. `--skip-external-recompute` is diagnostic
`INCOMPLETE` and cannot finalize. DS7 and DS8 pre-attack
data are paired replay diagnostics—not independent normals—if byte identity is
authenticated.

The official timelines are DS3 onset 118.9 s/pull-off 195 s; DS4 onset 113.8
s/pull-off 225 s (raw ends about 128.22 s); and DS7/DS8 injection 110 s,
transition 110–130 s, held 130–150 s, and time-push from 150 s.

`PHYSICS_FEASIBILITY_GO=false` and `PAPER_CANDIDATE_GO=false` mean not evaluated.
Stage-2 is not justified until a new, independently validated continuous 1 ms
tracker/source binding exists.

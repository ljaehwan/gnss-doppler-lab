# TRACE Stage-0 R2 native 1 ms receiver dump

TRACE-R2 instruments the authenticated GNSS-SDR tracking receiver. It does not
change the frozen TRACE/R1 score, threshold, pooling, ablation, scenario-gate,
or verdict code.

## Receiver source and isolation

The receiver used by the earlier real-data runs is GNSS-SDR at base commit
`1ddd4562723040fd66cb334b578a5b69455625f4`. Its actual source worktree at
`/home/ubuntu/build-gnss-sdr-complex9` is detached and dirty with the earlier
complex-nine-tap patch. It was audited but never cleaned, reset, or modified.
R2 uses a clean worktree from the same commit under the external SSD artifact
root. The complete patch is retained as `receiver_patch.diff` in the Git
artifact bundle because no authenticated writable receiver fork is configured.

## Receiver execution and causal mapping

For each valid GPS L1 C/A tracking update, the patch performs this order:

1. Snapshot the DLL/PLL/NCO and code/carrier phase state used by the current
   correlation.
2. Correlate the current raw interval to produce original complex E4, E3, E2,
   E, P, L, L2, L3, and L4 values. No Prompt normalization is performed in the
   receiver dump.
3. Run the real receiver DLL/PLL discriminators and filters.
4. Update the receiver code/carrier NCO and phase state for the next interval.
5. Write the current observation, current action snapshot, computed-next action
   snapshot, and the loop sequence that generated the current action.
6. Consume the computed next buffer length. On the next loop this exact action
   is snapshotted before correlation.

The validator requires exact equality between row `t`'s computed-next action
and row `t+1`'s current-used action, requires the explicit source sequence link,
and requires row-start displacement to equal row `t`'s computed next buffer
length. Thus the default tuple is directly authenticated as
`(complex_t, action_computed_for_next_interval_t, complex_t+1)`.

A tracking-session ID and sequence reset prevent a channel reassignment from
joining two PRNs. The first action of each acquisition has the sentinel source
sequence `UINT64_MAX`; no causal pair crosses that boundary.

## Binary schema

Each channel file starts with a 192-byte little-endian header and contains
416-byte fixed records. The header binds schema version, sample rate, scenario,
base receiver commit, integration convention, tap spacing, and tap offsets.
Each record contains absolute raw sample bounds, receiver time, channel, PRN,
tracking session, loop sequence, validity/boundary flags, complex nine taps,
real receiver discriminator/lock/C/N0 values, and separate current-used and
computed-next action structures.

`SignalSource.seconds_to_skip` resets the flowgraph counter. Therefore smoke
configs explicitly set `Tracking_1C.trace_raw_sample_offset`; the receiver adds
that value to every sample stamp so DS3/OS3 rows remain in the original
recording timeline. The config, raw sample range, raw SHA-256, receiver binary,
and output hashes are recorded in external manifests.

GPS L1 C/A taps are carrier-wiped complex correlators without navigation-bit
wipeoff. The boundary flag identifies data-symbol rollovers. Prompt-referenced
normalization occurs only in the lab adapter. Since it removes global carrier
phase, the frozen R1 adapter retains carrier action as a predictor feature but
does not apply a second full-Doppler phase rotation to normalized taps.

## Fail-closed Phase A

Phase A uses only short TEXBAT cleanStatic, TEXBAT DS3, and OAKBAT OS3 slices.
Full replay is forbidden unless all three have continuous native cadence,
four-PRN support, exact causal links, finite non-placeholder observations and
actions, a common schema, deterministic reproduction, safe reassignment, and
authenticated raw-source/sample-range binding.

The allowed failure labels are:

- `NATIVE_1MS_DUMP_IMPLEMENTATION_INVALID`
- `ACTION_MAPPING_UNRESOLVED`
- `INSUFFICIENT_MULTI_PRN_SUPPORT`
- `RAW_SOURCE_BINDING_FAILED`

Any such failure produces `INCONCLUSIVE_INPUT_OR_RECEIVER` and no attack score,
threshold, performance plot, or detector claim.

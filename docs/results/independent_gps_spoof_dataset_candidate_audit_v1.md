# Independent GPS spoofing dataset candidate audit v1

## Decision

An additional dataset is required for the final confirmatory claim. The locally
available attack batteries are either already opened or do not meet the frozen
CGC support rule. No existing local recording qualifies as a new blind test.

The next released action is a score-free bounded-prefix support preflight of
TUNI2025 SS-29. SS-29 is the strongest current candidate because it combines
real received GPS L1, environmental multipath, authentic satellites, and four
documented true-position spoofers in the same RF stream. Its detector score has
not been accessed in this project.

## Frozen support rule

The current external model requires one-second bins with at least 200 tracked
epochs for each included PRN, at least 8 simultaneous eligible PRNs, and at
least 60 primary bins for a full confirmatory recording. The clean-only audit
does not load the delay template or compute any detector score.

| Local clean source | Prior role | Exact-rule maximum N | Primary bins | Outcome |
|---|---|---:|---:|---|
| TEXBAT cleanStatic | opened development control | 1 | 0 | insufficient under the current 200-epoch rule; its retained CSV/pipeline is not the prospective receiver contract |
| OAKBAT cleanStatic | opened external control | 4 | 0 | insufficient support |
| GNSS-OpenIF S1 | opened real-multipath negative control | 11 | 3 | geometry exists briefly, but not for the required 60 bins |
| TUNI GPS C-5 | opened TUNI clean control | 4 | 0 | insufficient support |

Machine-readable evidence is in `clean_geometry_support_audit_v1.json`. Local
raw holdings are 10 TEXBAT files (400,514,694,512 bytes), 8 OAKBAT files
(76,800,000,000 bytes), and 4 downloaded TUNI GPS core files
(119,999,328,000 bytes). All associated attack scenarios have already been
opened by prior experiments.

## Fresh candidate triage

### TUNI2025 SS-29 — proceed to support-only preflight

- Official record: <https://doi.org/10.5281/zenodo.17250054>
- Scenario: GPS L1 authentic plus spoofed signals, multipath present, four
  true-position spoofers.
- Documented spoofed PRNs: 1, 2, 21, and 32.
- Full raw file: 29,999,832,000 bytes; MD5
  `f821028e530bf3e56a24ef99f76977cd`.
- The official README and scenario file have been stored on SSD and checksum
  verified. The raw IQ has not been accessed.
- A committed protocol permits only the first 2,000,000,000 bytes (10 seconds)
  and only PRN/epoch/tap-schema outputs. Delay estimation and detector scoring
  are forbidden.

If the prefix contains N>=8 in at least 3 of 5 startup bins and every documented
spoofed PRN is eligible in at least 3 bins, the full file can advance to a
separately preregistered one-shot detector test. Otherwise SS-29 terminates as
insufficient support.

### TUNI2025 SS-27/SS-28 — reserve follow-ups

These are unopened multipath plus one- and two-spoofer recordings from the same
official collection. They remain reserve candidates, but their lower spoofed
PRN counts make target acquisition less likely than SS-29. They must not be
opened until SS-29 reaches a terminal support decision.

### TUNI2025 SS-33 — positive control only

SS-33 is a delayed all-PRN spoofing recording. It is useful as an all-satellite
positive control, not as the core partial-spoofer-versus-multipath test. Its
40 GB raw file also exceeds the currently available 36 GB SSD headroom.

### Jammertest 2025 — reject for CGC tracking

- Official record: <https://doi.org/10.5281/zenodo.21332689>
- Metadata was stored at commit
  `3b778a12147ded5c86c3edfc586b5de6ae6a67d7` without downloading LFS payloads.
- The LFS pointer set totals 358,810,497,577 bytes, which exceeds local free
  space.
- Each Innosense item contains 263,120 interleaved scalar I/Q values. At 81 MHz
  this is about 1.62 ms per independently indexed item; the CRPA snapshots are
  documented as 10 microseconds. These are appropriate for waveform or
  interference classification, but they do not provide the continuous
  multi-second tracking stream required for nine-tap PRN geometry.

Jammertest therefore remains a potential future spectral baseline and is not a
candidate for the present CGC confirmatory claim.

## Storage constraint

The SSD currently has approximately 36 GB free. That is enough for the bounded
2 GB SS-29 support prefix, but a complete 30 GB download plus receiver outputs
would leave unsafe headroom. A full download is intentionally deferred until
the score-free prefix passes and storage is reclaimed or expanded.

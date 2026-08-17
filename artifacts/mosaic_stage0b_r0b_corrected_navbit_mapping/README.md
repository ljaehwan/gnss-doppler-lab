# MOSAIC Stage-0B R0b corrected NAV-bit mapping

Verdict: **CORRECTED_BOUNDARY_STRUCTURE_FAIL**. Stage-0B injection is not authorized because direct boundary flags are absent for pre-sync validated bits; no injection or detector/model experiment was run here.

Every frozen validated bit retained its dataset, PRN, logical value, polarity, subframe/word position, parity, preamble, TOW, and sequence. Only the window moved from `[frozen_start, frozen_end]` to `[frozen_start+1, frozen_end+1]`; raw endpoints were copied from the actual next TRACE rows rather than adding a nominal 5000/25000 samples.

Committed evidence contains 126000 rows: one preceding boundary row plus the corrected 20-epoch window for every bit. All corrected windows and endpoints were preserved, but only 854/6000 previous rows and 861/6000 corrected ends carry an explicit boundary flag; 5146 bits are pre-sync flag gaps. All corrected starts and internal epochs do not carry flags, sessions/PRNs and loop sequences are continuous, NCO span/join variation is preserved, and endpoint transcription errors are zero.

Corrected Prompt agreement is 6000/6000 with the frozen decision axes and polarity. Frozen GPS structure independently remains 200/200 parity words, 20/20 preambles, 10/10 TOW/subframe continuity, distribution 1=2986 and 0=3014, with ten unique PRN sequence hashes.

- OAK corrected common interval: `[150275296, 210202273)`, 11.98539540 s.
- TEX corrected common interval: `[817815304, 1117517038)`, 11.98806936 s.

All five PRNs are simultaneously available in each common interval. Injection outside the intersection is forbidden; starts must also avoid recorded NAV transitions. All 11/11 in-memory tamper-negative cases were rejected.

Scope remains one contiguous approximately 12-second interval per PRN with two consecutive subframes: `two_separated_intervals=false`, `distant_interval_validation=NOT_PERFORMED`. This is not a MOSAIC hypothesis/model GO.

Test record: 39 Stage-0A/R0/R0a/R0b-focused tests passed. The full suite reported 560 passed and the same six pre-existing missing `scripts/train_peak_floor_temporal_autoencoder.py` failures; R0b added no failure. The committed verifier is also exercised in a fresh-clone procedure.

Next action: acquire or export authenticated boundary evidence covering the pre-sync 12-second interval, or define a separately approved modulo-phase extrapolation contract; do not inject with this R0b artifact.

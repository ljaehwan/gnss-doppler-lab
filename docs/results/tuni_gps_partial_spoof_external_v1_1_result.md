# TUNI GPS partial-spoofer external result v1.1

Status: sealed support failure on 2026-08-29.

The preregistered real-RF run cannot support a spoof-detection conclusion.
The independently audited terminal decision is INSUFFICIENT_SUPPORT. This does
not validate or falsify the frozen physical statistic on partial spoofing.

## Execution boundary

- Base preregistration commit: 9827b313abebc26e071563512d11f6fe1df5133d.
- Pre-attack v1.1 amendment commit: 779a4ca37b559c08f52b8baee6da74446ade37e2.
- SS-17, SS-18, and SS-20 were first opened only after the amendment was pushed.
- No receiver retune, threshold change, exclusion, or post-attack retry was made.

## Receiver completion

| Recording | Valid receiver PRNs | Tracking epochs | Documented spoofed PRNs |
|---|---:|---:|---|
| C-5 | 8 | 613,800 | none |
| SS-17 | 4 | 333,212 | 1 |
| SS-18 | 5 | 528,488 | 1, 2 |
| SS-20 | 6 | 604,152 | 1, 2, 21, 32 |

All three attack files passed their frozen MD5 checks and reached physical EOF
with receiver return code zero.

## Eligible one-second support

| Recording | Delay rows | Eligible PRNs | Max PRNs/bin | N>=8 bins | N=7 bins |
|---|---:|---|---:|---:|---:|
| C-5 | 284 | 5, 7, 23, 27, 29 | 4 | 0 | 0 |
| SS-17 | 123 | 2, 16, 18, 26 | 2 | 0 | 0 |
| SS-18 | 214 | 16, 18, 30 | 3 | 0 | 0 |
| SS-20 | 195 | 2, 16, 18, 21, 32 | 5 | 0 | 0 |

No recording produced even one primary or seven-PRN secondary bin after the
fixed 200-epoch-per-PRN requirement. Therefore no geometry-score row exists.

Documented target support also failed:

- SS-17: PRN 1 was absent from eligible delay bins.
- SS-18: PRNs 1 and 2 were absent from eligible delay bins.
- SS-20: PRN 1 was absent; PRNs 2, 21, and 32 were present.

## Terminal decision

Applying the frozen support gates to the 816 delay rows gives:

INSUFFICIENT_SUPPORT

The primary-bin minimum is 60 per recording; every recording has zero. The
all-documented-target requirement also fails. No sensitivity, specificity,
AUC, partial-F alarm rate, or detection claim is scientifically estimable.

The runner completed all receiver and template-analysis loops, then raised
FileNotFoundError while hashing geometry_scores.csv because its generic CSV
writer intentionally emits no file for zero rows. The missing file is a
serialization symptom of zero geometry support, not a lost positive result.
The runner was not patched or rerun after attack access.

## Sealed artifacts

- SSD root: /home/ubuntu/ssd_data/gnss-early-detection/artifacts/tuni-gps-partial-spoof-external-v1-1
- release_state.json: SHA-256 30f6a32305b48b9f43697aa018b28048bb8c24ad08db3b8739a331a59153ab07
- delay_estimates.csv: 816 rows, 67,132 bytes, SHA-256 afb78f2e35338f17d3394c6032bc8c0e469eade85093ead7f80207f1d1d51512
- SS-17 36-file tree: 3f7b56c7027560ff2a017f7b9a16c32c35084b63dc409b60a60e8e9a2a330aaa
- SS-18 36-file tree: 940ecdb0ce437644d76e858575de8df17caf8f005784318b3a77f4a6bc0b3d25
- SS-20 36-file tree: 57859c6ff70153861823586561b3cc07bff53a0a7eb9ca142bf795df7e56e5f2
- geometry_scores.csv and summary.json are absent by the recorded serialization failure.

## Interpretation and next valid use

This run establishes a receiver-support boundary: distinct tracked PRNs in a
manifest do not imply enough simultaneous, 200-epoch nine-tap profiles for an
N>=8 geometry statistic. The present TUNI files can be used only for exploratory
receiver-support engineering after this result.

A confirmatory v2 must freeze a stronger clean-first simultaneous-support gate
and use an independent sealed attack dataset after receiver settings are fixed.
The already opened SS-17/18/20 files must not be relabelled as a new blind test.

# Q-SET-GNSS Stage-0A partial-PRN quantile feasibility

This branch is isolated from base `8025f331444e600bee97baf28ad3cb7af9410381`.

## Dataset preflight result

The two registered local Tuni2025 directories do not exist. No raw I/Q file was
opened, hashed, decoded, mmaped, or downloaded. The experiment therefore stops
at the preregistered dataset-availability boundary with
`BLOCKED_TUNI2025_DATASET_NOT_LOCAL`.

Official metadata inspection found two separate Zenodo concepts. The user-cited
concept `10.5281/zenodo.15470142` contains Galileo E1 scenarios and supplies the
requested 1/3/5-spoofed-PRN design. The descriptor's GPS concept
`10.5281/zenodo.15572975` supplies GPS L1 scenarios, but its no-multipath attack
files contain 1/2/4 spoofed PRNs, not 1/3/5. Its C-5 and C-7 clear-sky raw files
also have the same official MD5 and are not independent payloads.

The exact requested 1/3/5 design would require five Galileo raw files totaling
141,925,248,000 bytes. That choice requires a separately frozen Galileo E1
receiver-support plan; it cannot silently reuse the current GPS L1 C/A contract.
No scientific preregistration, implementation freeze, clean scoring, synthetic
control, attack evaluation, or Stage-0B authorization was performed.


# ACAF-NF Stage-1 R2 full-normal

Corrective Checkpoint 2 explicitly supersedes `b5d53fa`: that commit incorrectly read the authenticated original R1 tracker CSV instead of the fresh exporter replay. The original CSV is not relabeled or used here. Fresh cleanStatic MAT/DAT files are authenticated against their replay manifest and reconstructed with clean-only alignment. Corrected status: `CONTINUOUS_TRACKER_INVALID` / `FOUNDATION_INVALID`. Attack IQ was not scored and Checkpoint 3 physics is not authorized when foundation is invalid.

# CRID Stage-0 R4d TEXBAT DS3 empty-TRACE repair and exploratory locked-score audit

Final verdict: `EXPLORATORY_TEXBAT_DS3_NO_USEFUL_SIGNAL`

The versioned repair is limited to the native TRACE manifest adapter. Existing zero-record channels are recorded as `EMPTY_OPTIONAL_CHANNEL`, with logical payload size zero plus the actual container size and SHA256. Missing expected files still fail closed. All empty channels remain in the output-set aggregate but are excluded from tracking/support counts.

The historical R4c C0 output was preserved byte-for-byte. Its dumps, config, and log were recovered, but receiver exit code and terminal-drain status were not independently recoverable; it remains historical incomplete evidence and was not reused. C0-C3 were rerun sequentially in the separate R4d output root. All 4 replay manifests passed exit, terminal-drain, native TRACE, target tracking, and provenance checks. Each configuration had one empty optional channel and ten tracked target PRNs.

The frozen locked model, causal alignment, score, threshold `-21.942672917134093`, timeline, metric, and gate were unchanged. Deterministic scoring matched exactly over 73,480 supported epochs. The exploratory gate failed: pre-onset FPR was 0.05015111033244273, pAUC at FPR<=0.05 was 0.5110177270584036, and established detection rate was 0.05365174555579353. Technical replay/alignment/support and the lock-loss/tracked-PRN shortcut condition passed.

This is exploratory locked-model evidence only. R4b Phase A did not pass; this run is not formal Phase B, confirmatory detector validation, or deployment evidence. Only TEXBAT DS3 was accessed. DS1, DS4, DS7, DS8, and OAK attack data remained untouched; DS7/DS8 remain independent future holdouts.

# CMTE-A2 TEXBAT validation report

## Status

**PRIMARY INVALID / NO-GO.**

Preregistration commit: `e7cb2e5822923a129d72c475706f87721ddd8104`
Frozen execution commit: `71d00f310b6152868b2e02df2ca955cfecd43eb3`

The one-shot DS7/DS8 campaign was executed only after preregistration and trust-anchor freeze. Post-run validation found that exact floating timestamp equality was used for epoch grouping. All stored DS1–DS4/DS7/DS8 epoch rows had `tracked_prn_count=1`, so the intended multi-PRN mean evidence was not implemented in the sealed primary. The primary and its confidence intervals remain preserved for audit and are not replaced post hoc.

See `artifacts/cmte_a2_texbat_ds78/README.md` for chronological B0 evidence, source hashes, sealed metrics, limitations, success-criteria audit, and claim boundaries.

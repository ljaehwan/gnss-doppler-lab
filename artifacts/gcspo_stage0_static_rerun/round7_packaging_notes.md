# Round-7 successor packaging notes (evidence only)

This document is administrative packaging evidence. It is not part of the
frozen Stage-0 scientific contract, does not authorize protected access, and
does not replace `README.md`. The frozen `README.md` is restored byte-for-byte
at SHA-256 `eea2e10885d66bfc762f33b2e25147ab07b1bbceace505078e8770e4cdc18ac2`.

Round 6 packaged three clean-static full-workload reproductions created and
signed by the external controller against source commit
`9774fe1048e467808b53769f94a717507fac5a38`. The bounded public package is
indexed by `round6_a5_provenance.json`, hashed by
`round6_evidence_manifest.json`, compared in `round6_a5_parity.json`, and
summarized in `round6_audit_report.json`.

The two CUDA runs are byte-identical. The CPU run is within the unchanged
Round-4 preregistered absolute/relative tolerances. `cleanStatic` alone is gate
evidence; `cleanDynamic` remains OOD diagnosis. Round-5's unsigned failed root
remains rejected, immutable, and excluded.

Independent review rejected freeze `532c4dd014432b787553b199a93f01ddaa294c01`.
The exact-freeze 212-pass claim is preserved as historical but marked
superseded and rejected in `round6_independent_review_rejection.json`. This
successor repairs only the administrative README freeze and the fail-open
non-finite CPU/CUDA verification path. It does not change signed evidence,
scientific outputs, configuration, thresholds, scoring, gate scope, tolerance
values, tolerance OR semantics, or field eligibility.

Protected execution and push remain prohibited. Protected rows and bytes
opened are 0/0; attack runs are 0; no marker or ledger exists.

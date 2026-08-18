# MOSAIC Stage-0B R1b limited multi-PRN root-cause audit

This directory contains a post-hoc diagnostic audit of the frozen R1a
multi-PRN failures.  It is not detector development, does not rerun injection
or receiver replay, and cannot alter the R1a verdict
`NO_GO_MOSAIC_MULTI_PRN_RECOVERY`.

`root_cause_preregistration.json` freezes the hypotheses, coordinate signs,
comparators, diagnostic labels, decision rules, and recommendation truth table
before diagnostic calculation.  Result files are generated only from retained
case results, native 1 ms complex nine-tap TRACE dumps, and their clean TRACE
references.


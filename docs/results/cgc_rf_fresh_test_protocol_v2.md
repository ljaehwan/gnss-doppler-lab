# CGC RF fresh locked test protocol v2

## Frozen question

Does the already frozen clock-centered geometry residual rank carry-off spoofing above independent per-satellite multipath on three new 30-second receiver-RF pairs selected only for startup satellite support?

The candidate formula, complex nine-tap template estimator, binning, comparison boundary, aggregation, random seeds, and support gates are unchanged after release. No absolute threshold is fitted or applied.

## Data and release boundary

The three rows are copied exactly from the committed support preflight: `fv2-static-01`, `fv2-straight-01`, and `fv2-sweep-01`. Their preflight startup LOS counts were 13, 13, and 11. The preflight accessed only a one-second simulator LOS table; it generated neither multipath nor spoof data and computed no CGC outcome.

The config, this protocol, and the conforming runner must be committed and clean before the release token is accepted. Release creates both output roots and irrevocably fixes the three-pair roster. A technical resume is permitted only before any metrics have been emitted. Pair substitution, parameter tuning, and retest after metrics are forbidden.

## Frozen signal and receiver contract

Use the pinned simulation-v4 paired generator at 25 MHz. Within each pair, normal and spoof share authentic signal, time, motion, receiver seed, impairments, frontend processing, and pre-onset samples. Generate the independent-multipath member with the preregistered per-pair seed, all startup LOS PRNs, delay range 0.12–0.45 chips, amplitude range 0.20–0.70, and hash-derived independent phase. Process multipath and spoof members using the pinned 11-channel complex-nine-tap GNSS-SDR executable.

## Frozen analysis

Estimate one signed complex-template delay per PRN per one-second bin. Require at least eight PRNs. Fit LOS plus a common clock intercept and calculate

`R_c = SSE_full / sum(w_i (delay_i - weighted_mean(delay))^2)`.

The spoof detection score is `-R_c`. For each pair and scenario, retain bins beginning at `spoof start + max(transition, power ramp) + 1 second`, then take the scenario median. The primary pair separation is `median R_c(multipath) - median R_c(spoof)`; positive is the preregistered direction. Bin-level AUC is descriptive only.

## Decision

Report `SUPPORTED` only if all six configured gates pass: all three pairs complete, all three separations are positive, pair-block AUC is at least 0.80, clock-centering improves over the legacy residual on at least two pairs, every scenario-pair has at least five eligible bins, and every pair has at least eight startup LOS PRNs. Otherwise report `NOT_SUPPORTED` without tuning or retry.

## TEXBAT boundary

TEXBAT is not part of this primary test. A separate adapter may apply the same frozen estimator and residual to preserved complex-nine-tap TEXBAT DS1–DS3 epochs. Because those recordings provide spoof pre/post regions but no labeled independent-multipath class, that result is complementary change evidence and cannot replace this multipath-versus-spoof endpoint.

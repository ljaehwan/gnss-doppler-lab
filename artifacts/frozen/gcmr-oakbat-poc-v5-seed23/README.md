# Frozen GCMR OAKBAT v5 result (seed 23)

This directory preserves the existing OAKBAT GCMR v5 checkpoint and its exact evaluation outputs. It is historical evidence and was not rerun or retuned while importing it into Git.

## Frozen identity

- Training/code commit recorded by the artifact: `d4deb4dc0aba0070d99886f515a3917c4827144c`
- Threshold: `2.8224338538365963`
- Checkpoint: `model.pt`
- Evaluation: one cleanStatic-trained frozen model applied to OS1–OS4
- Classification: offline evaluation; decoded ephemeris may include an end-of-run oracle snapshot

## Reported results

- Sealed clean held: 2/119 alarms (1.68%)
- OS1 pre: 71/218 alarms (32.57%); post: 125/700 (17.86%)
- OS2 pre: 71/218 alarms (32.57%); post: 700/700 (100%)
- OS3 pre: 82/218 alarms (37.61%); post: 700/700 (100%)
- OS4 pre: 64/218 alarms (29.36%); post: 700/700 (100%)

The strong OS2–OS4 post-attack rates must not be described without the high OAKBAT pre-attack false-alarm rates. The artifact therefore supports historical reproduction and diagnosis, not a claim of a deployment-ready detector.

## Contents

- `model.pt`: frozen checkpoint
- `summary.json`, `provenance.json`: complete contract and source identities
- `cleanStatic_scores.csv`, `os1_scores.csv`–`os4_scores.csv`: frozen scores
- timeline and dashboard PNGs
- plotting script
- `SHA256SUMS`: byte-integrity manifest generated when imported into Git

Raw receiver inputs and the regenerable relation cache are not included in this Git directory.

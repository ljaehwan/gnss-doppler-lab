# SPLITCLOCK-GNSS Stage-0A R1 contract/model repair

Verdict: `NO_GO_SPLITCLOCK_CLEAN_FALSE_ALARMS`.

This is a clean-only and synthetic-on-clean Stage-0A result, not an attack-detection result. The design was frozen at `d472376ba2e59d766c93296b4755df4c89ccbe9b`, and the implementation was frozen at `3b2d943b15eed67df55051c123682ac92453a699` before the single C-1/C-3 execution.

Both clean panels passed geometry, sign/unit, cadence, finite-coverage, and continuous-support contracts. The C-3 holdout had zero epoch or persistent false alarms. It nevertheless failed the preregistered clean shortcut gate because the score/PRN-count correlation was `-0.5249624991219917`, whose magnitude exceeds `0.3`. The clean evidence remains `PROVISIONAL_LIMITED_DURATION`: only three independent calibration blocks and 22 persistent holdout decisions were available, with a zero-false-alarm 95% upper bound of `0.15437251281557463`.

The 3/5-PRN mild, moderate, and accelerating synthetic detection rates were all zero. Primary median localization F1 was `0.6333333333333333`. Negative and all-PRN boundary controls had zero persistent false alarms. A6 improved AUROC over A0 by `0.11805555555555558`; temporal/modal destruction reduced K=2 advantage by `0.7113495562863003`, and PRN renaming changed the score by only `2.2737367544323206e-13`.

Attack and Jammertest raw access were both zero bytes. The result does not authorize an attack freeze.

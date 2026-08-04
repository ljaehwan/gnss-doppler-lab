# NC-TOPI Stage-0B implementation TDD evidence

This implementation followed the frozen Stage-0B contract. No production audit
campaign or Stage-0B result artifact was generated.

## RED (before production code)

Command:

```text
/home/ubuntu/projects/gnss-doppler-lab/.venv/bin/python -m pytest -q tests/test_nc_topi_shortcut_audit.py
```

Observed result (`RED_ELAPSED=0.54` seconds): collection failed with exit status
2 because `gnss_doppler_lab.nc_topi_stage0b` did not exist:

```text
ImportError: cannot import name 'nc_topi_stage0b' from 'gnss_doppler_lab'
ERROR tests/test_nc_topi_shortcut_audit.py
1 error in 0.22s
```

The test was written and this failure was captured in
`/tmp/nc_topi_stage0b_red.txt` on the authoritative host before any production
module or audit script was created.

One initially over-strong reconstruction assertion expected the global maximum
absolute floating-point error alone to be at most `1e-12`. The contract instead
requires every row to satisfy the combined NumPy `rtol=1e-12, atol=1e-12`
predicate. The test was corrected to assert that exact row-wise predicate and
the maximum relative error; the verifier still records both maxima.

## GREEN and reconstruction gate

- Focused Stage-0B tests: `17 passed in 6.85s` (latest focused run).
- Earlier timed focused run: `13 passed in 4.58s`, wall `5.10s`.
- Exact original reconstruction-only gate: 55,591 rows, all rows within combined
  relative/absolute `1e-12`; max absolute error `0.0003986358642578125`, max
  relative error `3.745528836138847e-14`, q99.5 cap
  `1.4220200618224565`, wall `4.85s`.

## Frozen Stage-0 regressions

- `tests/test_nc_topi.py`: `39 passed in 14.11s`, wall `14.61s`.
- `tests/test_nc_topi_runner.py`: `37 passed in 442.98s`, wall `444.50s`.

All commands above ran in the authoritative SSH worktree with the required
project Python interpreter.

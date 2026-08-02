# AMCF-Lite TEXBAT feasibility

## Scope and status

AMCF-Lite is a **developmental, post-exposure feasibility experiment**, not a confirmatory detector result. DS1–DS3, DS7, and DS8 are already exposed. This implementation deliberately does not modify the CMTE-A2 implementation or any existing artifact. The campaign runner creates a new non-overwriting artifact tree.

DS4 is unavailable and is not synthesized or silently substituted.

## Input and causal contract

Canonical NPZ input fields are:

- `complex_iq[N,9,2]`, with final axis `[I,Q]`
- `sample_count[N]`, `time_s[N]`, `prn[N]`, `channel[N]`, `segment_index[N]`
- optional `cn0_db_hz[N]`
- tap order `E4,E3,E2,E,P,L,L2,L3,L4`

The runner pins and verifies all six canonical NPZ SHA-256 values before use. Each raw row is mapped to the first recording-relative 0.5 s decision grid at or after its timestamp. At each decision epoch there is at most one row per PRN: latest timestamp wins, followed by sample count, segment, channel, and a stable row-content SHA-256 for an exact tie. Future rows are forbidden. A raw row is not carried to later epochs. Recordings are loaded separately and segment/channel metadata participates in the deterministic selection; there is no recurrent state to bridge boundaries.

PRN is grouping/QA metadata only and is never a model feature.

## Prompt quality and representations

The prompt quality gate is the higher `q0.005` of `abs(P)` on cleanStatic train rows only (`0 <= time < 240 s`). Rows below the frozen gate are rejected. For accepted rows:

```text
Y_k = C_k * conj(P) / |P|^2
```

This is invariant to a common global carrier phase and to a common navigation-bit sign. Rejected rows become NaN before being removed, preventing near-zero prompts from becoming large finite normalized values.

Three separately fitted models use shared tap coordinates and no PRN identity:

- complex: `Re(Y), Im(Y)`
- magnitude: `abs(Y)`
- phase: `cos(angle(Y)), sin(angle(Y))`

## Chronological isolation

Clean roles use decision timestamps and fixed gaps:

- train: `[0,240)`
- validation: `[250,330)`
- calibration: `[340,410)`
- independent clean test: `[420,+inf)`

Only clean train is used for the prompt gate and model fit. Validation is reported by the model fit audit. Only clean calibration is accepted by the threshold API. Attack paths cannot be passed to that API without an explicit failure. Primary q99 and diagnostic q99.5 use the NumPy-style higher order statistic and strict `score > threshold` alarms.

## Model

`MaskedSetModel` is a small coordinate-conditioned masked-set/neural-process feasibility model:

- deterministic shared nonlinear token encoder, ELM-style frozen after seeded initialization;
- token input: delay coordinate, transformed observed value, observed flag;
- masked-set mean context, shared coordinate-conditioned decoder;
- fitted heteroscedastic location/scale heads;
- Student-t NLL with `df=4`;
- three random masks per clean train field;
- hidden width constrained to `<=64` and optimizer iterations constrained to `<=50`;
- deterministic NumPy/SciPy CPU execution and explicit seeds.

The frozen ELM-style encoder is a compute-saving feasibility compromise; this is not a fully end-to-end learned neural process. `model_audit.json` records optimizer status, iteration cap, train objective, validation NLL, and model-state hash.

## Query and score policy

Seed taps are E/P/L, indices `[3,4,5]`. Each seed is scored leave-one-out against the other two. Every added tap is scored before its value is revealed.

- fixed K5 set: `[2,3,4,5,6]`
- fixed K7 set: `[1,2,3,4,5,6,7]`
- all9: all taps
- random K5/K7: seeded extras selected without values; default seeds `11,23,37`
- adaptive K5/K7: sequential maximum mean predictive scale; selector API receives only observed values and candidate indices

The per-PRN score is the mean of the largest two queried Student-t NLL values. The decision-epoch score is the median across current PRNs, making it order invariant and robust to varying tracked N.

## Comparisons and outputs

The runner evaluates magnitude all9, phase all9, complex K3, fixed K5/K7, random K5/K7, adaptive K5/K7, complex all9, and B0-Exact. B0-Exact values are loaded from `artifacts/cmte_a2_texbat_epochfix/per_epoch`, intersected with AMCF epochs, and q99/q99.5 are recomputed on common clean `[340,410)` epochs. Every model in a scenario is restricted to the same common epoch timestamps when B0 is present.

Attack onsets are DS1–DS3 `100 s` and DS7/DS8 `110 s`. Phase masks use contained half-second windows with the corrected CMTE-A2 semantics. A clearly labeled matched-FPR file is fitted on normal clean test only and is diagnostic, not an independent operating point.

An atomic successful output contains:

- `qa.json`: schema/alignment, hashes, I/Q and tap order, prompt gate/rejection, invariance, tap magnitude/phase/curvature, causality, and tracked-N distributions
- `model_audit.json`, `config.json`, `thresholds.json`
- `metrics.csv`, `matched_fpr.csv`
- `per_epoch/<scenario>.csv`
- `plots/<scenario>.png`
- `README.md`, `hashes.json`

The destination must not exist. A sibling staging directory is removed on failure and renamed atomically only after all output generation succeeds.

## Commands and strict-TDD evidence

RED was recorded before `amcf_lite.py` existed:

```text
$ python3 -m pytest -q tests/test_amcf_lite.py
E   ModuleNotFoundError: No module named 'gnss_doppler_lab.amcf_lite'
1 error in 0.33s
```

GREEN after implementation:

```text
$ python3 -m pytest -q tests/test_amcf_lite.py
12 passed in 0.35s
```

The tests cover global phase, navigation sign, low-prompt stability/train-only gate, representation geometry, masked-value leakage, hidden-value-free adaptive selection with a manual uncertainty check, causal timestamps/stable latest tie, PRN/input order, variable N, normal-only calibration, deterministic refit, manual Student-t NLL/top-2 scoring, seeded value-free random selection, and exact chronological roles.

Real canonical-data smoke command:

```bash
python3 scripts/run_amcf_lite_texbat.py \
  --scenario DS1=/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds123-graph-input/exports/ds1.npz \
  --out /tmp/amcf-lite-real-smoke \
  --epochs 2 --hidden 12 --random-seeds 11 --smoke
```

Observed result on 2026-08-02:

```text
{"out": "/tmp/amcf-lite-real-smoke", "smoke": true}
ELAPSED=85.14 MAXRSS=325252
qa passed: true
clean selected rows/decision epochs: 2400/226
DS1 selected rows/decision epochs: 3079/300
causal future rows: 0; duplicate epoch/PRN: 0
models: 11; common clean calibration epochs/model: 41
metrics rows: 44; output hash entries: 11; hash mismatches: 0
```

The same output path was rerun and failed immediately with `FileExistsError`; no staging directory remained.

The 2-iteration smoke is an execution/real-data QA check, not a performance estimate. Its optimizer audit correctly says the iteration cap was reached. Full feasibility execution should use the default 25 iterations and all default random seeds:

```bash
python3 scripts/run_amcf_lite_texbat.py \
  --out artifacts/amcf_lite_texbat \
  --epochs 25 --hidden 32
```

## Relation to prior correlator-field methods

AMCF-Lite does **not** claim that complex correlator features, CAF-based neural detection, sparse correlator decomposition, or active sensing are individually new.

- **CAF-DNN-style work:** a CAF (cross-ambiguity function) DNN commonly consumes a fixed dense delay/Doppler correlation field and learns a supervised or task-specific classifier. AMCF-Lite has no local Doppler axis and no dense CAF image: it sees only a receiver-produced one-dimensional 9-tap tracking slice, trains on cleanStatic by masked prediction, and evaluates a fixed query budget. Therefore it is cheaper but cannot claim the delay-Doppler observability of a full CAF-DNN.
- **LASSO correlator decomposition:** sparse LASSO methods solve an explicit dictionary inverse problem to decompose a correlation shape into a small number of delayed components. They provide model-based sparse component estimates under a chosen dictionary. AMCF-Lite does not identify authentic/spoof/multipath components and has no sparse physical dictionary; it only measures whether queried complex taps are improbable under a clean masked-set predictor.
- **2025 complex CCAF decomposition:** recent complex CCAF decomposition work retains complex correlation structure and explicitly decomposes or represents components over a richer complex ambiguity field. AMCF-Lite preserves complex Re/Im after Prompt referencing, but still has only nine code-delay taps, diagonal predictive scales, and no Doppler dimension or component identifiability. It must not be presented as a replacement for complex CCAF decomposition.

The defensible feasibility question is narrower: **does clean-only conditional prediction of sparse complex tracking taps expose information lost by magnitude-only B0 features, and can a deterministic uncertainty policy approach all-nine-tap performance at K=5 or K=7?** Exact bibliographic metadata and claim overlap must be re-verified before using this wording in a paper.

## Caveats

- All attack results are post-exposure and exploratory.
- One clean recording supplies train, validation, calibration, and test roles; chronological separation does not create independent recordings.
- q99/q99.5 calibration has few common epochs after B0 intersection and can collapse to the same maximum order statistic.
- The smoke run caps rows and optimizer iterations and must not be cited as detector performance.
- Adaptive uncertainty comes from the fitted heteroscedastic decoder, but the token encoder is frozen; a fully trained neural-process follow-up may behave differently.
- Common prompt rejection may alter tracked N over time. QA records this, but it can still induce distribution shift.
- Matched-FPR uses clean test and is diagnostic only.

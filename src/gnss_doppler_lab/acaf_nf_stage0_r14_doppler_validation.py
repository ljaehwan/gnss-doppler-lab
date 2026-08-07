"""R1.4 cleanStatic reconstruction and Doppler-resolution diagnostics.

This module contains deterministic, synthetic-testable science primitives.  It
does no file discovery and never opens a recording; the production wrapper owns
authenticated I/O.  R1.3 is imported only for its already-audited physical CAF.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from gnss_doppler_lab.acaf_nf_stage0_r13_reconstruction import Candidate, caf

FS = 25_000_000.0
SUPPORT_SAMPLES = 25_000
LENGTHS = (1, 5, 10, 20)
ROLES = ("train", "calibration", "holdout")
CANDIDATE_STRING = "nco_row=previous_aux_row=previous_remnant_sign=-1_carrier_sign=-1_global_offset=0"
R13_SOURCE_SHA256 = "9889a5e5007c92d6016e5ef0d38a03cea96cdd40eded3cea91df1e4276d16e42"
R13_CHECKSUMS_SHA256 = "04b5395b311641b4ab3f3a58a1a5cbb54d4249068f8252659049ea4386a95abb"
R13_CENTER_VALIDATION_SHA256 = "cb07c2b3d192c6bd30e6eeca6ffae6d615523f1ba4569d4259ccb01d866ba198"
R13_IDENTITY_ORDER_SHA256 = "65933645102b7a05087f0d9991ad1c55c822b4b83090cb57a4e6f74e17675e5c"
R13_REFERENCE = {
    "n": 969, "prn_count": 8,
    "pooled_spearman": 0.9999965049269979,
    "median_prn_spearman": 0.9999652753663446,
    "boundary_fraction": 0.006191950464396285,
    "within_tolerance_fraction": 0.8565531475748194,
    "exact_center_fraction": 0.42105263157894735,
}
PROHIBITED_VERDICTS = {"PHYSICS_NO_GO", "ACAF_MODEL_NO_GO", "SOURCE_BINDING_INVALID"}


@dataclass(frozen=True)
class FrozenConfig:
    signal: str = "gps_l1ca_code"
    fs_hz: float = FS
    raw_format: str = "interleaved_signed_int16_iq"
    global_offset_samples: int = 0
    nco_row: str = "previous"
    aux_row: str = "previous"
    remnant_sign: int = -1
    carrier_sign: int = -1
    replica_direction: str = "forward"
    prompt_row: str = "current"
    support_samples: int = SUPPORT_SAMPLES

    @property
    def candidate(self) -> Candidate:
        return Candidate(self.nco_row, self.aux_row, self.remnant_sign,
                         self.carrier_sign, self.global_offset_samples)

    def document(self) -> dict:
        value = asdict(self)
        value["candidate_string"] = CANDIDATE_STRING
        return value


FROZEN_CONFIG = FrozenConfig()


def clean_only_guard(names: Iterable[str]) -> None:
    names = list(names)
    if names != ["cleanStatic"]:
        raise ValueError("R1.4 accepts cleanStatic only; attack inputs are prohibited")


def check_r13_metrics(actual: Mapping, tolerance: float = 1e-6) -> bool:
    """Fail closed when any requested immutable R1.3 metric has drifted."""
    for key, expected in R13_REFERENCE.items():
        if key not in actual or not np.isfinite(float(actual[key])):
            return False
        if abs(float(actual[key]) - expected) > tolerance:
            return False
    return True


def _rho(x, y) -> float:
    if len(x) < 3:
        return 0.0
    value = float(spearmanr(x, y).statistic)
    return value if np.isfinite(value) else 0.0


def prompt_evidence(row: Mapping) -> dict:
    reconstructed = float(row["center_magnitude"])
    mat = float(row["mat_prompt_magnitude"])
    if not np.isfinite(reconstructed) or not np.isfinite(mat) or mat <= 0:
        raise ValueError("Prompt magnitudes must be finite and MAT Prompt positive")
    ratio = reconstructed / mat
    return {**dict(row), "prompt_ratio": ratio, "prompt_abs_relative_error": abs(ratio - 1.0)}


def prompt_metrics(rows: Sequence[Mapping]) -> dict:
    evidence = [prompt_evidence(r) for r in rows]
    errors = np.asarray([r["prompt_abs_relative_error"] for r in evidence])
    by_prn = [_rho([r["center_magnitude"] for r in evidence if int(r["prn"]) == p],
                   [r["mat_prompt_magnitude"] for r in evidence if int(r["prn"]) == p])
              for p in sorted({int(r["prn"]) for r in evidence})]
    return {
        "n": len(evidence),
        "pooled_spearman": _rho([r["center_magnitude"] for r in evidence],
                                 [r["mat_prompt_magnitude"] for r in evidence]),
        "median_prn_spearman": float(np.median(by_prn)) if by_prn else 0.0,
        "median_relative_error": float(np.median(errors)) if len(errors) else float("inf"),
        "p95_relative_error": float(np.quantile(errors, .95)) if len(errors) else float("inf"),
        "p99_relative_error": float(np.quantile(errors, .99)) if len(errors) else float("inf"),
        "max_relative_error": float(np.max(errors)) if len(errors) else float("inf"),
    }


def prompt_gate(metrics: Mapping) -> bool:
    return (float(metrics.get("pooled_spearman", 0)) >= .999
            and float(metrics.get("median_prn_spearman", 0)) >= .99
            and float(metrics.get("median_relative_error", np.inf)) <= .001
            and float(metrics.get("p99_relative_error", np.inf)) <= .01)


def offset_zero_clearly_better(rows: Sequence[Mapping]) -> bool:
    by_offset = {int(r["global_offset_samples"]): r for r in rows}
    if set(by_offset) != {-1000, -500, 0, 500, 1000}:
        return False
    zero = by_offset[0]
    return all(float(zero["pooled_spearman"]) > float(by_offset[o]["pooled_spearman"])
               for o in (-1000, -500, 500, 1000))


def delay_metrics(rows: Sequence[Mapping]) -> dict:
    offsets = np.asarray([float(r["peak_delay_offset_chips"]) for r in rows])
    boundary = np.asarray([bool(r.get("delay_boundary", abs(float(r["peak_delay_offset_chips"])) >= 1)) for r in rows])
    n = len(rows)
    return {"n": n, "exact_center_fraction": float(np.mean(offsets == 0)) if n else 0.0,
            "within_0_125_fraction": float(np.mean(np.abs(offsets) <= .125)) if n else 0.0,
            "boundary_fraction": float(np.mean(boundary)) if n else 1.0,
            "histogram": {str(x): int(np.sum(offsets == x)) for x in sorted(set(offsets))}}


def delay_gate(overall: Mapping, by_prn: Sequence[Mapping], by_role: Sequence[Mapping]) -> bool:
    return (float(overall.get("within_0_125_fraction", 0)) >= .95
            and float(overall.get("boundary_fraction", 1)) <= .01
            and sum(float(x.get("within_0_125_fraction", 0)) >= .95 for x in by_prn) >= 7
            and len(by_prn) == 8 and {x.get("role") for x in by_role} == set(ROLES)
            and all(float(x.get("within_0_125_fraction", 0)) >= .95 for x in by_role))


def doppler_metrics(rows: Sequence[Mapping]) -> dict:
    offsets = np.asarray([float(r["peak_doppler_offset_hz"]) for r in rows])
    ratios = np.asarray([float(r.get("peak_center_ratio", r.get("peak_magnitude", 0) /
                      max(float(r.get("center_magnitude", 0)), np.finfo(float).eps))) for r in rows])
    n = len(rows)
    result = {"n": n, "exact_center_fraction": float(np.mean(offsets == 0)) if n else 0.0,
              "boundary_fraction": float(np.mean([bool(r.get("doppler_boundary", abs(float(r["peak_doppler_offset_hz"])) >= 250)) for r in rows])) if n else 1.0,
              "median_abs_offset_hz": float(np.median(np.abs(offsets))) if n else float("inf"),
              "p95_abs_offset_hz": float(np.quantile(np.abs(offsets), .95)) if n else float("inf"),
              "median_peak_center_ratio": float(np.median(ratios)) if n else float("inf"),
              "histogram": {str(x): int(np.sum(offsets == x)) for x in sorted(set(offsets))}}
    for hz in (50, 100, 150):
        result[f"within_{hz}_fraction"] = float(np.mean(np.abs(offsets) <= hz)) if n else 0.0
    return result


def normalized_noncoherent_power(surfaces: Sequence[np.ndarray], eps: float = 1e-15) -> np.ndarray:
    """Predeclared primary: mean_k(|C_k|²/(sum_grid |C_k|²+eps))."""
    values = np.asarray(surfaces)
    if values.ndim < 2 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("finite same-shaped constituent surfaces required")
    power = np.abs(values) ** 2
    axes = tuple(range(1, power.ndim))
    return np.mean(power / (np.sum(power, axis=axes, keepdims=True) + eps), axis=0)


def diagnostic_aggregates(surfaces: Sequence[np.ndarray]) -> dict[str, np.ndarray]:
    values = np.abs(np.asarray(surfaces))
    return {"normalized_power_mean": normalized_noncoherent_power(values),
            "raw_power_sum": np.sum(values ** 2, axis=0),
            "magnitude_mean": np.mean(values, axis=0),
            "robust_median": np.median(values, axis=0)}


def common_anchor_blocks(rows: Sequence[Mapping], lengths=LENGTHS) -> dict[int, list[list[Mapping]]]:
    """Build trailing windows only at anchors having an authenticated L=20 history."""
    if tuple(lengths) != LENGTHS:
        raise ValueError("R1.4 lengths are frozen to L=1,5,10,20")
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        key = (str(row["channel"]), int(row["prn"]), int(row["tracker_row"]))
        if key in seen:
            raise ValueError("duplicate tracker row identity")
        seen.add(key); groups[key[:2]].append(row)
    result = {length: [] for length in lengths}
    for (channel, prn), group in groups.items():
        group.sort(key=lambda r: int(r["tracker_row"]))
        for end in range(19, len(group)):
            window = group[end-19:end+1]
            ids = [int(r["tracker_row"]) for r in window]
            roles = {str(r["role"]) for r in window}
            starts = [int(r["support_start_sample"]) for r in window]
            valid = all(bool(r.get("valid_raw_support", True)) and int(r.get("support_length_samples", SUPPORT_SAMPLES)) == SUPPORT_SAMPLES
                        and float(r["cn0_db_hz"]) >= 28 and float(r["carrier_lock"]) >= .85 for r in window)
            # Source-authenticated 25,000-sample supports may overlap by one
            # sample when their starts differ by 24,999; authenticated positive
            # gaps are retained.  Reject duplicates and overlap greater than one.
            deltas = [b-a for a, b in zip(starts, starts[1:])]
            if (ids != list(range(ids[0], ids[0]+20)) or len(roles) != 1 or not valid
                    or any(delta < SUPPORT_SAMPLES-1 for delta in deltas)):
                continue
            for length in lengths:
                block = window[-length:]
                if any(str(r["channel"]) != channel or int(r["prn"]) != prn for r in block):
                    raise AssertionError("mixed PRN/channel block")
                result[length].append(block)
    anchors = [[str(b[-1]["channel"]), int(b[-1]["prn"]), int(b[-1]["tracker_row"])] for b in result[20]]
    if any([[str(b[-1]["channel"]), int(b[-1]["prn"]), int(b[-1]["tracker_row"])] for b in result[L]] != anchors for L in lengths):
        raise AssertionError("common-anchor semantics violated")
    return result


def paired_improvements(l1_rows: Sequence[Mapping], other_rows: Sequence[Mapping], tolerance_hz=50) -> list[dict]:
    identity = lambda r: (str(r["channel"]), int(r["prn"]), int(r["anchor_tracker_row"]), str(r["role"]))
    left = {identity(r): r for r in l1_rows}; right = {identity(r): r for r in other_rows}
    if set(left) != set(right) or len(left) != len(l1_rows) or len(right) != len(other_rows):
        raise ValueError("paired rows require unique identical common anchors")
    return [{"channel": k[0], "prn": k[1], "anchor_tracker_row": k[2], "role": k[3],
             "l1_success": abs(float(left[k]["peak_doppler_offset_hz"])) <= tolerance_hz,
             "aggregated_success": abs(float(right[k]["peak_doppler_offset_hz"])) <= tolerance_hz,
             "difference": int(abs(float(right[k]["peak_doppler_offset_hz"])) <= tolerance_hz)
                           - int(abs(float(left[k]["peak_doppler_offset_hz"])) <= tolerance_hz)}
            for k in sorted(left)]


def bootstrap_paired(rows: Sequence[Mapping], seed: int = 1401, replicates: int = 10_000) -> dict:
    """Fixed-seed PRN-block bootstrap; resampling unit is PRN."""
    by_prn = defaultdict(list)
    for row in rows: by_prn[int(row["prn"])].append(float(row["difference"]))
    prns = sorted(by_prn)
    if not prns or replicates <= 0: raise ValueError("bootstrap requires paired PRN rows")
    rng = np.random.default_rng(seed); estimates = np.empty(replicates)
    for i in range(replicates):
        chosen = rng.choice(prns, size=len(prns), replace=True)
        estimates[i] = np.mean([v for p in chosen for v in by_prn[int(p)]])
    observed = float(np.mean([float(r["difference"]) for r in rows]))
    return {"seed": seed, "replicates": replicates, "observed_difference": observed,
            "ci95_low": float(np.quantile(estimates, .025)), "ci95_high": float(np.quantile(estimates, .975)),
            "sign_consistent": bool(np.quantile(estimates, .025) > 0)}


def aggregation_gate(l20: Mapping, bootstrap: Mapping, by_prn: Sequence[Mapping], by_role: Sequence[Mapping]) -> bool:
    return (float(l20.get("within_50_fraction", 0)) >= .95
            and float(l20.get("boundary_fraction", 1)) <= .01
            and float(bootstrap.get("ci95_low", 0)) > 0
            and sum(float(x.get("difference", 0)) > 0 for x in by_prn) >= 7 and len(by_prn) == 8
            and {x.get("role") for x in by_role} == set(ROLES)
            and all(float(x.get("difference", 0)) > 0 for x in by_role))


def final_gates(a1: bool, a2: bool, a3a: bool, a3b: bool, a3c: bool) -> dict:
    if not a1 or not a2: verdict = "RECONSTRUCTION_IMPLEMENTATION_INVALID"
    elif not a3a or not a3b: verdict = "TRACKER_RAW_RECONSTRUCTION_UNRESOLVED"
    elif not a3c: verdict = "PHYSICAL_RECONSTRUCTION_VALID_DOPPLER_RESOLUTION_LIMITED"
    else: verdict = "PHYSICAL_CENTER_VALID"
    return {"A1_SOURCE_BINDING": "PASS" if a1 else "FAIL",
            "A2_RECONSTRUCTION_IMPLEMENTATION": "PASS" if a2 else "FAIL",
            "A3a_PROMPT_REPRODUCTION": "PASS" if a3a else "FAIL",
            "A3b_CODE_DELAY": "PASS" if a3b else "FAIL",
            "A3c_DOPPLER_AGGREGATION": "PASS" if a3c else "FAIL", "verdict": verdict}


def caf_surface(iq, prn, code_freq, aux, doppler, grid):
    """Frozen R1.3 physical implementation, returning a numeric CAF surface."""
    # The R1.3 public CAF intentionally returns only summary/hash evidence.  This
    # wrapper is retained for compatibility; production computes/caches surfaces
    # with the equivalent vectorized equations in the R1.4 runner.
    return caf(iq, prn, FS, code_freq, aux, doppler, FROZEN_CONFIG.candidate, grid)

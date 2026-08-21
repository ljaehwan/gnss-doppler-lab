"""Fail-closed helpers for the versioned CRID R4a threshold repair."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import numpy as np


BASE_SHA = "04ea478e5cb9f4d5da05563c1d883b2f7b20a28b"
PREREGISTRATION_SHA = "982696413845b9643047d07e9229e51bf700e3d1"
QUANTILE = 0.99
QUANTILE_METHOD = "higher"
NUMERIC_MULTIPLIER = 1e-12
COMMITTED = {
    "OAK": {"q99": -21.705587048010322, "fpr": 0.00730093543235227},
    "TEX": {"q99": -21.942672917134093, "fpr": 0.012708150744960562},
}
EXPECTED_SPLIT = {
    "OAK": {
        "train": (150100296, 182037405, 70609),
        "guard1": (182037888, 183585342, 3139),
        "calibration": (183586221, 201521350, 36089),
        "guard2": (201522279, 203079808, 3138),
        "holdout": (203079809, 224989820, 43936),
    },
    "TEX": {
        "train": (750505555, 901013382, 74254),
        "guard1": (901013790, 908990496, 3300),
        "calibration": (908991342, 1002581863, 37952),
        "guard2": (1002583030, 1010858046, 3301),
        "holdout": (1010859485, 1124942029, 46203),
    },
}


class ThresholdProvenanceError(RuntimeError):
    """Raised when a preregistered binding or decision gate fails."""


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for payload in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(payload)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def require_file_binding(path: Path, expected_size: int, expected_sha256: str) -> None:
    candidate = Path(path)
    if candidate.stat().st_size != int(expected_size):
        raise ThresholdProvenanceError(f"size binding mismatch: {candidate}")
    if sha256_file(candidate) != str(expected_sha256):
        raise ThresholdProvenanceError(f"SHA-256 binding mismatch: {candidate}")


def require_clean_path(path: Path, clean_roots: tuple[Path, ...], exact_files: tuple[Path, ...] = ()) -> Path:
    """Reject non-clean SSD paths before any filesystem access."""
    candidate = Path(path)
    lexical = candidate.as_posix()
    if any(token in lexical.lower() for token in ("/controls/", "/scores/", "texbat/ds", "oakbat/os")):
        raise ThresholdProvenanceError(f"forbidden control/attack path: {candidate}")
    if candidate in exact_files:
        return candidate
    if not any(candidate == root or root in candidate.parents for root in clean_roots):
        raise ThresholdProvenanceError(f"path outside clean allowlist: {candidate}")
    return candidate


def independent_chronological_split(samples: np.ndarray) -> dict[str, np.ndarray]:
    unique = np.unique(np.asarray(samples, dtype=np.int64))
    count = len(unique)
    cuts = [int(count * fraction) for fraction in (0.45, 0.47, 0.70, 0.72)]
    names = ("train", "guard1", "calibration", "guard2", "holdout")
    bounds = ((0, cuts[0]), (cuts[0], cuts[1]), (cuts[1], cuts[2]), (cuts[2], cuts[3]), (cuts[3], count))
    result = {name: np.ascontiguousarray(unique[start:end], dtype="<i8") for name, (start, end) in zip(names, bounds, strict=True)}
    if any(len(result[name]) == 0 for name in names):
        raise ThresholdProvenanceError("empty guarded clean split")
    return result


def split_identity(domain: str, candidate: Mapping[str, np.ndarray], reference: Mapping[str, np.ndarray]) -> dict:
    rows = {}
    passed = True
    for name in ("train", "guard1", "calibration", "guard2", "holdout"):
        left = np.ascontiguousarray(candidate[name], dtype="<i8")
        right = np.ascontiguousarray(reference[name], dtype="<i8")
        expected = EXPECTED_SPLIT[domain][name]
        endpoints = (int(left[0]), int(left[-1]), len(left))
        identical = left.tobytes() == right.tobytes()
        match = identical and endpoints == expected
        passed &= match
        rows[name] = {
            "first_sample": endpoints[0], "last_sample": endpoints[1], "count": endpoints[2],
            "expected_first_sample": expected[0], "expected_last_sample": expected[1], "expected_count": expected[2],
            "candidate_sha256": sha256_bytes(left.tobytes()), "reference_sha256": sha256_bytes(right.tobytes()),
            "byte_identical": identical, "status": "PASS" if match else "FAIL",
        }
    return {"splits": rows, "status": "PASS" if passed else "FAIL"}


def evaluate_threshold_equivalence(
    domain: str,
    recomputed_threshold: float,
    holdout_scores: np.ndarray,
    *,
    quantile_method: str = QUANTILE_METHOD,
) -> tuple[dict, dict]:
    committed = float(COMMITTED[domain]["q99"])
    expected_fpr = float(COMMITTED[domain]["fpr"])
    scores = np.ascontiguousarray(holdout_scores, dtype="<f8")
    finite = bool(np.all(np.isfinite(scores)) and np.isfinite(recomputed_threshold))
    bound = NUMERIC_MULTIPLIER * max(1.0, abs(committed))
    difference = abs(float(recomputed_threshold) - committed)
    numeric_ok = finite and difference <= bound and quantile_method == QUANTILE_METHOD
    committed_alarm = np.ascontiguousarray(scores > committed, dtype=np.uint8)
    recomputed_alarm = np.ascontiguousarray(scores > float(recomputed_threshold), dtype=np.uint8)
    alarm_identical = committed_alarm.tobytes() == recomputed_alarm.tobytes()
    committed_count = int(committed_alarm.sum())
    recomputed_count = int(recomputed_alarm.sum())
    committed_fpr = float(committed_count / len(scores)) if len(scores) else float("nan")
    recomputed_fpr = float(recomputed_count / len(scores)) if len(scores) else float("nan")
    fpr_equal = committed_count == recomputed_count and committed_fpr == recomputed_fpr
    expected_fpr_match = committed_fpr == expected_fpr
    numeric = {
        "committed_q99": committed, "recomputed_q99": float(recomputed_threshold),
        "absolute_difference": difference, "sanity_bound": bound, "finite": finite,
        "quantile": QUANTILE, "quantile_method": quantile_method,
        "numeric_sanity_pass": numeric_ok,
        "status": "PASS" if numeric_ok else "FAIL",
    }
    alarm = {
        "holdout_score_count": len(scores), "holdout_score_sha256": sha256_bytes(scores.tobytes()),
        "committed_alarm_sha256": sha256_bytes(committed_alarm.tobytes()),
        "recomputed_alarm_sha256": sha256_bytes(recomputed_alarm.tobytes()),
        "alarm_vectors_byte_identical": alarm_identical,
        "committed_false_positive_count": committed_count, "recomputed_false_positive_count": recomputed_count,
        "committed_fpr": committed_fpr, "recomputed_fpr": recomputed_fpr,
        "false_positive_count_and_fpr_equal": fpr_equal,
        "expected_committed_fpr": expected_fpr, "expected_committed_fpr_exact_match": expected_fpr_match,
        "status": "PASS" if alarm_identical and fpr_equal and expected_fpr_match else "FAIL",
    }
    return numeric, alarm

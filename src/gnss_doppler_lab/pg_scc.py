"""Shared scoring, pooling, metrics, and artifact helpers for PG-SCC."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import beta, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from gnss_doppler_lab.pg_scc_physics import one_source_residual, two_source_glrt


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fields or (values[0].keys() if values else ()))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(values)


def load_feature_cache(npz_path: Path, json_path: Path) -> list[dict[str, Any]]:
    archive = np.load(npz_path, allow_pickle=False)
    metadata = load_json(json_path)
    if len(metadata) != len(archive["surfaces"]):
        raise RuntimeError("feature metadata/array length mismatch")
    result = []
    for index, row in enumerate(metadata):
        result.append({**row, "surface": archive["surfaces"][index], "l20_variance": archive["variances"][index]})
    return result


def pool(values: Sequence[float], method: str) -> float:
    data = np.sort(np.asarray(values, float))
    if data.size < 4 or not np.isfinite(data).all():
        raise ValueError("pooling requires at least four finite PRN-local scores")
    if method == "median":
        return float(np.median(data))
    if method == "robust_mean":
        trim = max(0, int(math.floor(0.1 * len(data))))
        return float(np.mean(data[trim:len(data) - trim] if trim else data))
    if method == "topk_mean":
        return float(np.mean(data[-max(1, math.ceil(len(data) / 3)):]))
    raise ValueError(f"unknown pooling: {method}")


def select_pooling(h0: Sequence[Sequence[float]], h1: Sequence[Sequence[float]]) -> tuple[str, dict[str, float]]:
    diagnostics = {}
    for method in ("median", "robust_mean", "topk_mean"):
        negative = [pool(x, method) for x in h0]
        positive = [pool(x, method) for x in h1]
        diagnostics[method] = float(roc_auc_score([0] * len(negative) + [1] * len(positive), negative + positive))
    return max(diagnostics, key=lambda name: (diagnostics[name], -("median", "robust_mean", "topk_mean").index(name))), diagnostics


def score_rows(
    rows: Sequence[Mapping[str, Any]], auth_template: np.ndarray, covariance: np.ndarray,
    masks: Mapping[str, Sequence[int]], normalization: str,
) -> list[dict[str, Any]]:
    from gnss_doppler_lab.pg_scc_physics import normalize_complex
    output: list[dict[str, Any]] = []
    for row in rows:
        surface = normalize_complex(np.asarray(row["surface"]), normalization)
        common = {key: row.get(key) for key in (
            "scenario", "phase", "second", "time_s", "channel", "prn", "raw_start_sample", "raw_end_sample",
            "raw_power", "prompt_magnitude",
        )}
        dense = two_source_glrt(surface, auth_template, covariance)
        output.append({**common, "method": "dense_two_source_glrt", "budget": 187, "score": dense.score})
        output.append({**common, "method": "dense_one_source_residual", "budget": 187,
                       "score": one_source_residual(surface, auth_template, covariance)})
        output.append({**common, "method": "raw_power_only", "budget": 0, "score": float(row.get("raw_power", 0.0))})
        for name, indices in masks.items():
            output.append({**common, "method": name, "budget": len(indices),
                           "score": two_source_glrt(surface, auth_template, covariance, indices=indices).score})
    return output


def pool_events(rows: Sequence[Mapping[str, Any]], method: str) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["scenario"], row["phase"], row["second"], row["method"], int(row["budget"]))].append(row)
    output = []
    for (scenario, phase, second, detector, budget), group in sorted(groups.items()):
        if len(group) < 4:
            continue
        output.append({
            "scenario": scenario, "phase": phase, "second": second,
            "time_s": float(np.mean([float(x["time_s"]) for x in group])),
            "method": detector, "budget": budget, "prn_count": len(group),
            "pooled_score": pool([float(x["score"]) for x in group], method),
        })
    return output


def fit_thresholds(clean_pooled: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    thresholds: dict[str, Any] = {}
    for method, budget in sorted({(x["method"], int(x["budget"])) for x in clean_pooled}):
        values = np.asarray([
            float(x["pooled_score"]) for x in clean_pooled
            if x["phase"] == "calibration" and x["method"] == method and int(x["budget"]) == budget
        ])
        if values.size < 10:
            raise RuntimeError(f"insufficient calibration events for {method}: {values.size}")
        thresholds[f"{method}:K{budget}"] = {
            "q99": float(np.quantile(values, 0.99, method="higher")),
            "q99.5": float(np.quantile(values, 0.995, method="higher")),
            "events": int(values.size), "source": "cleanStatic calibration event pooling only",
        }
    return thresholds


def exact_binomial_ci(alarms: int, total: int, confidence: float = 0.95) -> list[float]:
    if total <= 0 or not 0 <= alarms <= total:
        raise ValueError("valid binomial counts required")
    alpha = 1.0 - confidence
    lower = 0.0 if alarms == 0 else float(beta.ppf(alpha / 2, alarms, total - alarms + 1))
    upper = 1.0 if alarms == total else float(beta.ppf(1 - alpha / 2, alarms + 1, total - alarms))
    return [lower, upper]


def binary_metrics(labels: Sequence[int], scores: Sequence[float]) -> dict[str, float]:
    y, score = np.asarray(labels, int), np.asarray(scores, float)
    return {
        "roc_auc": float(roc_auc_score(y, score)),
        "pr_auc": float(average_precision_score(y, score)),
        "normalized_low_fpr_pauc": float(roc_auc_score(y, score, max_fpr=0.05)),
    }


def event_alarm_metrics(rows: Sequence[Mapping[str, Any]], threshold: float, onset: float) -> dict[str, Any]:
    time = np.asarray([float(x["time_s"]) for x in rows])
    score = np.asarray([float(x["pooled_score"]) for x in rows])
    post = time >= onset
    alarm = score >= threshold
    alarm_times = time[post & alarm]
    return {
        "events": int(post.sum()), "alarms": int((post & alarm).sum()),
        "detection_rate": float(np.mean(alarm[post])) if np.any(post) else None,
        "first_alarm_delay_s": float(alarm_times.min() - onset) if alarm_times.size else None,
        "persistent_alarm_ratio": float(np.mean(alarm[post])) if np.any(post) else None,
    }


def rank_score_correlation(left: Sequence[float], right: Sequence[float]) -> dict[str, float]:
    return {"pearson": float(np.corrcoef(left, right)[0, 1]), "spearman": float(spearmanr(left, right).statistic)}


def artifact_manifest(root: Path, *, exclude: Sequence[str] = ("artifact_manifest_sha256.json",)) -> dict[str, str]:
    excluded = set(exclude)
    return {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(root.rglob("*")) if path.is_file() and str(path.relative_to(root)) not in excluded
    }


def verify_manifest(root: Path, manifest_name: str = "artifact_manifest_sha256.json") -> list[str]:
    expected = load_json(root / manifest_name)
    errors = []
    for relative, digest in expected.items():
        path = root / relative
        if not path.is_file() or sha256(path) != digest:
            errors.append(relative)
    return errors

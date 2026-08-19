"""Frozen clean-only same-emitter verification machinery for CINDER Stage-0A."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


SEEDS = (2026081901, 2026081902, 2026081903, 2026081904, 2026081905,
         2026081906, 2026081907, 2026081908, 2026081909, 2026081910)


@dataclass(frozen=True)
class FrozenMetric:
    center: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    variance_floor: float
    shrinkage: float


def robust_standardizer(features: np.ndarray, *, floor: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=np.float64)
    center = np.median(x, axis=0)
    mad = 1.4826 * np.median(np.abs(x - center), axis=0)
    return center, np.maximum(mad, floor)


def remove_receiver_common(features: np.ndarray, blocks: np.ndarray) -> np.ndarray:
    """Subtract the within-block across-PRN robust center, split independently."""
    x = np.asarray(features, dtype=np.float64).copy()
    b = np.asarray(blocks)
    for block in np.unique(b):
        use = b == block
        x[use] -= np.median(x[use], axis=0)
    return x


def _all_difference_rows(features: np.ndarray, prns: np.ndarray, blocks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    same: list[np.ndarray] = []
    different: list[np.ndarray] = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            if blocks[i] == blocks[j]:
                continue
            delta = features[i] - features[j]
            if prns[i] == prns[j]:
                same.append(delta)
            else:
                different.append(delta)
    if not same or not different:
        raise ValueError("metric training requires same- and different-PRN cross-block pairs")
    return np.asarray(same), np.asarray(different)


def fit_shrinkage_metric(
    feature_train: np.ndarray,
    metric_train: np.ndarray,
    metric_prns: np.ndarray,
    metric_blocks: np.ndarray,
    *,
    variance_floor: float = 1e-6,
    shrinkage: float = 0.2,
) -> FrozenMetric:
    center, scale = robust_standardizer(feature_train, floor=variance_floor)
    z = (np.asarray(metric_train) - center) / scale
    same, different = _all_difference_rows(z, np.asarray(metric_prns), np.asarray(metric_blocks))
    within = np.var(same, axis=0) + variance_floor
    between = np.var(different, axis=0)
    raw = np.maximum(between - within, 0.0) / within
    raw = (1.0 - shrinkage) * raw + shrinkage * np.mean(raw)
    if not np.any(raw > 0):
        raw = np.ones_like(raw)
    weights = raw / np.mean(raw)
    return FrozenMetric(center=center, scale=scale, weights=weights,
                        variance_floor=variance_floor, shrinkage=shrinkage)


def pair_score(left: np.ndarray, right: np.ndarray, metric: FrozenMetric | None = None) -> float:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if metric is None:
        scale = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12)
        return float(np.dot(a, b) / scale)
    delta = (a - b) / metric.scale
    return -float(np.sum(metric.weights * delta * delta))


def matched_pairs(
    features: np.ndarray,
    prns: np.ndarray,
    blocks: np.ndarray,
    nuisance: np.ndarray,
    *,
    seed: int,
) -> list[dict[str, object]]:
    """Create one nuisance-matched negative for every exact-gap positive."""
    x, p, b = np.asarray(features), np.asarray(prns), np.asarray(blocks)
    n = np.asarray(nuisance, dtype=float)
    lookup = {(int(b[i]), int(p[i])): i for i in range(len(x))}
    unique_blocks, unique_prns = sorted(set(map(int, b))), sorted(set(map(int, p)))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for ia, block_a in enumerate(unique_blocks):
        for block_b in unique_blocks[ia + 1:]:
            gap = block_b - block_a
            for prn in unique_prns:
                li, ri = lookup[(block_a, prn)], lookup[(block_b, prn)]
                target = np.abs(n[li] - n[ri])
                candidates: list[tuple[float, float, int, int]] = []
                for pa in unique_prns:
                    for pb in unique_prns:
                        if pa == pb:
                            continue
                        ni, nj = lookup[(block_a, pa)], lookup[(block_b, pb)]
                        cost = float(np.linalg.norm((np.abs(n[ni] - n[nj]) - target) / (np.std(n, axis=0) + 1e-6)))
                        candidates.append((cost, float(rng.random()), ni, nj))
                candidates.sort()
                chosen = candidates[int(rng.integers(0, min(3, len(candidates))))]
                _, _, ni, nj = chosen
                rows.append({"label": 1, "left": li, "right": ri, "gap_blocks": gap,
                             "left_block": block_a, "right_block": block_b,
                             "left_prn": prn, "right_prn": prn, "match_cost": 0.0})
                rows.append({"label": 0, "left": ni, "right": nj, "gap_blocks": gap,
                             "left_block": block_a, "right_block": block_b,
                             "left_prn": int(p[ni]), "right_prn": int(p[nj]),
                             "match_cost": chosen[0]})
    return rows


def score_pairs(features: np.ndarray, pairs: list[dict[str, object]], metric: FrozenMetric | None) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray([int(row["label"]) for row in pairs])
    scores = np.asarray([pair_score(features[int(row["left"])], features[int(row["right"])], metric) for row in pairs])
    return labels, scores


def calibration_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    y, s = np.asarray(labels), np.asarray(scores)
    thresholds = np.unique(s)
    best = (-np.inf, float(thresholds[0]))
    for threshold in thresholds:
        pred = s >= threshold
        tpr = float(np.mean(pred[y == 1]))
        tnr = float(np.mean(~pred[y == 0]))
        candidate = ((tpr + tnr) / 2.0, -float(threshold))
        if candidate > (best[0], -best[1]):
            best = (candidate[0], float(threshold))
    return best[1]


def verification_metrics(labels: np.ndarray, scores: np.ndarray, *, threshold: float) -> dict[str, float]:
    y, s = np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("both pair classes are required")
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1.0 - tpr
    eer_index = int(np.argmin(np.abs(fpr - fnr)))
    pred = s >= threshold
    return {
        "roc_auc": float(roc_auc_score(y, s)),
        "low_fpr_standardized_pauc_0_05": float(roc_auc_score(y, s, max_fpr=0.05)),
        "pr_auc": float(average_precision_score(y, s)),
        "eer": float((fpr[eer_index] + fnr[eer_index]) / 2.0),
        "balanced_accuracy": float((np.mean(pred[y == 1]) + np.mean(~pred[y == 0])) / 2.0),
        "threshold": float(threshold),
        "positive_pairs": int(np.sum(y == 1)),
        "negative_pairs": int(np.sum(y == 0)),
    }


def block_bootstrap_auc(
    labels: np.ndarray,
    scores: np.ndarray,
    pairs: list[Mapping[str, object]],
    *,
    seed: int,
    repetitions: int = 2000,
) -> np.ndarray:
    """Parent-block cluster bootstrap using endpoint multiplicity weights."""
    y, s = np.asarray(labels), np.asarray(scores)
    blocks = sorted({int(row["left_block"]) for row in pairs} | {int(row["right_block"]) for row in pairs})
    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    for _ in range(repetitions):
        draw = rng.choice(blocks, size=len(blocks), replace=True)
        counts = {block: int(np.sum(draw == block)) for block in blocks}
        indices: list[int] = []
        for index, row in enumerate(pairs):
            multiplicity = counts[int(row["left_block"])] * counts[int(row["right_block"])]
            indices.extend([index] * multiplicity)
        if indices and len(np.unique(y[indices])) == 2:
            aucs.append(float(roc_auc_score(y[indices], s[indices])))
    if len(aucs) < repetitions // 2:
        raise ValueError("insufficient valid parent-block bootstrap replicates")
    return np.asarray(aucs)


def summarize_seed_values(values: Iterable[float]) -> dict[str, float]:
    x = np.asarray(tuple(values), dtype=float)
    return {"median": float(np.median(x)), "q25": float(np.quantile(x, 0.25)),
            "q75": float(np.quantile(x, 0.75)), "worst": float(np.min(x))}

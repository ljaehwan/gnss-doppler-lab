"""Attack-label-free global coordinate selectors for PG-SCC."""
from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

from gnss_doppler_lab.pg_scc_physics import (
    CENTER, COORDINATES, DEFAULT_SEARCH, N_COORDINATES,
    analytic_same_prn_template, inject_same_prn_second_source, normalize_complex,
    two_source_glrt,
)


@dataclass(frozen=True)
class SyntheticBank:
    surfaces: np.ndarray
    labels: np.ndarray
    split: np.ndarray
    parameters: tuple[dict[str, float | str], ...]


def _combo_split(combo: tuple[float, ...]) -> str:
    payload = ",".join(f"{value:.9g}" for value in combo).encode()
    return "validation" if int(hashlib.sha256(payload).hexdigest()[:8], 16) % 4 == 0 else "train"


def build_synthetic_bank(
    clean_train: np.ndarray,
    clean_selection: np.ndarray,
    *,
    normalization: str,
    seed: int,
    max_h1_per_split: int = 480,
) -> SyntheticBank:
    """Create disjoint physical-parameter train/validation H0/H1 banks."""
    rng = np.random.default_rng(seed)
    clean_by_split = {
        "train": np.asarray([normalize_complex(x, normalization) for x in clean_train]),
        "validation": np.asarray([normalize_complex(x, normalization) for x in clean_selection]),
    }
    combos = list(itertools.product(
        (-0.75, -0.375, -0.125, 0.0, 0.125, 0.375, 0.75),
        (-150.0, -75.0, 0.0, 75.0, 150.0),
        tuple(np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)),
        (0.25, 0.5, 1.0, 1.5),
        (0.0, 0.02, 0.05),
    ))
    combos = [combo for combo in combos if combo[0] != 0.0 or combo[1] != 0.0]
    by_split = {name: [combo for combo in combos if _combo_split(combo) == name] for name in clean_by_split}
    surfaces: list[np.ndarray] = []
    labels: list[int] = []
    splits: list[str] = []
    parameters: list[dict[str, float | str]] = []
    for split, clean in clean_by_split.items():
        # H0 examples are retained separately so the mask objective penalizes clean evidence.
        for base in clean:
            surfaces.append(base); labels.append(0); splits.append(split)
            parameters.append({"split": split, "class": "H0"})
        order = rng.permutation(len(by_split[split]))[:max_h1_per_split]
        for number, combo_index in enumerate(order):
            tau, doppler, phase, amplitude, noise = by_split[split][int(combo_index)]
            base = clean[number % len(clean)]
            surfaces.append(inject_same_prn_second_source(
                base, delta_tau_chips=tau, delta_doppler_hz=doppler,
                relative_amplitude=amplitude, relative_phase_rad=phase,
                noise_sigma=noise, rng=rng, normalization=normalization,
            ))
            labels.append(1); splits.append(split)
            parameters.append({
                "split": split, "class": "H1", "delta_tau_chips": tau,
                "delta_doppler_hz": doppler, "relative_phase_rad": phase,
                "relative_amplitude": amplitude, "noise_sigma": noise,
            })
    return SyntheticBank(
        surfaces=np.asarray(surfaces, dtype=np.complex128), labels=np.asarray(labels, dtype=np.int8),
        split=np.asarray(splits), parameters=tuple(parameters),
    )


def residual_evidence(surfaces: np.ndarray, auth_template: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Per-coordinate H0 mismatch used only as a differentiable selector proxy."""
    values = np.asarray(surfaces, complex)
    template = np.asarray(auth_template, complex)
    weights = 1.0 / np.maximum(np.real(np.diag(covariance)), 1e-8)
    denom = np.sum(weights * np.abs(template) ** 2)
    alpha = np.sum(values * np.conj(template)[None, :] * weights[None, :], axis=1) / max(denom, 1e-12)
    residual = values - alpha[:, None] * template[None, :]
    return (np.abs(residual) ** 2 * weights[None, :]).astype(np.float64)


def dense_teacher_scores(surfaces: np.ndarray, auth_template: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    return np.asarray([two_source_glrt(x, auth_template, covariance, search=DEFAULT_SEARCH).score for x in surfaces])


def greedy_teacher_mask(features: np.ndarray, teacher: np.ndarray, budget: int) -> list[int]:
    """Greedy ridge teacher-preservation (S0), always retaining the prompt."""
    x = np.asarray(features, float)
    target = np.asarray(teacher, float)
    target = (target - target.mean()) / max(target.std(), 1e-9)
    x = (x - x.mean(0, keepdims=True)) / np.maximum(x.std(0, keepdims=True), 1e-9)
    selected = [CENTER]
    while len(selected) < budget:
        best: tuple[float, int] | None = None
        for candidate in range(N_COORDINATES):
            if candidate in selected:
                continue
            design = np.column_stack((np.ones(len(x)), x[:, [*selected, candidate]]))
            gram = design.T @ design + np.eye(design.shape[1]) * 1e-3
            beta = np.linalg.solve(gram, design.T @ target)
            mse = float(np.mean((target - design @ beta) ** 2))
            value = (mse, candidate)
            if best is None or value < best:
                best = value
        if best is None:
            raise RuntimeError("greedy selector exhausted candidates")
        selected.append(best[1])
    return selected


def train_global_topk_mask(
    features: np.ndarray,
    teacher: np.ndarray,
    labels: np.ndarray,
    budget: int,
    *,
    seed: int,
    epochs: int = 350,
) -> tuple[list[int], dict[str, object]]:
    """Train one input-independent differentiable top-K mask (S1)."""
    x_raw = np.asarray(features, np.float64)
    mean, std = x_raw.mean(0), np.maximum(x_raw.std(0), 1e-6)
    x = (x_raw - mean) / std
    target_raw = np.asarray(teacher, np.float64)
    target = (target_raw - target_raw.mean()) / max(float(target_raw.std()), 1e-6)
    clean = np.asarray(labels) == 0
    logits = np.zeros(N_COORDINATES, dtype=np.float64)
    scale, bias = 0.1, 0.0
    first = {"logits": np.zeros_like(logits), "scale": 0.0, "bias": 0.0}
    second = {"logits": np.zeros_like(logits), "scale": 0.0, "bias": 0.0}
    rng = np.random.default_rng(seed + budget)
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        temperature = max(0.15, 1.5 * (0.99 ** epoch))
        shifted = logits / temperature - np.max(logits / temperature)
        probability = np.exp(shifted) / np.exp(shifted).sum()
        weight = budget * probability
        weighted_feature = x @ weight
        prediction = scale * weighted_feature + bias
        error = prediction - target
        mse = float(np.mean(error ** 2))
        clean_positive = np.maximum(prediction[clean], 0.0)
        clean_penalty = float(np.mean(clean_positive ** 2))
        left = rng.integers(0, len(x), size=256)
        right = rng.integers(0, len(x), size=256)
        direction = np.sign(target[left] - target[right])
        valid = direction != 0
        margin = 0.1 - direction[valid] * (prediction[left][valid] - prediction[right][valid])
        active = margin > 0
        ranking = float(np.mean(np.maximum(margin, 0.0)))
        concentration = float(np.sum((weight / budget) ** 2))
        loss = mse + 0.15 * clean_penalty + 0.10 * ranking + 0.01 * concentration
        gradient_prediction = 2.0 * error / len(error)
        clean_indices = np.flatnonzero(clean)
        gradient_prediction[clean_indices] += 0.15 * 2.0 * clean_positive / max(len(clean_indices), 1)
        valid_left, valid_right = left[valid][active], right[valid][active]
        valid_direction = direction[valid][active]
        for left_index, right_index, sign in zip(valid_left, valid_right, valid_direction):
            gradient_prediction[left_index] += -0.10 * sign / max(valid.sum(), 1)
            gradient_prediction[right_index] += 0.10 * sign / max(valid.sum(), 1)
        gradient_scale = float(np.dot(gradient_prediction, weighted_feature))
        gradient_bias = float(gradient_prediction.sum())
        gradient_weight = scale * (x.T @ gradient_prediction) + 0.02 * weight / (budget ** 2)
        gradient_logits = (budget / temperature) * probability * (
            gradient_weight - float(np.dot(gradient_weight, probability))
        )
        gradients = {"logits": gradient_logits, "scale": gradient_scale, "bias": gradient_bias}
        for name, gradient in gradients.items():
            first[name] = 0.9 * first[name] + 0.1 * gradient
            second[name] = 0.999 * second[name] + 0.001 * gradient * gradient
            first_hat = first[name] / (1.0 - 0.9 ** (epoch + 1))
            second_hat = second[name] / (1.0 - 0.999 ** (epoch + 1))
            update = 0.04 * first_hat / (np.sqrt(second_hat) + 1e-8)
            if name == "logits":
                logits -= update
            elif name == "scale":
                scale -= float(update)
            else:
                bias -= float(update)
        if epoch % 25 == 0 or epoch == epochs - 1:
            history.append({
                "epoch": float(epoch), "loss": float(loss), "mse": mse,
                "clean_false_evidence_penalty": clean_penalty,
                "ranking_loss": ranking, "temperature": float(temperature),
                "soft_exact_k_sum": float(weight.sum()),
            })
    learned = logits.copy()
    learned[CENTER] = np.inf
    selected = np.argsort(-learned, kind="stable")[:budget].astype(int).tolist()
    if CENTER not in selected:
        selected[-1] = CENTER
    selected = [CENTER, *sorted(i for i in set(selected) if i != CENTER)]
    if len(selected) != budget:
        raise RuntimeError("top-K projection failed exact budget")
    return selected, {
        "seed": seed + budget, "epochs": epochs,
        "optimizer": "explicit NumPy Adam on differentiable soft-top-K weights",
        "objective": {
            "teacher_mse": 1.0, "clean_h0_false_evidence": 0.15,
            "synthetic_h1_pairwise_ranking": 0.10,
            "phase_gain_noise_robustness": "input normalization and bank sweep",
            "exact_k": "K*softmax during training; deterministic top-K projection after training",
        },
        "history": history,
        "logits": np.where(
            np.isfinite(learned), learned,
            float(np.max(learned[np.isfinite(learned)]) + 1),
        ).tolist(),
    }


def symmetric_mask_from_logits(logits: Sequence[float], budget: int) -> list[int]:
    """Build center + inversion-symmetric coordinate pairs for odd K."""
    if budget % 2 != 1:
        raise ValueError("symmetric mask requires odd K")
    values = np.asarray(logits, float)
    pairs: list[tuple[float, int, int]] = []
    seen = {CENTER}
    for index, (tau, doppler) in enumerate(COORDINATES):
        if index in seen or index == CENTER:
            continue
        opposite = int(np.argmin(np.sum((COORDINATES - np.asarray([-tau, -doppler])) ** 2, axis=1)))
        if opposite == index or opposite in seen:
            continue
        seen.update((index, opposite))
        pairs.append((-(values[index] + values[opposite]), min(index, opposite), max(index, opposite)))
    pairs.sort()
    selected = [CENTER]
    for _, left, right in pairs[: (budget - 1) // 2]:
        selected.extend((left, right))
    if len(selected) != budget:
        raise RuntimeError("symmetric exact-K construction failed")
    return selected


def mask_validation(
    masks: Mapping[str, Sequence[int]], surfaces: np.ndarray, teacher: np.ndarray,
    auth_template: np.ndarray, covariance: np.ndarray,
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for name, indices in masks.items():
        score = np.asarray([two_source_glrt(x, auth_template, covariance, indices=indices).score for x in surfaces])
        spearman = float(spearmanr(teacher, score).statistic)
        result[name] = {
            "pearson": float(np.corrcoef(teacher, score)[0, 1]), "rank_proxy_correlation": spearman,
            "normalized_rmse": float(np.sqrt(np.mean(((score - score.mean()) / max(score.std(), 1e-9) - (teacher - teacher.mean()) / max(teacher.std(), 1e-9)) ** 2))),
        }
    return result

"""Soft-membership robust dynamic-panel state-space model for SPLITCLOCK R1."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from .splitclock_r1_contract import FIT_EPOCHS, HOLDOUT_EPOCHS, MIN_CLUSTER_MASS, STUDENT_T_DF, WINDOW_EPOCHS

H = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
F = np.asarray([[1.0, 1.0], [0.0, 1.0]])


@dataclass(frozen=True)
class ScoreResult:
    score: float
    normalized_score: float
    raw_gain: float
    penalty: float
    k1_heldout_loglik: float
    k2_heldout_loglik: float
    k1_fit_loglik: float
    k2_fit_loglik: float
    memberships: np.ndarray
    eligible: np.ndarray
    evaluation_mask: np.ndarray
    n_valid: int
    delta_p: int
    cluster_masses: tuple[float, float]
    selected_restart: int
    fit_digest: str


def student_logpdf(residual: np.ndarray | float, scale: np.ndarray | float) -> np.ndarray:
    r = np.asarray(residual, dtype=float); s = np.maximum(np.asarray(scale, dtype=float), 1e-9); df = STUDENT_T_DF
    constant = math.lgamma((df + 1.0) / 2.0) - math.lgamma(df / 2.0) - 0.5 * math.log(df * math.pi)
    return constant - np.log(s) - 0.5 * (df + 1.0) * np.log1p((r / s) ** 2 / df)


def robust_scales(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    scales = []
    for modality in range(values.shape[2]):
        observed = values[:, :, modality][valid[:, :, modality]]
        if len(observed) < 5: raise ValueError("insufficient observations for modality scale")
        center = np.median(observed); mad = np.median(np.abs(observed - center)) / 0.6744897501960817
        floors = (0.5, 0.02, 0.02); scales.append(max(float(mad), floors[modality]))
    return np.asarray(scales)


def calibrate_noise(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit modality and process scales from a chronological C-1 development panel."""

    centered = np.asarray(values, dtype=float).copy(); mask = np.asarray(valid, dtype=bool) & np.isfinite(centered)
    for prn in range(centered.shape[1]):
        for modality in range(3):
            use = mask[:, prn, modality]
            if np.any(use): centered[:, prn, modality] -= np.median(centered[:, prn, modality][use])
    residual = centered.copy(); paths = np.full((len(centered), 2), np.nan)
    for t in range(len(centered)):
        code = centered[t, :, 0][mask[t, :, 0]]
        dynamic = centered[t, :, 1:3][mask[t, :, 1:3]]
        paths[t] = (np.median(code) if len(code) else np.nan, np.median(dynamic) if len(dynamic) else np.nan)
        for modality, component in ((0, 0), (1, 1), (2, 1)):
            residual[t, :, modality] -= paths[t, component]
    scales = robust_scales(residual, mask)
    innovations = []
    for t in range(1, len(paths)):
        if np.isfinite(paths[t]).all() and np.isfinite(paths[t - 1]).all(): innovations.append(paths[t] - F @ paths[t - 1])
    if not innovations: raise ValueError("no finite C-1 process innovations")
    innovations = np.asarray(innovations); process = []
    for component, floor in enumerate((0.05, 0.005)):
        data = innovations[:, component]; process.append(max(float(np.median(np.abs(data - np.median(data))) / 0.6744897501960817), floor))
    return scales, np.asarray(process)


def _eligible(mask: np.ndarray, modalities: tuple[int, ...]) -> np.ndarray:
    fit = mask[:FIT_EPOCHS]
    if len(modalities) == 1:
        return np.sum(fit[:, :, modalities[0]], axis=0) >= 4
    code = np.sum(fit[:, :, 0], axis=0)
    dynamic = np.sum(fit[:, :, 1] | fit[:, :, 2], axis=0)
    return (code >= 4) & (dynamic >= 4)


def _center_fit_only(values: np.ndarray, mask: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    for prn in np.flatnonzero(eligible):
        for modality in range(3):
            use = mask[:FIT_EPOCHS, prn, modality]
            if np.any(use): result[:, prn, modality] -= np.median(result[:FIT_EPOCHS, prn, modality][use])
    return result


def _path_fit(values: np.ndarray, mask: np.ndarray, membership: np.ndarray, scales: np.ndarray, process: np.ndarray) -> np.ndarray:
    epochs = FIT_EPOCHS; dimension = 2 * epochs
    states = np.zeros((epochs, 2), dtype=float)
    for t in range(epochs):
        for component, modalities in enumerate(((0,), (1, 2))):
            samples = []
            for modality in modalities:
                use = mask[t, :, modality] & (membership > 1e-8)
                samples.extend(values[t, use, modality].tolist())
            states[t, component] = float(np.median(samples)) if samples else (states[t - 1, component] if t else 0.0)
    for _ in range(12):
        normal = np.eye(dimension) * 1e-9; rhs = np.zeros(dimension)
        for t in range(epochs):
            for prn in range(values.shape[1]):
                if membership[prn] <= 1e-8: continue
                for modality in range(3):
                    if not mask[t, prn, modality]: continue
                    h = H[modality]; residual = values[t, prn, modality] - h @ states[t]
                    robust = (STUDENT_T_DF + 1.0) / (STUDENT_T_DF + (residual / scales[modality]) ** 2)
                    weight = membership[prn] * robust / scales[modality] ** 2
                    sl = slice(2 * t, 2 * t + 2); normal[sl, sl] += weight * np.outer(h, h); rhs[sl] += weight * h * values[t, prn, modality]
        q_inv = np.diag(1.0 / np.maximum(process, 1e-6) ** 2)
        for t in range(1, epochs):
            current = slice(2 * t, 2 * t + 2); prior = slice(2 * (t - 1), 2 * (t - 1) + 2)
            normal[current, current] += q_inv; normal[prior, prior] += F.T @ q_inv @ F
            normal[current, prior] -= q_inv @ F; normal[prior, current] -= F.T @ q_inv
        updated = np.linalg.solve(normal, rhs).reshape(epochs, 2)
        if np.max(np.abs(updated - states)) < 1e-8: states = updated; break
        states = updated
    return states


def _path_prn_loglik(values: np.ndarray, mask: np.ndarray, states: np.ndarray, scales: np.ndarray) -> np.ndarray:
    result = np.zeros(values.shape[1])
    for prn in range(values.shape[1]):
        for t in range(FIT_EPOCHS):
            for modality in range(3):
                if mask[t, prn, modality]: result[prn] += float(student_logpdf(values[t, prn, modality] - H[modality] @ states[t], scales[modality]))
    return result


def _fit_mixture_loglik(values: np.ndarray, mask: np.ndarray, states: tuple[np.ndarray, np.ndarray], pi: np.ndarray, scales: np.ndarray) -> float:
    total = 0.0
    for t in range(FIT_EPOCHS):
        for prn in range(values.shape[1]):
            for modality in range(3):
                if not mask[t, prn, modality]: continue
                ll0 = float(student_logpdf(values[t, prn, modality] - H[modality] @ states[0][t], scales[modality])) + math.log(max(1.0 - pi[prn], 1e-12))
                ll1 = float(student_logpdf(values[t, prn, modality] - H[modality] @ states[1][t], scales[modality])) + math.log(max(pi[prn], 1e-12))
                total += float(np.logaddexp(ll0, ll1))
    return total


def _initializations(values: np.ndarray, mask: np.ndarray, eligible: np.ndarray) -> list[np.ndarray]:
    count = values.shape[1]; neutral = np.full(count, 0.5); summaries = []
    for prn in range(count):
        if not eligible[prn]: summaries.append([0.0, 0.0]); continue
        code = values[:FIT_EPOCHS, prn, 0]; use = mask[:FIT_EPOCHS, prn, 0]; times = np.arange(FIT_EPOCHS)[use]
        slope = np.polyfit(times, code[use], 1)[0] if len(times) >= 2 else 0.0
        dynamic_values = values[:FIT_EPOCHS, prn, 1:3][mask[:FIT_EPOCHS, prn, 1:3]]
        summaries.append([slope, float(np.median(dynamic_values)) if len(dynamic_values) else 0.0])
    matrix = np.asarray(summaries)[eligible]; matrix -= np.median(matrix, axis=0)
    if len(matrix) >= 2 and np.any(matrix):
        _, _, vh = np.linalg.svd(matrix, full_matrices=False); projection = matrix @ vh[0]; scale = max(np.median(np.abs(projection - np.median(projection))) / 0.6744897501960817, 1e-6)
        soft = 1.0 / (1.0 + np.exp(-2.0 * projection / scale))
    else: soft = np.full(np.sum(eligible), 0.5)
    positive = neutral.copy(); positive[eligible] = np.clip(soft, 1e-3, 1.0 - 1e-3)
    negative = neutral.copy(); negative[eligible] = 1.0 - positive[eligible]
    return [neutral, positive, negative]


def _ensure_mass(pi: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    result = pi.copy(); n = int(np.sum(eligible))
    if n < 4: return result
    for _ in range(20):
        masses = (float(np.sum(1.0 - result[eligible])), float(np.sum(result[eligible])))
        if min(masses) >= MIN_CLUSTER_MASS: break
        result[eligible] = 0.9 * result[eligible] + 0.1 * 0.5
    return result


def score_window(values: np.ndarray, valid: np.ndarray, scales: np.ndarray, process: np.ndarray, modalities: tuple[int, ...] = (0, 1, 2), hard_assignment: bool = False) -> ScoreResult:
    values = np.asarray(values, dtype=float); mask = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if values.ndim != 3 or values.shape[0] != WINDOW_EPOCHS or values.shape[2] != 3: raise ValueError("panel must be 10 x PRN x 3")
    modality_mask = np.zeros(3, dtype=bool); modality_mask[list(modalities)] = True; mask &= modality_mask[None, None, :]
    if set(modalities) == {0, 1, 2}:
        epoch_prns = np.sum(mask[:, :, 0] & (mask[:, :, 1] | mask[:, :, 2]), axis=1)
    else:
        epoch_prns = np.sum(np.any(mask[:, :, list(modalities)], axis=2), axis=1)
    if np.any(epoch_prns < 5): raise ValueError("M>=5 epoch gate failed")
    eligible = _eligible(mask, modalities)
    if np.sum(eligible) < 5: raise ValueError("fewer than five fit-eligible PRNs")
    mask[:, ~eligible, :] = False
    centered = _center_fit_only(values, mask, eligible)
    k1_states = _path_fit(centered, mask, eligible.astype(float), scales, process)
    k1_fit = float(np.sum(_path_prn_loglik(centered, mask, k1_states, scales)[eligible]))
    candidates = []
    for restart, initialization in enumerate(_initializations(centered, mask, eligible)):
        pi = initialization.copy(); pi[~eligible] = 0.5
        for _ in range(30):
            states0 = _path_fit(centered, mask, (1.0 - pi) * eligible, scales, process); states1 = _path_fit(centered, mask, pi * eligible, scales, process)
            ll0 = _path_prn_loglik(centered, mask, states0, scales); ll1 = _path_prn_loglik(centered, mask, states1, scales)
            updated = pi.copy(); difference = np.clip(ll1[eligible] - ll0[eligible], -30.0, 30.0); updated[eligible] = 1.0 / (1.0 + np.exp(-difference)); updated[eligible] = np.clip(updated[eligible], 1e-6, 1.0 - 1e-6); updated = _ensure_mass(updated, eligible)
            if np.max(np.abs(updated[eligible] - pi[eligible])) < 1e-8: pi = updated; break
            pi = updated
        states = (states0, states1)
        if np.mean(states[0][:, 0]) > np.mean(states[1][:, 0]): states = (states[1], states[0]); pi = 1.0 - pi
        fit_ll = _fit_mixture_loglik(centered, mask, states, pi, scales)
        candidates.append((fit_ll, -restart, restart, states, pi))
    fit_ll, _, restart, states, pi = max(candidates, key=lambda item: (item[0], item[1]))
    if hard_assignment:
        hard = (pi >= 0.5).astype(float)
        if min(np.sum(hard[eligible]), np.sum(1.0 - hard[eligible])) < MIN_CLUSTER_MASS:
            order = np.argsort(pi[eligible], kind="mergesort"); indices = np.flatnonzero(eligible); hard[indices] = 1.0; hard[indices[order[:2]]] = 0.0
        pi = hard
        states = (_path_fit(centered, mask, (1.0 - pi) * eligible, scales, process), _path_fit(centered, mask, pi * eligible, scales, process))
        fit_ll = _fit_mixture_loglik(centered, mask, states, np.clip(pi, 1e-12, 1.0 - 1e-12), scales)
    evaluation = mask[FIT_EPOCHS:].copy(); n_valid = int(np.sum(evaluation))
    if not n_valid: raise ValueError("no heldout observations")
    k1_hold = k2_hold = 0.0; k1_prediction = k1_states[-1].copy(); path_prediction = [states[0][-1].copy(), states[1][-1].copy()]
    for relative_t in range(HOLDOUT_EPOCHS):
        k1_prediction = F @ k1_prediction; path_prediction = [F @ path_prediction[0], F @ path_prediction[1]]
        absolute_t = FIT_EPOCHS + relative_t
        for prn in range(values.shape[1]):
            for modality in range(3):
                if not evaluation[relative_t, prn, modality]: continue
                y = centered[absolute_t, prn, modality]
                ll1 = float(student_logpdf(y - H[modality] @ k1_prediction, scales[modality])); k1_hold += ll1
                ll0 = float(student_logpdf(y - H[modality] @ path_prediction[0], scales[modality])) + math.log(max(1.0 - pi[prn], 1e-12))
                ll2 = float(student_logpdf(y - H[modality] @ path_prediction[1], scales[modality])) + math.log(max(pi[prn], 1e-12)); k2_hold += float(np.logaddexp(ll0, ll2))
    raw_gain = k2_hold - k1_hold; delta_p = 2 + int(np.sum(eligible)); penalty = 0.5 * delta_p * math.log(n_valid); score = raw_gain - penalty
    digest_values = np.concatenate([k1_states.ravel(), states[0].ravel(), states[1].ravel(), pi[eligible]])
    digest = hashlib.sha256(np.asarray(digest_values, dtype="<f8").tobytes()).hexdigest()
    masses = (float(np.sum(1.0 - pi[eligible])), float(np.sum(pi[eligible])))
    return ScoreResult(score, score / n_valid, raw_gain, penalty, k1_hold, k2_hold, k1_fit, fit_ll, pi, eligible, evaluation, n_valid, delta_p, masses, restart, digest)


def inject_clock(values: np.ndarray, subset: np.ndarray, onset: int, d0_m: float, velocity_mps: float, acceleration_mps2: float) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy(); relative = np.arange(len(result), dtype=float) - onset; active = relative >= 0
    trajectory = np.zeros(len(result)); trajectory[active] = d0_m + velocity_mps * relative[active] + 0.5 * acceleration_mps2 * relative[active] ** 2
    rate = np.zeros(len(result)); rate[active] = velocity_mps + acceleration_mps2 * relative[active]
    increment = np.diff(np.r_[0.0, trajectory])
    result[:, subset, 0] += trajectory[:, None]; result[:, subset, 1] += rate[:, None]; result[:, subset, 2] += increment[:, None]
    return result

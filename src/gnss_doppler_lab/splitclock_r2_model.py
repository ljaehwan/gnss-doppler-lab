"""Terminal R2 soft persistent-membership SPLITCLOCK state-space model."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import numpy as np

from .splitclock_r1_model import calibrate_noise, inject_clock, robust_scales
from .splitclock_r2_contract import (
    FIT_EPOCHS,
    HOLDOUT_EPOCHS,
    MIN_CLUSTER_MASS,
    STUDENT_T_DF,
    WINDOW_EPOCHS,
)

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
    observation_wise_k2_heldout_loglik: float
    observation_wise_score: float
    memberships: np.ndarray
    eligible: np.ndarray
    evaluation_mask: np.ndarray
    centering: np.ndarray
    n_valid: int
    delta_p: int
    cluster_masses: tuple[float, float]
    selected_restart: int
    fit_digest: str


def student_logpdf(residual: np.ndarray | float, scale: np.ndarray | float) -> np.ndarray:
    r = np.asarray(residual, dtype=float)
    s = np.maximum(np.asarray(scale, dtype=float), 1e-9)
    constant = (
        math.lgamma((STUDENT_T_DF + 1.0) / 2.0)
        - math.lgamma(STUDENT_T_DF / 2.0)
        - 0.5 * math.log(STUDENT_T_DF * math.pi)
    )
    return constant - np.log(s) - 0.5 * (STUDENT_T_DF + 1.0) * np.log1p(
        (r / s) ** 2 / STUDENT_T_DF
    )


def _eligible(mask: np.ndarray, modalities: tuple[int, ...]) -> np.ndarray:
    fit = mask[:FIT_EPOCHS]
    if len(modalities) == 1:
        return np.sum(fit[:, :, modalities[0]], axis=0) >= 4
    code = np.sum(fit[:, :, 0], axis=0)
    dynamic = np.sum(fit[:, :, 1] | fit[:, :, 2], axis=0)
    return (code >= 4) & (dynamic >= 4)


def center_fit_only(
    values: np.ndarray, mask: np.ndarray, eligible: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Apply R2-F1 centering using fit observations only."""

    result = np.asarray(values, dtype=float).copy()
    centers = np.zeros((values.shape[1], 3), dtype=float)
    for prn in np.flatnonzero(eligible):
        use = mask[:FIT_EPOCHS, prn, 0]
        if np.any(use):
            centers[prn, 0] = float(np.median(result[:FIT_EPOCHS, prn, 0][use]))
            result[:, prn, 0] -= centers[prn, 0]
    for modality in (1, 2):
        use = mask[:FIT_EPOCHS, :, modality] & eligible[None, :]
        if np.any(use):
            global_center = float(np.median(result[:FIT_EPOCHS, :, modality][use]))
            centers[eligible, modality] = global_center
            result[:, eligible, modality] -= global_center
    return result, centers


def _path_fit(
    values: np.ndarray,
    mask: np.ndarray,
    membership: np.ndarray,
    scales: np.ndarray,
    process: np.ndarray,
) -> np.ndarray:
    dimension = 2 * FIT_EPOCHS
    states = np.zeros((FIT_EPOCHS, 2), dtype=float)
    for t in range(FIT_EPOCHS):
        for component, modalities in enumerate(((0,), (1, 2))):
            samples: list[float] = []
            for modality in modalities:
                use = mask[t, :, modality] & (membership > 1e-8)
                samples.extend(values[t, use, modality].tolist())
            states[t, component] = (
                float(np.median(samples))
                if samples
                else (states[t - 1, component] if t else 0.0)
            )
    for _ in range(12):
        normal = np.eye(dimension) * 1e-9
        rhs = np.zeros(dimension)
        for t in range(FIT_EPOCHS):
            for prn in range(values.shape[1]):
                if membership[prn] <= 1e-8:
                    continue
                for modality in range(3):
                    if not mask[t, prn, modality]:
                        continue
                    h = H[modality]
                    residual = values[t, prn, modality] - h @ states[t]
                    robust = (STUDENT_T_DF + 1.0) / (
                        STUDENT_T_DF + (residual / scales[modality]) ** 2
                    )
                    weight = membership[prn] * robust / scales[modality] ** 2
                    sl = slice(2 * t, 2 * t + 2)
                    normal[sl, sl] += weight * np.outer(h, h)
                    rhs[sl] += weight * h * values[t, prn, modality]
        q_inv = np.diag(1.0 / np.maximum(process, 1e-6) ** 2)
        for t in range(1, FIT_EPOCHS):
            current = slice(2 * t, 2 * t + 2)
            prior = slice(2 * (t - 1), 2 * (t - 1) + 2)
            normal[current, current] += q_inv
            normal[prior, prior] += F.T @ q_inv @ F
            normal[current, prior] -= q_inv @ F
            normal[prior, current] -= F.T @ q_inv
        updated = np.linalg.solve(normal, rhs).reshape(FIT_EPOCHS, 2)
        if np.max(np.abs(updated - states)) < 1e-8:
            states = updated
            break
        states = updated
    return states


def _path_prn_loglik(
    values: np.ndarray,
    mask: np.ndarray,
    states: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    result = np.zeros(values.shape[1])
    for prn in range(values.shape[1]):
        for t in range(FIT_EPOCHS):
            for modality in range(3):
                if mask[t, prn, modality]:
                    result[prn] += float(
                        student_logpdf(
                            values[t, prn, modality] - H[modality] @ states[t],
                            scales[modality],
                        )
                    )
    return result


def _ensure_mass(q: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    result = q.copy()
    if int(np.sum(eligible)) < 4:
        return result
    for _ in range(20):
        masses = (
            float(np.sum(1.0 - result[eligible])),
            float(np.sum(result[eligible])),
        )
        if min(masses) >= MIN_CLUSTER_MASS:
            break
        result[eligible] = 0.9 * result[eligible] + 0.1 * 0.5
    return result


def persistent_prn_membership(
    fit_ll0: np.ndarray, fit_ll1: np.ndarray, eligible: np.ndarray
) -> np.ndarray:
    """Return one equal-prior fit posterior membership per PRN."""

    q = np.full(len(fit_ll0), 0.5)
    difference = np.clip(fit_ll1[eligible] - fit_ll0[eligible], -30.0, 30.0)
    q[eligible] = np.clip(
        1.0 / (1.0 + np.exp(-difference)), 1e-6, 1.0 - 1e-6
    )
    return _ensure_mass(q, eligible)


def persistent_prn_mixture_loglik(
    path0_loglik: np.ndarray,
    path1_loglik: np.ndarray,
    q: np.ndarray,
    eligible: np.ndarray,
) -> float:
    """Marginalize the fixed latent path exactly once per eligible PRN."""

    total = 0.0
    for prn in np.flatnonzero(eligible):
        total += float(
            np.logaddexp(
                math.log(max(1.0 - q[prn], 1e-12)) + path0_loglik[prn],
                math.log(max(q[prn], 1e-12)) + path1_loglik[prn],
            )
        )
    return total


def observation_wise_mixture_diagnostic(
    path0_terms: np.ndarray,
    path1_terms: np.ndarray,
    q: np.ndarray,
    mask: np.ndarray,
) -> float:
    """R1 shortcut retained only as a named non-primary diagnostic."""

    total = 0.0
    for t, prn, modality in zip(*np.nonzero(mask)):
        total += float(
            np.logaddexp(
                math.log(max(1.0 - q[prn], 1e-12))
                + path0_terms[t, prn, modality],
                math.log(max(q[prn], 1e-12)) + path1_terms[t, prn, modality],
            )
        )
    return total


def _initializations(
    values: np.ndarray, mask: np.ndarray, eligible: np.ndarray
) -> list[np.ndarray]:
    count = values.shape[1]
    neutral = np.full(count, 0.5)
    summaries: list[list[float]] = []
    for prn in range(count):
        if not eligible[prn]:
            summaries.append([0.0, 0.0])
            continue
        code = values[:FIT_EPOCHS, prn, 0]
        use = mask[:FIT_EPOCHS, prn, 0]
        times = np.arange(FIT_EPOCHS)[use]
        slope = np.polyfit(times, code[use], 1)[0] if len(times) >= 2 else 0.0
        dynamic = values[:FIT_EPOCHS, prn, 1:3][
            mask[:FIT_EPOCHS, prn, 1:3]
        ]
        summaries.append(
            [slope, float(np.median(dynamic)) if len(dynamic) else 0.0]
        )
    matrix = np.asarray(summaries)[eligible]
    matrix -= np.median(matrix, axis=0)
    if len(matrix) >= 2 and np.any(matrix):
        _, _, vh = np.linalg.svd(matrix, full_matrices=False)
        projection = matrix @ vh[0]
        scale = max(
            np.median(np.abs(projection - np.median(projection)))
            / 0.6744897501960817,
            1e-6,
        )
        soft = 1.0 / (1.0 + np.exp(-2.0 * projection / scale))
    else:
        soft = np.full(np.sum(eligible), 0.5)
    positive = neutral.copy()
    positive[eligible] = np.clip(soft, 1e-3, 1.0 - 1e-3)
    negative = neutral.copy()
    negative[eligible] = 1.0 - positive[eligible]
    return [neutral, positive, negative]


def _canonicalize(
    states: tuple[np.ndarray, np.ndarray], q: np.ndarray
) -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
    summaries = [
        (float(np.mean(path[:, 0])), float(np.mean(path[:, 1]))) for path in states
    ]
    if summaries[0] > summaries[1]:
        return (states[1], states[0]), 1.0 - q
    return states, q


def score_window(
    values: np.ndarray,
    valid: np.ndarray,
    scales: np.ndarray,
    process: np.ndarray,
    modalities: tuple[int, ...] = (0, 1, 2),
    hard_assignment: bool = False,
) -> ScoreResult:
    values = np.asarray(values, dtype=float)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(values)
    if (
        values.ndim != 3
        or values.shape[0] != WINDOW_EPOCHS
        or values.shape[2] != 3
    ):
        raise ValueError("panel must be 10 x PRN x 3")
    modality_mask = np.zeros(3, dtype=bool)
    modality_mask[list(modalities)] = True
    mask &= modality_mask[None, None, :]
    if set(modalities) == {0, 1, 2}:
        epoch_prns = np.sum(
            mask[:, :, 0] & (mask[:, :, 1] | mask[:, :, 2]), axis=1
        )
    else:
        epoch_prns = np.sum(
            np.any(mask[:, :, list(modalities)], axis=2), axis=1
        )
    if np.any(epoch_prns < 5):
        raise ValueError("M>=5 epoch gate failed")
    eligible = _eligible(mask, modalities)
    if np.sum(eligible) < 5:
        raise ValueError("fewer than five fit-eligible PRNs")
    mask[:, ~eligible, :] = False
    centered, centering = center_fit_only(values, mask, eligible)

    k1_states = _path_fit(
        centered, mask, eligible.astype(float), scales, process
    )
    k1_fit_terms = _path_prn_loglik(centered, mask, k1_states, scales)
    k1_fit = float(np.sum(k1_fit_terms[eligible]))

    candidates: list[tuple[float, int, int, tuple[np.ndarray, np.ndarray], np.ndarray]] = []
    for restart, initialization in enumerate(
        _initializations(centered, mask, eligible)
    ):
        q = initialization.copy()
        q[~eligible] = 0.5
        for _ in range(30):
            states0 = _path_fit(
                centered, mask, (1.0 - q) * eligible, scales, process
            )
            states1 = _path_fit(centered, mask, q * eligible, scales, process)
            ll0 = _path_prn_loglik(centered, mask, states0, scales)
            ll1 = _path_prn_loglik(centered, mask, states1, scales)
            updated = persistent_prn_membership(ll0, ll1, eligible)
            if np.max(np.abs(updated[eligible] - q[eligible])) < 1e-8:
                q = updated
                break
            q = updated
        states0 = _path_fit(
            centered, mask, (1.0 - q) * eligible, scales, process
        )
        states1 = _path_fit(centered, mask, q * eligible, scales, process)
        ll0 = _path_prn_loglik(centered, mask, states0, scales)
        ll1 = _path_prn_loglik(centered, mask, states1, scales)
        q = persistent_prn_membership(ll0, ll1, eligible)
        states, q = _canonicalize((states0, states1), q)
        if states[0] is states1:
            ll0, ll1 = ll1, ll0
        fit_ll = persistent_prn_mixture_loglik(ll0, ll1, q, eligible)
        candidates.append((fit_ll, -restart, restart, states, q))
    fit_ll, _, restart, states, q = max(
        candidates, key=lambda item: (item[0], item[1])
    )

    if hard_assignment:
        hard = (q >= 0.5).astype(float)
        if min(
            np.sum(hard[eligible]), np.sum(1.0 - hard[eligible])
        ) < MIN_CLUSTER_MASS:
            order = np.argsort(q[eligible], kind="mergesort")
            indices = np.flatnonzero(eligible)
            hard[indices] = 1.0
            hard[indices[order[:2]]] = 0.0
        q = hard
        states = (
            _path_fit(centered, mask, (1.0 - q) * eligible, scales, process),
            _path_fit(centered, mask, q * eligible, scales, process),
        )
        states, q = _canonicalize(states, q)
        ll0 = _path_prn_loglik(centered, mask, states[0], scales)
        ll1 = _path_prn_loglik(centered, mask, states[1], scales)
        fit_ll = persistent_prn_mixture_loglik(
            ll0, ll1, np.clip(q, 1e-12, 1.0 - 1e-12), eligible
        )

    evaluation = mask[FIT_EPOCHS:].copy()
    n_valid = int(np.sum(evaluation))
    if not n_valid:
        raise ValueError("no heldout observations")

    k1_hold = 0.0
    hold0 = np.zeros(values.shape[1])
    hold1 = np.zeros(values.shape[1])
    observation0 = np.zeros_like(evaluation, dtype=float)
    observation1 = np.zeros_like(evaluation, dtype=float)
    k1_prediction = k1_states[-1].copy()
    path_prediction = [states[0][-1].copy(), states[1][-1].copy()]
    for relative_t in range(HOLDOUT_EPOCHS):
        k1_prediction = F @ k1_prediction
        path_prediction = [F @ path_prediction[0], F @ path_prediction[1]]
        absolute_t = FIT_EPOCHS + relative_t
        for prn in range(values.shape[1]):
            for modality in range(3):
                if not evaluation[relative_t, prn, modality]:
                    continue
                y = centered[absolute_t, prn, modality]
                k1_term = float(
                    student_logpdf(
                        y - H[modality] @ k1_prediction, scales[modality]
                    )
                )
                term0 = float(
                    student_logpdf(
                        y - H[modality] @ path_prediction[0], scales[modality]
                    )
                )
                term1 = float(
                    student_logpdf(
                        y - H[modality] @ path_prediction[1], scales[modality]
                    )
                )
                k1_hold += k1_term
                hold0[prn] += term0
                hold1[prn] += term1
                observation0[relative_t, prn, modality] = term0
                observation1[relative_t, prn, modality] = term1
    heldout_prns = eligible & np.any(evaluation, axis=(0, 2))
    k2_hold = persistent_prn_mixture_loglik(hold0, hold1, q, heldout_prns)
    observation_wise_hold = observation_wise_mixture_diagnostic(
        observation0, observation1, q, evaluation
    )

    raw_gain = k2_hold - k1_hold
    delta_p = 2 + int(np.sum(eligible))
    penalty = 0.5 * delta_p * math.log(n_valid)
    score = raw_gain - penalty
    observation_wise_score = observation_wise_hold - k1_hold - penalty
    digest_values = np.concatenate(
        [
            k1_states.ravel(),
            states[0].ravel(),
            states[1].ravel(),
            q[eligible],
            centering[eligible].ravel(),
        ]
    )
    digest = hashlib.sha256(
        np.asarray(digest_values, dtype="<f8").tobytes()
    ).hexdigest()
    masses = (
        float(np.sum(1.0 - q[eligible])),
        float(np.sum(q[eligible])),
    )
    return ScoreResult(
        score,
        score / n_valid,
        raw_gain,
        penalty,
        k1_hold,
        k2_hold,
        k1_fit,
        fit_ll,
        observation_wise_hold,
        observation_wise_score,
        q,
        eligible,
        evaluation,
        centering,
        n_valid,
        delta_p,
        masses,
        restart,
        digest,
    )


def persistence_statistic(scores: np.ndarray | list[float]) -> float:
    """Return max_j min(s_j,s_j+1,s_j+2) on one matched horizon."""

    values = np.asarray(scores, dtype=float)
    if len(values) < 3:
        return float("-inf")
    return float(max(np.min(values[index : index + 3]) for index in range(len(values) - 2)))


def first_persistent_alarm_index(
    scores: np.ndarray | list[float], threshold: float
) -> int | None:
    values = np.asarray(scores, dtype=float)
    for index in range(len(values) - 2):
        if np.all(values[index : index + 3] > threshold):
            return index + 2
    return None


def matched_horizon_statistics(
    a0_scores: np.ndarray | list[float],
    a6_scores: np.ndarray | list[float],
) -> dict[str, float | int]:
    """Apply exactly the same persistence statistic to A0 and A6."""

    a0 = np.asarray(a0_scores, dtype=float)
    a6 = np.asarray(a6_scores, dtype=float)
    if a0.shape != a6.shape:
        raise ValueError("A0/A6 horizon mismatch")
    return {
        "score_count": int(len(a0)),
        "A0_T": persistence_statistic(a0),
        "A6_T": persistence_statistic(a6),
    }


def localization_f1(result: ScoreResult, subset: np.ndarray) -> float:
    """Evaluate a latent two-path partition with label-swap invariance."""

    truth = set(map(int, subset))
    eligible = np.flatnonzero(result.eligible)
    if not truth or len(eligible) < len(truth):
        return 0.0
    order = eligible[np.argsort(result.memberships[eligible], kind="mergesort")]
    candidates = (set(map(int, order[: len(truth)])), set(map(int, order[-len(truth) :])))
    return max(2.0 * len(truth & candidate) / (len(truth) + len(candidate)) for candidate in candidates)


def localization_record(
    alarm_result: ScoreResult | None,
    oracle_result: ScoreResult | None,
    subset: np.ndarray,
) -> dict[str, float | bool]:
    """Keep missed primary localization at zero and oracle strictly diagnostic."""

    return {
        "detected": alarm_result is not None,
        "primary_f1": 0.0 if alarm_result is None else localization_f1(alarm_result, subset),
        "oracle_f1": 0.0 if oracle_result is None else localization_f1(oracle_result, subset),
        "oracle_used_for_gate": False,
    }

"""Q-COMET Stage-0: normal-only linear prediction and common-onset evidence."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np
from sklearn.covariance import LedoitWolf

from .q_comet_data import EpochData, TAP_OFFSETS_CHIPS


NO_SCORE = np.nan


def complex_to_real(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    return np.concatenate((values.real, values.imag), axis=-1).astype(np.float64, copy=False)


def real_to_complex(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.shape[-1] % 2:
        raise ValueError("real representation must have an even last dimension")
    half = values.shape[-1] // 2
    return values[..., :half] + 1j * values[..., half:]


def _group_order(data: EpochData) -> list[np.ndarray]:
    groups = []
    key = data.prn.astype(np.int64) * 100000 + data.segment.astype(np.int64)
    for item in np.unique(key):
        ix = np.flatnonzero(key == item)
        groups.append(ix[np.argsort(data.time_s[ix], kind="stable")])
    return groups


def supervised_rows(data: EpochData, *, lags: int, start_s: float, end_s: float):
    x, y, row_index = [], [], []
    values = complex_to_real(data.complex_taps)
    for ix in _group_order(data):
        for pos in range(lags, len(ix)):
            target = ix[pos]
            history = ix[pos-lags:pos]
            if not (start_s <= data.time_s[target] < end_s):
                continue
            if np.any(np.diff(data.epoch[np.r_[history, target]]) != 1):
                continue
            x.append(values[history].reshape(-1)); y.append(values[target]); row_index.append(target)
    return np.asarray(x), np.asarray(y), np.asarray(row_index, np.int64)


@dataclass
class LinearPredictor:
    kind: str
    lags: int
    coefficients: np.ndarray | None = None
    intercept: np.ndarray | None = None
    ridge: float = 0.0

    def predict_rows(self, data: EpochData, *, start_s: float = -np.inf, end_s: float = np.inf):
        x, y, ix = supervised_rows(data, lags=self.lags, start_s=start_s, end_s=end_s)
        if self.kind == "persistence":
            predicted = x[:, -18:]
        else:
            predicted = x @ self.coefficients + self.intercept
        return ix, y, predicted


def fit_predictor(data: EpochData, *, kind: str, lags: int, ridge: float,
                  train_range: tuple[float, float]) -> LinearPredictor:
    if kind == "persistence":
        return LinearPredictor(kind="persistence", lags=1)
    if kind != "ridge_var":
        raise ValueError("unknown predictor kind")
    x, y, _ = supervised_rows(data, lags=lags, start_s=train_range[0], end_s=train_range[1])
    if len(x) < 100:
        raise ValueError("insufficient clean normal training rows")
    mean_x, mean_y = x.mean(0), y.mean(0)
    xc, yc = x - mean_x, y - mean_y
    gram = xc.T @ xc + float(ridge) * np.eye(x.shape[1])
    coef = np.linalg.solve(gram, xc.T @ yc)
    return LinearPredictor(kind=kind, lags=lags, coefficients=coef,
                           intercept=mean_y - mean_x @ coef, ridge=float(ridge))


@dataclass(frozen=True)
class Whitener:
    covariance: np.ndarray
    inverse_sqrt: np.ndarray
    shrinkage: float
    eigenvalue_floor: float

    def transform(self, residual: np.ndarray) -> np.ndarray:
        return np.asarray(residual) @ self.inverse_sqrt.T


def fit_whitener(residual: np.ndarray, *, eigenvalue_floor_ratio: float = 1e-4) -> Whitener:
    residual = np.asarray(residual, np.float64)
    fit = LedoitWolf(assume_centered=False).fit(residual)
    cov = np.asarray(fit.covariance_)
    eigenvalue, vectors = np.linalg.eigh(cov)
    floor = max(float(np.median(eigenvalue)) * eigenvalue_floor_ratio, np.finfo(float).eps)
    eigenvalue = np.maximum(eigenvalue, floor)
    regular = (vectors * eigenvalue) @ vectors.T
    inverse = (vectors * (1.0 / np.sqrt(eigenvalue))) @ vectors.T
    return Whitener(regular, inverse, float(fit.shrinkage_), floor)


def nuisance_jacobian(reference_real: np.ndarray, inverse_sqrt: np.ndarray,
                      *, tap_indices: np.ndarray | None = None) -> np.ndarray:
    """Observed-peak tangents for amplitude, carrier phase, delay, and Doppler."""
    reference = real_to_complex(np.asarray(reference_real))
    taps = TAP_OFFSETS_CHIPS if tap_indices is None else TAP_OFFSETS_CHIPS[np.asarray(tap_indices)]
    derivative = np.gradient(reference, taps)
    tangents = np.stack((reference, 1j * reference, derivative, 1j * taps * reference), axis=0)
    return complex_to_real(tangents) @ inverse_sqrt.T


def quotient_project(whitened: np.ndarray, reference_real: np.ndarray,
                     inverse_sqrt: np.ndarray, *, tap_indices: np.ndarray | None = None) -> np.ndarray:
    j = nuisance_jacobian(reference_real, inverse_sqrt, tap_indices=tap_indices).T
    q, _ = np.linalg.qr(j, mode="reduced")
    return whitened - q @ (q.T @ whitened)


@dataclass
class InnovationTable:
    row_index: np.ndarray
    time_s: np.ndarray
    epoch: np.ndarray
    prn: np.ndarray
    quotient: np.ndarray
    whitened: np.ndarray
    raw_residual: np.ndarray
    prompt_power: np.ndarray


def innovations(data: EpochData, predictor: LinearPredictor, whitener: Whitener,
                *, start_s: float = -np.inf, end_s: float = np.inf,
                tap_indices: Iterable[int] | None = None, quotient: bool = True) -> InnovationTable:
    ix, observed, predicted = predictor.predict_rows(data, start_s=start_s, end_s=end_s)
    selected = None if tap_indices is None else np.asarray(tuple(tap_indices), int)
    if selected is not None:
        observed_c = real_to_complex(observed)[:, selected]
        predicted_c = real_to_complex(predicted)[:, selected]
        observed = complex_to_real(observed_c); predicted = complex_to_real(predicted_c)
        # A sub-view uses its own principal submatrix whitener.
        real_ix = np.r_[selected, selected + 9]
        covariance = whitener.covariance[np.ix_(real_ix, real_ix)]
        ev, vec = np.linalg.eigh(covariance); ev = np.maximum(ev, whitener.eigenvalue_floor)
        inverse_sqrt = (vec * (1 / np.sqrt(ev))) @ vec.T
    else:
        inverse_sqrt = whitener.inverse_sqrt
    raw = observed - predicted
    white = raw @ inverse_sqrt.T
    projected = np.empty_like(white)
    for pos in range(len(ix)):
        projected[pos] = quotient_project(white[pos], predicted[pos], inverse_sqrt,
                                          tap_indices=selected) if quotient else white[pos]
    return InnovationTable(ix, data.time_s[ix], data.epoch[ix], data.prn[ix], projected,
                           white, raw, data.prompt_power[ix])


def predictor_validation_nll(data: EpochData, predictor: LinearPredictor,
                             validation_range: tuple[float, float]) -> float:
    _, observed, predicted = predictor.predict_rows(data, start_s=validation_range[0], end_s=validation_range[1])
    residual = observed - predicted
    covariance = fit_whitener(residual).covariance
    sign, logdet = np.linalg.slogdet(covariance)
    if sign <= 0:
        return math.inf
    solve = np.linalg.solve(covariance, residual.T).T
    return float(np.mean(np.sum(residual * solve, axis=1) + logdet))


def _basis(length: int) -> np.ndarray:
    x = np.arange(length, dtype=float)
    ramp = x / max(length - 1, 1)
    transient = 1.0 - np.exp(-x / 5.0)
    return np.column_stack((np.ones(length), ramp, transient))


def analytic_log_bf(sequence: np.ndarray, *, prior_variance: float = 0.25) -> float:
    """Gaussian linear-model marginal BF with free signed vector coefficients."""
    y = np.asarray(sequence, float)
    if y.ndim != 2 or len(y) == 0:
        return 0.0
    b = _basis(len(y)); gram = b.T @ b
    precision = np.eye(3) / prior_variance + gram
    sign, logdet = np.linalg.slogdet(np.eye(3) + prior_variance * gram)
    if sign <= 0:
        return -np.inf
    cross = b.T @ y
    gain = float(np.sum(cross * np.linalg.solve(precision, cross)))
    return 0.5 * gain - 0.5 * y.shape[1] * logdet


def _mixture_log(log_bf: float, participation: float) -> float:
    a = math.log1p(-participation)
    b = math.log(participation) + min(float(log_bf), 700.0)
    return float(np.logaddexp(a, b))


def score_common_onset(table: InnovationTable, *, memory_epochs: int,
                       participation: float, prior_variance: float,
                       min_prns: int = 4, penalty: float | None = None,
                       values: np.ndarray | None = None) -> list[dict]:
    values = table.quotient if values is None else np.asarray(values)
    epoch_values = sorted(map(int, np.unique(table.epoch)))
    time_by_epoch = {int(e): float(np.max(table.time_s[table.epoch == e])) for e in epoch_values}
    lookup = {(int(e), int(p)): values[i] for i, (e, p) in enumerate(zip(table.epoch, table.prn))}
    rows = []
    for epoch in epoch_values:
        support = sorted({int(p) for e, p in zip(table.epoch, table.prn) if int(e) == epoch})
        if len(support) < min_prns:
            rows.append({"epoch": epoch, "time_s": time_by_epoch[epoch], "score": NO_SCORE,
                         "estimated_onset_epoch": None, "tracked_prns": len(support), "participation": NO_SCORE})
            continue
        best_score, best_k, best_bf = -np.inf, None, None
        start_min = max(epoch_values[0], epoch - memory_epochs + 1)
        for onset in range(start_min, epoch + 1):
            terms, bfs = [], []
            for prn in support:
                sequence = [lookup[(e, prn)] for e in range(onset, epoch + 1) if (e, prn) in lookup]
                if len(sequence) != epoch - onset + 1:
                    continue
                bf = analytic_log_bf(np.asarray(sequence), prior_variance=prior_variance)
                terms.append(_mixture_log(bf, participation)); bfs.append(bf)
            if len(terms) < min_prns:
                continue
            trial = sum(terms) - (penalty if penalty is not None else 0.5 * math.log(memory_epochs + 1))
            if trial > best_score:
                best_score, best_k, best_bf = trial, onset, bfs
        if best_k is None:
            rows.append({"epoch": epoch, "time_s": time_by_epoch[epoch], "score": NO_SCORE,
                         "estimated_onset_epoch": None, "tracked_prns": len(support), "participation": NO_SCORE})
        else:
            posterior = [1 / (1 + math.exp(np.clip(math.log((1-participation)/participation) - bf, -700, 700))) for bf in best_bf]
            rows.append({"epoch": epoch, "time_s": time_by_epoch[epoch], "score": float(max(0.0, best_score)),
                         "estimated_onset_epoch": int(best_k), "tracked_prns": len(support),
                         "participation": float(np.mean(posterior))})
    return rows


def score_independent_changepoints(table: InnovationTable, *, memory_epochs: int,
                                   participation: float, prior_variance: float,
                                   min_prns: int = 4) -> list[dict]:
    """A3: each PRN maximizes over its own onset before PRN evidence is combined."""
    values = table.quotient; lookup = {(int(e), int(p)): values[i] for i, (e, p) in enumerate(zip(table.epoch, table.prn))}
    time_by_epoch = {int(e): float(np.max(table.time_s[table.epoch == e])) for e in np.unique(table.epoch)}
    rows = []
    for epoch in sorted(map(int, np.unique(table.epoch))):
        support = sorted({int(p) for e, p in zip(table.epoch, table.prn) if int(e) == epoch})
        if len(support) < min_prns:
            rows.append({"epoch": epoch, "time_s": time_by_epoch[epoch], "score": NO_SCORE,
                         "estimated_onset_epoch": None, "tracked_prns": len(support), "participation": NO_SCORE}); continue
        best = []
        for prn in support:
            candidates = []
            for onset in range(max(min(lookup)[0], epoch-memory_epochs+1), epoch+1):
                seq = [lookup[(e, prn)] for e in range(onset, epoch+1) if (e, prn) in lookup]
                if len(seq) == epoch-onset+1:
                    candidates.append(analytic_log_bf(np.asarray(seq), prior_variance=prior_variance))
            if candidates: best.append(max(candidates))
        score = sum(_mixture_log(bf, participation) for bf in best) - 0.5 * len(best) * math.log(memory_epochs+1)
        rows.append({"epoch": epoch, "time_s": time_by_epoch[epoch], "score": float(max(0, score)),
                     "estimated_onset_epoch": None, "tracked_prns": len(support),
                     "participation": float(np.mean([bf > 0 for bf in best])) if best else NO_SCORE})
    return rows


def rank1_values(values: np.ndarray, prns: np.ndarray, epochs: np.ndarray) -> np.ndarray:
    """A5 shared-direction restriction: project each epoch onto its leading PRN direction."""
    out = np.zeros_like(values)
    for epoch in np.unique(epochs):
        ix = np.flatnonzero(epochs == epoch)
        if len(ix) < 2: continue
        _, _, vh = np.linalg.svd(values[ix], full_matrices=False)
        direction = vh[0]
        out[ix] = np.outer(values[ix] @ direction, direction)
    return out


def empirical_threshold(values: Iterable[float], quantile: float) -> float:
    finite = np.sort(np.asarray([v for v in values if np.isfinite(v)], float))
    if finite.size == 0:
        raise ValueError("no finite calibration scores")
    return float(finite[int(np.ceil(quantile * len(finite))) - 1])

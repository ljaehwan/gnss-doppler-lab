"""Analytic CORA residual tokens and rank-1 shared-emitter likelihood."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .cora_cross_cumulant import cross_cumulant_matrix
from .mosaic_raw_recorrelation import receiver_carrier_wipeoff, receiver_code_replicas


DELAY_GRID_CHIPS = np.asarray((-0.25, 0.0, 0.25), dtype=np.float64)
DOPPLER_GRID_HZ = np.asarray((-25.0, 0.0, 25.0), dtype=np.float64)
RIDGE = 1e-3
SHRINKAGE = 0.2
VARIANCE_FLOOR = 1e-8


@dataclass(frozen=True)
class SharedLikelihood:
    score: float
    raw_improvement: float
    complexity_penalty: float
    rank1_strength: float
    participating_prns: int
    loadings: np.ndarray


def nuisance_tangent_projector() -> np.ndarray:
    delay, doppler = np.meshgrid(DELAY_GRID_CHIPS, DOPPLER_GRID_HZ, indexing="ij")
    tangent = np.column_stack((np.ones(9), delay.reshape(-1), doppler.reshape(-1)))
    q, _ = np.linalg.qr(tangent)
    return np.eye(9) - q @ q.T


TANGENT_PROJECTOR = nuisance_tangent_projector()


def raw_residual_token(iq: np.ndarray, *, prn: int, action: np.void, sample_rate_hz: float) -> tuple[np.ndarray, dict[str, float]]:
    """Remove one nominal PRN replica and correlate a frozen 3x3 local grid."""
    samples = np.asarray(iq, dtype=np.complex64).reshape(-1); count = len(samples)
    code0 = receiver_code_replicas(prn, count,
        float(action["action_used_code_phase_step_chips_per_sample"]),
        float(action["action_used_residual_code_phase_chips"]), (0.0,))[0]
    wipe = receiver_carrier_wipeoff(count,
        float(action["action_used_residual_carrier_phase_rad"]),
        float(action["action_used_carrier_phase_step_rad_per_sample"]))
    raw_replica = code0 * np.conj(wipe)
    coefficient = np.vdot(raw_replica, samples) / max(float(np.vdot(raw_replica, raw_replica).real), 1e-12)
    residual = samples - coefficient * raw_replica
    delays = [float(v) for v in DELAY_GRID_CHIPS for _ in DOPPLER_GRID_HZ]
    replicas = receiver_code_replicas(prn, count,
        float(action["action_used_code_phase_step_chips_per_sample"]),
        float(action["action_used_residual_code_phase_chips"]), delays)
    n = np.arange(count, dtype=np.float32); values = []
    for index, delta_hz in enumerate(np.tile(DOPPLER_GRID_HZ, len(DELAY_GRID_CHIPS))):
        extra = np.exp(-2j * np.pi * float(delta_hz) * n / float(sample_rate_hz))
        values.append(np.sum(replicas[index] * residual * wipe * extra, dtype=np.complex64))
    token = TANGENT_PROJECTOR @ np.asarray(values, dtype=np.complex128)
    norm = float(np.linalg.norm(token)); token = token / max(norm, VARIANCE_FLOOR)
    return token, {"raw_rms": float(np.sqrt(np.mean(np.abs(samples) ** 2))),
                   "residual_rms": float(np.sqrt(np.mean(np.abs(residual) ** 2))),
                   "ls_coefficient_abs": float(abs(coefficient)), "token_prequotient_norm": norm}


def fit_shared_conditioner(tokens: np.ndarray, context: np.ndarray) -> dict[str, np.ndarray]:
    """Fit one PRN-shared clean-only complex ridge conditioner."""
    z = np.asarray(tokens, dtype=np.complex128); c = np.asarray(context, dtype=np.float64)
    if z.ndim != 3 or c.shape[:2] != z.shape[:2]: raise ValueError("token/context shape mismatch")
    design = np.column_stack((np.ones(c.shape[0] * c.shape[1]), c.reshape(-1, c.shape[2])))
    target = z.reshape(-1, z.shape[2]); gram = design.T @ design + RIDGE * np.eye(design.shape[1])
    beta = np.linalg.solve(gram, design.T @ target)
    innovation = target - design @ beta
    covariance = innovation.conj().T @ innovation / max(len(innovation) - 1, 1)
    diagonal = np.diag(np.diag(covariance)); covariance = (1 - SHRINKAGE) * covariance + SHRINKAGE * diagonal
    values, vectors = np.linalg.eigh(covariance); floor = max(float(np.max(values)) * 1e-6, VARIANCE_FLOOR)
    whitener = vectors @ np.diag(1.0 / np.sqrt(np.maximum(values, floor))) @ vectors.conj().T
    return {"beta": beta, "whitener": whitener, "covariance": covariance}


def condition_tokens(tokens: np.ndarray, context: np.ndarray, model: dict[str, np.ndarray]) -> np.ndarray:
    z = np.asarray(tokens, dtype=np.complex128); c = np.asarray(context, dtype=np.float64)
    design = np.column_stack((np.ones(c.shape[0] * c.shape[1]), c.reshape(-1, c.shape[2])))
    innovation = z.reshape(-1, z.shape[2]) - design @ model["beta"]
    return (innovation @ model["whitener"]).reshape(z.shape)


def shared_emitter_likelihood(matrix: np.ndarray, *, null_variance: float, bic: bool = True) -> SharedLikelihood:
    k = np.asarray(matrix, dtype=np.float64); n = len(k)
    if k.shape != (n, n) or n < 4 or not np.allclose(k, k.T): raise ValueError("symmetric >=4 PRN matrix required")
    off = ~np.eye(n, dtype=bool); sigma2 = max(float(null_variance), VARIANCE_FLOOR)
    h0 = float(np.sum(k[off] ** 2) / sigma2)
    values, vectors = np.linalg.eigh(k); top = max(float(values[-1]), 0.0); loading = vectors[:, -1]
    rank1 = top * np.outer(loading, loading); np.fill_diagonal(rank1, 0.0)
    h1 = float(np.sum((k[off] - rank1[off]) ** 2) / sigma2)
    improvement = max(h0 - h1, 0.0); observations = n * (n - 1) // 2
    penalty = float(n * np.log(max(observations, 2))) if bic else 0.0
    score = improvement - penalty
    participation = int(np.sum(np.abs(loading) >= 0.5 * np.max(np.abs(loading))))
    return SharedLikelihood(score=float(score), raw_improvement=improvement, complexity_penalty=penalty,
                            rank1_strength=top, participating_prns=participation, loadings=loading)


def score_token_block(tokens: np.ndarray, *, null_variance: float) -> tuple[SharedLikelihood, np.ndarray]:
    matrix = cross_cumulant_matrix(tokens, variance_floor=VARIANCE_FLOOR)
    return shared_emitter_likelihood(matrix, null_variance=null_variance), matrix


def temporal_desynchronize(tokens: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    z = np.asarray(tokens).copy()
    for index, shift in enumerate(np.asarray(offsets, dtype=int)): z[:, index] = np.roll(z[:, index], shift, axis=0)
    return z


def phase_surrogate(tokens: np.ndarray, seed: int) -> np.ndarray:
    z = np.asarray(tokens, dtype=np.complex128); rng = np.random.default_rng(seed); out = np.empty_like(z)
    for prn in range(z.shape[1]):
        for projection in range(z.shape[2]):
            spectrum = np.fft.fft(z[:, prn, projection]); phases = rng.uniform(0, 2*np.pi, len(spectrum))
            out[:, prn, projection] = np.fft.ifft(np.abs(spectrum) * np.exp(1j * phases))
            order = np.argsort(np.abs(out[:, prn, projection])); target = np.sort(np.abs(z[:, prn, projection]))
            magnitudes = np.empty_like(target); magnitudes[order] = target
            out[:, prn, projection] = magnitudes * np.exp(1j * np.angle(out[:, prn, projection]))
    return out

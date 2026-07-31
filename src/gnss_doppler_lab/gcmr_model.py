"""Permutation-safe geometry-conditioned GCMR-Net and clean scoring.

The network deliberately has no satellite identifier input.  Every real pair is
processed by the same encoder and decoder, while event pooling is symmetric.
Scaler and score-calibration fitting must be performed on train/clean-reference
sets by the caller; fitted statistics are serialised in model state dictionaries.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
from torch import nn

OBSERVATION_DIM = 10
CONDITION_DIM = 8


def _tensor(value, *, dtype=None, name="value"):
    result = value if torch.is_tensor(value) else torch.as_tensor(value)
    if dtype is not None:
        result = result.to(dtype=dtype)
    if result.device.type == "meta":
        raise ValueError(f"{name} cannot be a meta tensor")
    return result


def _validate_batch(observations, observation_mask, conditions, pair_mask):
    observations = _tensor(observations, dtype=torch.float32, name="observations")
    observation_mask = _tensor(observation_mask, dtype=torch.bool, name="observation_mask")
    conditions = _tensor(conditions, dtype=torch.float32, name="conditions")
    pair_mask = _tensor(pair_mask, dtype=torch.bool, name="pair_mask")
    if observations.ndim != 3 or observations.shape[-1] != OBSERVATION_DIM:
        raise ValueError("observations must have shape [B, P, 10]")
    b, p, _ = observations.shape
    if observation_mask.shape != observations.shape:
        raise ValueError("observation_mask must match observations")
    if conditions.shape != (b, p, CONDITION_DIM):
        raise ValueError("conditions must have shape [B, P, 8]")
    if pair_mask.shape != (b, p):
        raise ValueError("pair_mask must have shape [B, P]")
    if b == 0 or p == 0 or not pair_mask.any(dim=1).all():
        raise ValueError("every event must contain at least one real pair")
    valid_observation = observation_mask & pair_mask.unsqueeze(-1)
    if not valid_observation.any(dim=(1, 2)).all():
        raise ValueError("every event must contain at least one observed channel")
    if not torch.isfinite(observations[valid_observation]).all():
        raise ValueError("valid observations must be finite")
    if not torch.isfinite(conditions[pair_mask]).all():
        raise ValueError("real-pair conditions must be finite")
    return observations, observation_mask, conditions, pair_mask


def collate_gcmr_events(events: Iterable, *, device=None, dtype=torch.float32):
    """Validate and pad variable-size ``GcmrPairRelationEvent`` objects.

    Padding is zero-valued and explicitly identified by ``pair_mask``.  Masked
    observation payloads may be nonfinite (relation construction uses NaN), but
    they are replaced with zero so they can never leak into network arithmetic.
    """
    events = list(events)
    if not events:
        raise ValueError("at least one event is required")
    sizes = []
    for index, event in enumerate(events):
        try:
            obs = np.asarray(event.observations)
            mask = np.asarray(event.observation_mask)
            cond = np.asarray(event.conditions)
            pairs = np.asarray(event.pair_prns)
        except AttributeError as exc:
            raise ValueError("events must be GCMR pair-relation events") from exc
        p = obs.shape[0] if obs.ndim == 2 else -1
        if p <= 0:
            raise ValueError(f"event {index} must contain at least one pair")
        if obs.shape != (p, OBSERVATION_DIM) or mask.shape != obs.shape or cond.shape != (p, CONDITION_DIM) or pairs.shape != (p, 2):
            raise ValueError(f"event {index} has invalid pair array shapes")
        if mask.dtype != np.bool_:
            raise ValueError("observation masks must be boolean")
        if not mask.any():
            raise ValueError(f"event {index} has no observed channels")
        if not np.isfinite(obs[mask]).all() or not np.isfinite(cond).all():
            raise ValueError(f"event {index} has nonfinite valid values")
        sizes.append(p)
    b, pmax = len(events), max(sizes)
    observations = torch.zeros((b, pmax, OBSERVATION_DIM), dtype=dtype, device=device)
    observation_mask = torch.zeros((b, pmax, OBSERVATION_DIM), dtype=torch.bool, device=device)
    conditions = torch.zeros((b, pmax, CONDITION_DIM), dtype=dtype, device=device)
    pair_mask = torch.zeros((b, pmax), dtype=torch.bool, device=device)
    for row, (event, p) in enumerate(zip(events, sizes)):
        mask = torch.as_tensor(np.asarray(event.observation_mask), dtype=torch.bool, device=device)
        raw = torch.as_tensor(np.asarray(event.observations), dtype=dtype, device=device)
        observations[row, :p] = torch.where(mask, raw, torch.zeros_like(raw))
        observation_mask[row, :p] = mask
        conditions[row, :p] = torch.as_tensor(np.asarray(event.conditions), dtype=dtype, device=device)
        pair_mask[row, :p] = True
    return {"observations": observations, "observation_mask": observation_mask,
            "conditions": conditions, "pair_mask": pair_mask}


class RobustGcmrScaler(nn.Module):
    """Per-channel train-set median/MAD scaler with mask-aware fitting."""

    def __init__(self, *, minimum_scale=1e-6):
        super().__init__()
        if not math.isfinite(minimum_scale) or minimum_scale <= 0:
            raise ValueError("minimum_scale must be positive and finite")
        self.minimum_scale = float(minimum_scale)
        self.register_buffer("observation_center", torch.zeros(OBSERVATION_DIM))
        self.register_buffer("observation_scale", torch.ones(OBSERVATION_DIM))
        self.register_buffer("condition_center", torch.zeros(CONDITION_DIM))
        self.register_buffer("condition_scale", torch.ones(CONDITION_DIM))
        self.register_buffer("fitted", torch.tensor(False))

    def _statistics(self, values, masks, name):
        centers, scales = [], []
        for channel in range(values.shape[-1]):
            supported = values[..., channel][masks[..., channel]]
            if supported.numel() == 0:
                raise ValueError(f"{name} channel {channel} has no support")
            if not torch.isfinite(supported).all():
                raise ValueError(f"{name} support must be finite")
            center = supported.median()
            scale = (supported - center).abs().median() * 1.4826
            if not torch.isfinite(center) or not torch.isfinite(scale):
                raise ValueError(f"{name} statistics must be finite")
            scale = scale.clamp_min(self.minimum_scale)
            centers.append(center); scales.append(scale)
        return torch.stack(centers), torch.stack(scales)

    @torch.no_grad()
    def fit(self, observations, observation_mask, conditions, pair_mask):
        observations, observation_mask, conditions, pair_mask = _validate_batch(
            observations, observation_mask, conditions, pair_mask)
        valid = observation_mask & pair_mask.unsqueeze(-1)
        condition_valid = pair_mask.unsqueeze(-1).expand_as(conditions)
        oc, os = self._statistics(observations, valid, "observation")
        cc, cs = self._statistics(conditions, condition_valid, "condition")
        self.observation_center.copy_(oc.to(self.observation_center))
        self.observation_scale.copy_(os.to(self.observation_scale))
        self.condition_center.copy_(cc.to(self.condition_center))
        self.condition_scale.copy_(cs.to(self.condition_scale))
        self.fitted.fill_(True)
        return self

    def _require_fitted(self):
        if not bool(self.fitted.item()):
            raise RuntimeError("robust scaler has not been fitted")

    def transform_observations(self, observations, observation_mask=None):
        self._require_fitted()
        result = (observations - self.observation_center) / self.observation_scale
        return torch.where(observation_mask, result, torch.zeros_like(result)) if observation_mask is not None else result

    def transform_conditions(self, conditions):
        self._require_fitted()
        return (conditions - self.condition_center) / self.condition_scale

    def inverse_observations(self, standardized):
        self._require_fitted()
        return standardized * self.observation_scale + self.observation_center


class GcmrNet(nn.Module):
    """Shared-pair encoder, symmetric event pooling, and geometry-only decoder."""

    def __init__(self, *, pair_hidden=32, event_hidden=64, latent_dim=32, scaler=None):
        super().__init__()
        if min(pair_hidden, event_hidden, latent_dim) <= 0:
            raise ValueError("hidden and latent dimensions must be positive")
        self.scaler = scaler if scaler is not None else RobustGcmrScaler()
        self.pair_encoder = nn.Sequential(
            nn.Linear(OBSERVATION_DIM * 2 + CONDITION_DIM, pair_hidden),
            nn.LayerNorm(pair_hidden), nn.SiLU(),
            nn.Linear(pair_hidden, event_hidden), nn.SiLU())
        self.event_encoder = nn.Sequential(
            nn.Linear(event_hidden * 2, event_hidden), nn.LayerNorm(event_hidden),
            nn.SiLU(), nn.Linear(event_hidden, latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + CONDITION_DIM, event_hidden),
            nn.LayerNorm(event_hidden), nn.SiLU(),
            nn.Linear(event_hidden, pair_hidden), nn.SiLU(),
            nn.Linear(pair_hidden, OBSERVATION_DIM))

    def fit_scaler(self, observations, observation_mask, conditions, pair_mask):
        self.scaler.fit(observations, observation_mask, conditions, pair_mask)
        return self

    @staticmethod
    def _pool(encoded, pair_mask):
        # Gathering real rows makes both reduction order and result independent
        # of padding. Featurewise sorting makes that order canonical under pair
        # permutations (and remains differentiable almost everywhere).
        pooled = []
        for rows, valid in zip(encoded, pair_mask):
            real = rows[valid]
            ordered = torch.sort(real, dim=0).values
            pooled.append(torch.cat((ordered.sum(dim=0) / real.shape[0], ordered[-1]), dim=0))
        return torch.stack(pooled)

    def encode(self, observations, observation_mask, conditions, pair_mask):
        standardized_obs = self.scaler.transform_observations(observations, observation_mask)
        standardized_cond = self.scaler.transform_conditions(conditions)
        inputs = torch.cat((standardized_obs, observation_mask.to(observations.dtype), standardized_cond), dim=-1)
        encoded = self.pair_encoder(inputs)
        return self.event_encoder(self._pool(encoded, pair_mask)), standardized_cond

    def decode(self, latent, standardized_conditions):
        broadcast = latent.unsqueeze(1).expand(-1, standardized_conditions.shape[1], -1)
        standardized_reconstruction = self.decoder(torch.cat((broadcast, standardized_conditions), dim=-1))
        return self.scaler.inverse_observations(standardized_reconstruction)

    def forward(self, observations, observation_mask, conditions, pair_mask):
        observations, observation_mask, conditions, pair_mask = _validate_batch(
            observations, observation_mask, conditions, pair_mask)
        latent, standardized_conditions = self.encode(observations, observation_mask, conditions, pair_mask)
        return self.decode(latent, standardized_conditions), latent


def masked_reconstruction_mse(reconstruction, target, observation_mask, pair_mask, *, observation_scale):
    """Mean squared error over valid relation channels only."""
    if reconstruction.shape != target.shape or observation_mask.shape != target.shape:
        raise ValueError("reconstruction, target, and observation_mask shapes must match")
    if pair_mask.shape != target.shape[:2]:
        raise ValueError("pair_mask shape must match the first two target dimensions")
    valid = observation_mask.bool() & pair_mask.bool().unsqueeze(-1)
    if not valid.any():
        raise ValueError("masked reconstruction loss has no valid targets")
    scale = torch.as_tensor(observation_scale, dtype=target.dtype, device=target.device)
    if scale.shape != (OBSERVATION_DIM,) or not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("observation_scale must contain 10 positive finite fitted scales")
    difference = ((reconstruction - target) / scale)[valid]
    if not torch.isfinite(difference).all():
        raise ValueError("valid reconstruction values and targets must be finite")
    return difference.square().mean()


def fixed_center_compactness(latent, center=None):
    """Event-level compactness around a caller-provided, non-learnable center."""
    if latent.ndim != 2 or latent.shape[0] == 0:
        raise ValueError("latent must have shape [B, D] with B > 0")
    center = torch.zeros(latent.shape[-1], device=latent.device, dtype=latent.dtype) if center is None else torch.as_tensor(center, device=latent.device, dtype=latent.dtype)
    if center.shape != (latent.shape[-1],) or center.requires_grad or not torch.isfinite(center).all():
        raise ValueError("compactness center must be a finite fixed vector")
    return (latent - center).square().sum(dim=-1).mean()


def gcmr_loss(reconstruction, target, observation_mask, pair_mask, *, observation_scale, latent=None,
              compactness_weight=0.0, compactness_center=None):
    loss = masked_reconstruction_mse(reconstruction, target, observation_mask, pair_mask, observation_scale=observation_scale)
    if not math.isfinite(compactness_weight) or compactness_weight < 0:
        raise ValueError("compactness_weight must be finite and nonnegative")
    if compactness_weight:
        if latent is None:
            raise ValueError("latent is required for compactness")
        loss = loss + compactness_weight * fixed_center_compactness(latent, compactness_center)
    return loss


def event_reconstruction_errors(reconstruction, target, observation_mask, pair_mask, *, observation_scale):
    """Return one valid-channel reconstruction MSE per event."""
    valid = observation_mask.bool() & pair_mask.bool().unsqueeze(-1)
    scale = torch.as_tensor(observation_scale, dtype=target.dtype, device=target.device)
    if scale.shape != (OBSERVATION_DIM,) or not torch.isfinite(scale).all() or not (scale > 0).all():
        raise ValueError("observation_scale must contain 10 positive finite fitted scales")
    errors = []
    for row in range(target.shape[0]):
        if not valid[row].any():
            raise ValueError("every event needs a valid reconstruction target")
        difference = ((reconstruction[row] - target[row]) / scale)[valid[row]]
        if not torch.isfinite(difference).all():
            raise ValueError("valid standardized reconstruction residuals must be finite")
        errors.append(difference.square().mean())
    return torch.stack(errors)


class CleanReferenceScoreCalibrator:
    """Clean-only robust calibration of reconstruction and latent anomaly terms."""

    def __init__(self, *, shrinkage=0.1, minimum_scale=1e-9):
        if not math.isfinite(shrinkage) or not 0 < shrinkage <= 1:
            raise ValueError("shrinkage must be in (0, 1]")
        if not math.isfinite(minimum_scale) or minimum_scale <= 0:
            raise ValueError("minimum_scale must be positive and finite")
        self.shrinkage = float(shrinkage); self.minimum_scale = float(minimum_scale)
        self.fitted_ = False

    def _arrays(self, reconstruction_errors, latent):
        error = np.asarray(reconstruction_errors, dtype=np.float64)
        z = np.asarray(latent, dtype=np.float64)
        if error.ndim != 1 or z.ndim != 2 or len(error) != len(z) or len(error) == 0:
            raise ValueError("errors [N] and latent [N, D] must have matching nonempty rows")
        if z.shape[1] == 0 or not np.isfinite(error).all() or not np.isfinite(z).all():
            raise ValueError("score inputs must be finite and latent dimension nonempty")
        return error, z

    def _distance(self, latent):
        delta = latent - self.latent_center_
        squared = np.einsum("ni,ij,nj->n", delta, self.precision_, delta)
        return np.sqrt(np.maximum(squared, 0.0))

    def _robust(self, values):
        center = float(np.median(values))
        scale = float(1.4826 * np.median(np.abs(values - center)))
        return center, max(scale, self.minimum_scale)

    def fit(self, reconstruction_errors, latent):
        error, z = self._arrays(reconstruction_errors, latent)
        if len(error) < 2:
            raise ValueError("at least two clean reference events are required")
        self.latent_center_ = np.median(z, axis=0)
        delta = z - self.latent_center_
        covariance = delta.T @ delta / len(z)
        d = z.shape[1]
        average_variance = float(np.trace(covariance) / d)
        floor = max(average_variance * self.minimum_scale, self.minimum_scale)
        covariance = ((1.0 - self.shrinkage) * covariance
                      + self.shrinkage * max(average_variance, floor) * np.eye(d))
        self.precision_ = np.linalg.inv(covariance)
        self.reconstruction_center_, self.reconstruction_scale_ = self._robust(error)
        distances = self._distance(z)
        self.latent_distance_center_, self.latent_distance_scale_ = self._robust(distances)
        self.fitted_ = True
        return self

    def components(self, reconstruction_errors, latent):
        if not self.fitted_:
            raise RuntimeError("clean-reference calibrator has not been fitted")
        error, z = self._arrays(reconstruction_errors, latent)
        if z.shape[1] != self.latent_center_.shape[0]:
            raise ValueError("latent dimension differs from fitted reference")
        reconstruction_z = np.maximum(0.0, (error - self.reconstruction_center_) / self.reconstruction_scale_)
        distance_z = np.maximum(0.0, (self._distance(z) - self.latent_distance_center_) / self.latent_distance_scale_)
        return reconstruction_z, distance_z

    def score(self, reconstruction_errors, latent):
        reconstruction_z, distance_z = self.components(reconstruction_errors, latent)
        result = (reconstruction_z + distance_z) / 2.0
        if not np.isfinite(result).all():
            raise ValueError("calibrated scores are nonfinite")
        return result

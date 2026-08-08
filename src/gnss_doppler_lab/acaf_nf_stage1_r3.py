"""ACAF-NF Stage-1 R3 normal-only neural-field primitives.

The production runner owns recording discovery and enforces the clean/attack
checkpoint.  This module contains deterministic, label-blind mathematics used
by both the runner and the independent verifier.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

FS_HZ = 25_000_000.0
SUPPORT_SAMPLES = 25_000
L20 = 20
DELAYS = np.arange(-1.0, 1.0001, 0.125, dtype=np.float64)
DOPPLERS = np.arange(-250.0, 250.0001, 50.0, dtype=np.float64)
GRID = np.asarray([(d / 250.0, c) for d in DOPPLERS for c in DELAYS], dtype=np.float32)
CENTER = int(np.ravel_multi_index((5, 8), (len(DOPPLERS), len(DELAYS))))
BUDGETS = (3, 5, 9, 16, 187)
CLEAN_ROLES = {
    "train": (10.0, 45.0),
    "selection": (47.0, 62.0),
    "calibration": (64.0, 82.0),
    "holdout": (84.0, 100.0),
}
ATTACK_PHASES = {
    "ds3": {"strict_pre": (100.0, 118.9), "injection_takeover": (118.9, 195.0), "established_pull_off": (195.0, math.inf)},
    "ds4": {"strict_pre": (100.0, 113.8), "injection_takeover": (113.8, math.inf)},
    "ds7": {"strict_pre": (100.0, 110.0), "injection_takeover": (110.0, 150.0), "established_pull_off": (150.0, math.inf)},
    "ds8": {"strict_pre": (100.0, 110.0), "injection_takeover": (110.0, 150.0), "established_pull_off": (150.0, math.inf)},
}
PRIMARY_FAMILY = {"ds3": "ds3", "ds4": "ds4", "ds7": "ds7_ds8", "ds8": "ds7_ds8"}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def role_for_support(start_sample: int, end_sample: int) -> str | None:
    """Return a clean role only when the causal 20-ms support is contained."""
    start = start_sample / FS_HZ
    end = end_sample / FS_HZ
    for role, (left, right) in CLEAN_ROLES.items():
        if left <= start and end <= right:
            return role
    return None


def attack_phase(scenario: str, start_sample: int, end_sample: int) -> str | None:
    start = start_sample / FS_HZ
    end = end_sample / FS_HZ
    for name, (left, right) in ATTACK_PHASES[scenario].items():
        if left <= start and end <= right:
            return name
    return None


def assert_no_byte_overlap(rows_by_role: Mapping[str, Sequence[Mapping[str, object]]]) -> None:
    """Reject byte overlap within one authenticated recording across roles."""
    roles = sorted(rows_by_role)
    for i, left in enumerate(roles):
        for right in roles[i + 1:]:
            for a in rows_by_role[left]:
                for b in rows_by_role[right]:
                    if a["recording_sha256"] != b["recording_sha256"]:
                        continue
                    if max(int(a["raw_start_sample"]), int(b["raw_start_sample"])) < min(
                        int(a["raw_end_sample"]), int(b["raw_end_sample"])
                    ):
                        raise ValueError(f"raw byte overlap between {left} and {right}")


def assert_no_clean_attack_time_overlap(
    clean_rows: Sequence[Mapping[str, object]], attack_rows: Sequence[Mapping[str, object]]
) -> None:
    """Ensure evaluated attack time support is outside every clean fit role.

    Recording hashes differ, so this is deliberately a receiver-time underlay
    audit rather than a byte-identity audit.
    """
    for clean in clean_rows:
        for attack in attack_rows:
            if max(int(clean["raw_start_sample"]), int(attack["raw_start_sample"])) < min(
                int(clean["raw_end_sample"]), int(attack["raw_end_sample"])
            ):
                raise ValueError("clean role and attack evaluation share receiver-time support")


def normalize_epoch(surface: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    """Remove per-ms prompt gain and global carrier phase."""
    value = np.asarray(surface, dtype=np.complex128)
    if value.shape != (len(DOPPLERS), len(DELAYS)) or not np.isfinite(value).all():
        raise ValueError("finite 11x17 complex CAF required")
    prompt = value[5, 8]
    if abs(prompt) <= epsilon:
        raise ValueError("prompt is too small to normalize")
    return value * np.exp(-1j * np.angle(prompt)) / (abs(prompt) + epsilon)


def aggregate_l20(surfaces: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Robust L20 location and diagonal complex variance, preserving direction."""
    values = np.asarray([normalize_epoch(x) for x in surfaces], dtype=np.complex128)
    if values.shape != (L20, len(DOPPLERS), len(DELAYS)):
        raise ValueError("exactly 20 CAF surfaces are required")
    center = np.median(values.real, axis=0) + 1j * np.median(values.imag, axis=0)
    residual = values - center
    variance = np.median(residual.real ** 2 + residual.imag ** 2, axis=0)
    return center, np.maximum(variance, 1e-8)


@dataclass(frozen=True)
class NFConfig:
    latent_dim: int = 48
    hidden_dim: int = 96
    variance_floor: float = 1e-4
    variance_ceiling: float = 25.0
    context_features: bool = True


class SetNeuralField(nn.Module):
    """PRN-shared DeepSets coordinate-conditioned complex normal field."""

    def __init__(self, config: NFConfig):
        super().__init__()
        self.config = config
        h, z = config.hidden_dim, config.latent_dim
        self.point = nn.Sequential(nn.Linear(4, h), nn.GELU(), nn.Linear(h, z), nn.GELU())
        decoder_in = z + 2 + (2 if config.context_features else 0)
        self.decoder = nn.Sequential(nn.Linear(decoder_in, h), nn.GELU(), nn.Linear(h, h), nn.GELU(), nn.Linear(h, 4))

    def encode(self, coordinates: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        points = torch.cat((coordinates, values), dim=-1)
        encoded = self.point(points) * mask.unsqueeze(-1)
        count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return encoded.sum(dim=1) / count

    def forward(
        self,
        context_coordinates: torch.Tensor,
        context_values: torch.Tensor,
        context_mask: torch.Tensor,
        target_coordinates: torch.Tensor,
        receiver_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(context_coordinates, context_values, context_mask)
        latent = latent.unsqueeze(1).expand(-1, target_coordinates.shape[1], -1)
        fields = [target_coordinates, latent]
        if self.config.context_features:
            if receiver_context is None or receiver_context.shape[-1] != 2:
                raise ValueError("C/N0 and lock context required")
            fields.append(receiver_context.unsqueeze(1).expand(-1, target_coordinates.shape[1], -1))
        output = self.decoder(torch.cat(fields, dim=-1))
        mean = output[..., :2]
        variance = torch.exp(output[..., 2:]).clamp(self.config.variance_floor, self.config.variance_ceiling)
        return mean, variance


def gaussian_nll(target: torch.Tensor, mean: torch.Tensor, variance: torch.Tensor) -> torch.Tensor:
    if target.shape != mean.shape or mean.shape != variance.shape:
        raise ValueError("target, mean, and variance shapes must match")
    return 0.5 * (((target - mean) ** 2) / variance + torch.log(variance)).sum(dim=-1).mean()


@torch.no_grad()
def predict_distribution(
    model: SetNeuralField,
    values: np.ndarray,
    observed: Sequence[int],
    targets: Sequence[int],
    receiver_context: Sequence[float],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.column_stack((np.asarray(values).real, np.asarray(values).imag)).astype(np.float32)
    context_indices = list(observed) or [CENTER]
    # Empty-prior scoring uses a zero mask; CENTER is only a shape placeholder.
    mask_value = 1.0 if observed else 0.0
    cc = torch.as_tensor(GRID[context_indices][None], device=device)
    cv = torch.as_tensor(y[context_indices][None], device=device)
    cm = torch.full((1, len(context_indices)), mask_value, device=device)
    tc = torch.as_tensor(GRID[list(targets)][None], device=device)
    rc = torch.as_tensor(np.asarray(receiver_context, np.float32)[None], device=device)
    mean, variance = model(cc, cv, cm, tc, rc if model.config.context_features else None)
    return mean[0].cpu().numpy(), variance[0].cpu().numpy()


def sequential_trace(
    model: SetNeuralField,
    values: np.ndarray,
    budget: int,
    receiver_context: Sequence[float],
    device: torch.device,
    *,
    policy: str,
    fixed_order: Sequence[int] | None = None,
    magnitude_only: bool = False,
) -> tuple[float, list[dict[str, object]]]:
    """Score only pre-observation innovation; never reconstruct a seen point."""
    if budget < 1 or budget > len(GRID):
        raise ValueError("budget is a complex-coordinate count in [1,187]")
    observed: list[int] = []
    trace: list[dict[str, object]] = []
    for step in range(budget):
        if step == 0:
            selected = CENTER
        elif policy == "active_adaptive":
            remaining = [i for i in range(len(GRID)) if i not in observed]
            _, variance_all = predict_distribution(model, values, observed, remaining, receiver_context, device)
            selected = remaining[int(np.argmax(np.sum(np.log(variance_all), axis=1)))]
        else:
            if fixed_order is None:
                raise ValueError("fixed policy requires an order")
            selected = int(fixed_order[step])
        mean, variance = predict_distribution(model, values, observed, [selected], receiver_context, device)
        actual = np.asarray([values[selected].real, values[selected].imag], dtype=np.float64)
        if magnitude_only:
            delta = np.asarray([abs(values[selected]) - np.linalg.norm(mean[0]), 0.0])
        else:
            delta = actual - mean[0]
        surprise = float(np.sum(delta * delta / variance[0]) + np.sum(np.log(variance[0])))
        di, ci = np.unravel_index(selected, (len(DOPPLERS), len(DELAYS)))
        trace.append({
            "step": step + 1, "index": selected, "delay_chips": float(DELAYS[ci]),
            "doppler_hz": float(DOPPLERS[di]), "actual_real": float(actual[0]),
            "actual_imag": float(actual[1]), "mu_real": float(mean[0, 0]),
            "mu_imag": float(mean[0, 1]), "var_real": float(variance[0, 0]),
            "var_imag": float(variance[0, 1]), "innovation_real": float(delta[0]),
            "innovation_imag": float(delta[1]), "sequential_surprise": surprise,
            "pre_observation": True,
        })
        observed.append(selected)
    return float(np.mean([x["sequential_surprise"] for x in trace])), trace


@lru_cache(maxsize=8)
def fixed_policy_orders(seed: int = 20260808) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    others = [i for i in range(len(GRID)) if i != CENTER]
    random_order = [CENTER, *rng.permutation(others).tolist()]
    # Farthest-point coverage, deterministic and normal/attack independent.
    uniform = [CENTER]
    while len(uniform) < len(GRID):
        candidates = [i for i in range(len(GRID)) if i not in uniform]
        selected = max(candidates, key=lambda i: (min(np.sum((GRID[i] - GRID[j]) ** 2) for j in uniform), -i))
        uniform.append(selected)
    epl = [CENTER, int(np.ravel_multi_index((5, 4), (11, 17))), int(np.ravel_multi_index((5, 12), (11, 17)))]
    delay9 = [CENTER, *[int(np.ravel_multi_index((5, i), (11, 17))) for i in range(4, 13) if i != 8]]
    return {"uniform_fixed": uniform, "random_fixed": random_order, "epl_3": epl, "fixed_delay_9": delay9}


def pool_scores(values: Sequence[float], method: str) -> float:
    x = np.sort(np.asarray(values, dtype=np.float64))
    if x.size < 4 or not np.isfinite(x).all():
        raise ValueError("at least four finite PRN scores required")
    if method == "median":
        return float(np.median(x))
    if method == "topk_mean":
        return float(np.mean(x[-max(1, math.ceil(x.size / 3)):]))
    if method == "trimmed_mean":
        trim = int(math.floor(x.size * 0.1))
        return float(np.mean(x[trim:x.size - trim] if trim else x))
    if method == "soft_topk":
        z = np.exp((x - np.max(x)) / 0.5)
        return float(np.sum(z * x) / np.sum(z))
    raise ValueError(f"unknown pooling method {method}")


def choose_pooling(selection_events: Sequence[Sequence[float]]) -> tuple[str, dict[str, float]]:
    """Choose the most tail-stable pooling on clean selection only."""
    candidates = ("median", "topk_mean", "trimmed_mean", "soft_topk")
    diagnostics: dict[str, float] = {}
    for method in candidates:
        scores = np.asarray([pool_scores(event, method) for event in selection_events])
        diagnostics[method] = float(np.quantile(scores, .99) - np.median(scores))
    return min(candidates, key=lambda name: (diagnostics[name], name)), diagnostics


def binary_metrics(labels: Sequence[int], scores: Sequence[float], max_fpr: float = .05) -> dict[str, float]:
    y = np.asarray(labels, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    if set(y.tolist()) != {0, 1} or len(y) != len(s) or not np.isfinite(s).all():
        raise ValueError("finite two-class labels/scores required")
    return {
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        "normalized_partial_auc_fpr_0_05": float(roc_auc_score(y, s, max_fpr=max_fpr)),
    }


def alarm_metrics(times: Sequence[float], scores: Sequence[float], threshold: float, onset: float) -> dict[str, float | None]:
    t, s = np.asarray(times, float), np.asarray(scores, float)
    alarm = s >= threshold
    post = t >= onset
    alarm_times = t[alarm & post]
    return {
        "attack_detection_rate": float(np.mean(alarm[post])) if np.any(post) else None,
        "first_alarm_delay_s": float(np.min(alarm_times) - onset) if alarm_times.size else None,
        "sustained_alarm_fraction": float(np.mean(alarm[post])) if np.any(post) else None,
    }


def paired_block_bootstrap(
    times: Sequence[float], left: Sequence[float], right: Sequence[float], *, seed: int, replicates: int = 10_000
) -> dict[str, object]:
    """Paired 10-second block mean difference, left minus right."""
    t, a, b = np.asarray(times, float), np.asarray(left, float), np.asarray(right, float)
    if not (len(t) == len(a) == len(b)) or not np.isfinite(np.r_[t, a, b]).all():
        raise ValueError("finite paired inputs required")
    block = np.floor(t / 10.0).astype(np.int64)
    effects = np.asarray([np.mean((a - b)[block == k]) for k in np.unique(block)])
    if effects.size < 2:
        return {"status": "INCONCLUSIVE", "reason": "fewer_than_two_10s_blocks", "blocks": int(effects.size)}
    rng = np.random.default_rng(seed)
    sampled = rng.choice(effects, size=(replicates, effects.size), replace=True).mean(axis=1)
    return {
        "status": "PASS", "effect": float(np.mean(effects)),
        "ci95": [float(np.quantile(sampled, .025)), float(np.quantile(sampled, .975))],
        "block_seconds": 10, "blocks": int(effects.size), "replicates": replicates, "seed": seed,
    }


def verify_freeze_manifest(root: Path, manifest_name: str = "freeze_manifest.json") -> dict[str, str]:
    document = json.loads((root / manifest_name).read_text(encoding="utf-8"))
    required = {
        "model.pt", "model_context.pt", "model_no_context.pt", "normal_field_reference.npz",
        "query_policy.json", "thresholds.json", "pooling.json", "calibration.json",
    }
    if set(document.get("files", {})) != required:
        raise RuntimeError("freeze manifest does not contain the exact protected files")
    for relative, expected in document["files"].items():
        actual = sha256(root / relative)
        if actual != expected:
            raise RuntimeError(f"frozen file drift: {relative}")
    return dict(document["files"])

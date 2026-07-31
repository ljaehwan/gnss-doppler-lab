#!/usr/bin/env python3
"""Train/evaluate DA-PFRT with strict leave-one-scenario-out folds.

DA-PFRT is a discriminative, domain-adversarial Peak–Floor Relational
Transformer. It requires no test-time normal prefix. Each fold excludes the
held scenario from scaler fitting, model fitting, early stopping, and fixed
threshold selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

MAX_PRNS = 32
CADENCE_S = 0.5
DATASET_MANIFEST_SCHEMA = "gnss-doppler-lab.da-pfrt-dataset-manifest.v1"
EXPECTED_SCENARIOS = ("cleanStatic", "os2", "os3", "os4")
_DATASET_SPEC_KEYS = {"morph_csv", "morph_sha256", "floor_csv", "floor_sha256", "onset_s", "identity"}
_IDENTITY_KEYS = {"scenario", "morph_label", "morph_run_id", "morph_source_fingerprint", "floor_scenario"}
TAP_FEATURES = [
    "tap_E4_rel_prompt_mean", "tap_E3_rel_prompt_mean", "tap_E2_rel_prompt_mean",
    "tap_E_rel_prompt_mean", "tap_P_rel_prompt_mean", "tap_L_rel_prompt_mean",
    "tap_L2_rel_prompt_mean", "tap_L3_rel_prompt_mean", "tap_L4_rel_prompt_mean",
]
DEFAULT_MORPH_FEATURES = TAP_FEATURES + [
    "left_right_imbalance_mean", "left_right_imbalance_std",
    "peak_index_mean", "peak_index_std", "peak_width_mean", "peak_width_std",
    "peak_sharpness_mean", "peak_sharpness_std", "prompt_mag_cv",
    "dmcpd_prompt_dominance_mean", "dmcpd_prompt_to_max_side_mean",
    "dmcpd_max_side_to_prompt_mean", "dmcpd_second_side_to_prompt_mean",
    "dmcpd_centroid_shift_mean", "dmcpd_centroid_shift_std",
    "dmcpd_width_variance_mean", "dmcpd_left_right_energy_abs_mean",
    "dmcpd_curvature_e1l1_mean",
    "dmcpd_pair1_signed_asym_mean", "dmcpd_pair1_abs_asym_mean",
    "dmcpd_pair2_signed_asym_mean", "dmcpd_pair2_abs_asym_mean",
    "dmcpd_pair3_signed_asym_mean", "dmcpd_pair3_abs_asym_mean",
    "dmcpd_pair4_signed_asym_mean", "dmcpd_pair4_abs_asym_mean",
]
DEFAULT_FLOOR_FEATURES = [
    "i_mean", "q_mean", "i_std", "q_std", "iq_corr", "power_mean", "power_std",
    "phase_inc_mean", "phase_inc_std", "phase_coh", "psd_entropy", "psd_flatness",
    "amp_mean", "amp_std", "amp_skew", "amp_kurt",
    "damp_mean", "damp_std", "damp_skew", "damp_kurt",
] + [f"ac_{i:02d}" for i in range(21)] + [f"psd_band_{i:02d}" for i in range(16)]


@dataclass
class AlignedData:
    times: np.ndarray
    available_times: np.ndarray
    morph: np.ndarray
    floor: np.ndarray
    prn_mask: np.ndarray
    morph_features: list[str]
    floor_features: list[str]


@dataclass
class SequenceData:
    times: np.ndarray
    available_times: np.ndarray
    morph: np.ndarray
    floor: np.ndarray
    prn_mask: np.ndarray

    def __len__(self) -> int:
        return int(self.times.shape[0])


def _require_numeric(frame: pd.DataFrame, columns: list[str], source: str) -> np.ndarray:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} missing required features: {missing}")
    values = frame[columns].apply(pd.to_numeric, errors="raise").to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{source} contains non-finite feature values")
    return values


def _prn_slot(value: object) -> int:
    text = str(value).strip().upper()
    if text.startswith("G"):
        text = text[1:]
    try:
        number = int(text)
    except ValueError as exc:
        raise ValueError(f"unsupported GPS PRN: {value}") from exc
    if not 1 <= number <= MAX_PRNS:
        raise ValueError(f"GPS PRN outside 1..{MAX_PRNS}: {value}")
    return number - 1


def align_modalities(
    morph_frame: pd.DataFrame, floor_frame: pd.DataFrame,
    morph_features: list[str] | None = None, floor_features: list[str] | None = None,
    tolerance_s: float = 1e-4,
) -> AlignedData:
    morph_features = list(morph_features or DEFAULT_MORPH_FEATURES)
    floor_features = list(floor_features or DEFAULT_FLOOR_FEATURES)
    for column in ("window_bin_s", "prn", "window_end_s"):
        if column not in morph_frame:
            raise ValueError(f"morphology CSV missing {column}")
    for column in ("window_start_s", "window_end_s"):
        if column not in floor_frame:
            raise ValueError(f"floor CSV missing {column}")
    morph, floor = morph_frame.copy(), floor_frame.copy()
    _require_numeric(morph, morph_features, "morphology CSV")
    _require_numeric(floor, floor_features, "floor CSV")
    morph["_time"] = pd.to_numeric(morph["window_bin_s"], errors="raise").astype(float)
    floor["_time"] = pd.to_numeric(floor["window_start_s"], errors="raise").astype(float)
    for name, frame in (("morphology", morph), ("floor", floor)):
        end = pd.to_numeric(frame["window_end_s"], errors="raise").to_numpy(float)
        clock = frame["_time"].to_numpy(float)
        if not np.isfinite(clock).all() or not np.isfinite(end).all():
            raise ValueError(f"{name} CSV requires finite window_end_s and score times")
        if np.any(end < clock):
            raise ValueError(f"{name} CSV window_end_s violates causal availability")
    if floor["_time"].duplicated().any():
        raise ValueError("floor CSV has duplicate window_start_s")
    floor_times = floor["_time"].to_numpy(float)
    rows = []
    for time_value in sorted(morph["_time"].unique()):
        nearest = int(np.argmin(np.abs(floor_times - time_value)))
        if abs(floor_times[nearest] - time_value) <= tolerance_s:
            rows.append((float(time_value), nearest))
    if not rows:
        raise ValueError("no aligned morphology/floor epochs")
    times = np.asarray([row[0] for row in rows], dtype=np.float64)
    count = len(rows)
    morph_tensor = np.zeros((count, MAX_PRNS, len(morph_features)), dtype=np.float32)
    mask = np.zeros((count, MAX_PRNS), dtype=bool)
    floor_tensor = np.zeros((count, len(floor_features)), dtype=np.float32)
    available_times = np.zeros(count, dtype=np.float64)
    grouped = {float(time): group for time, group in morph.groupby("_time", sort=False)}
    for index, (time_value, floor_index) in enumerate(rows):
        group = grouped[time_value]
        slots = [_prn_slot(prn) for prn in group["prn"]]
        if len(slots) != len(set(slots)):
            raise ValueError(f"duplicate PRN at epoch {time_value}")
        for slot, value in zip(slots, _require_numeric(group, morph_features, "morphology CSV")):
            morph_tensor[index, slot] = value
            mask[index, slot] = True
        floor_tensor[index] = floor.iloc[floor_index][floor_features].to_numpy(np.float32)
        morph_available = float(pd.to_numeric(group["window_end_s"], errors="raise").max())
        floor_available = float(floor.iloc[floor_index]["window_end_s"])
        available_times[index] = max(time_value, morph_available, floor_available)
    return AlignedData(times, available_times, morph_tensor, floor_tensor, mask,
                       morph_features, floor_features)


def make_sequences(data: AlignedData, seq_len: int, stride: int = 1) -> SequenceData:
    if seq_len < 2 or stride < 1:
        raise ValueError("seq_len must be >=2 and stride >=1")
    indices = []
    for start in range(0, len(data.times) - seq_len + 1, stride):
        index = np.arange(start, start + seq_len)
        if np.allclose(np.diff(data.times[index]), CADENCE_S, atol=1e-6, rtol=0):
            indices.append(index)
    if not indices:
        raise ValueError("no contiguous sequence windows in partition")
    index = np.stack(indices)
    return SequenceData(data.times[index].astype(np.float64),
                        data.available_times[index].astype(np.float64),
                        data.morph[index].astype(np.float32),
                        data.floor[index].astype(np.float32),
                        data.prn_mask[index].astype(bool))


def sequence_score_available_times(data: SequenceData) -> np.ndarray:
    """Return causal score availability as the maximum over every sequence input."""
    available = np.asarray(data.available_times, dtype=np.float64)
    times = np.asarray(data.times, dtype=np.float64)
    if available.shape != times.shape or not np.isfinite(available).all():
        raise ValueError("sequence availability must be finite and match sequence times")
    if np.any(available < times):
        raise ValueError("sequence availability must be causal")
    return available.max(axis=1)


def fit_robust_scalers(data: AlignedData) -> dict[str, np.ndarray]:
    morph_values = data.morph[data.prn_mask]
    if len(morph_values) == 0:
        raise ValueError("training split has no observed PRNs")
    def fit(values):
        median = np.median(values, axis=0).astype(np.float32)
        mad = (1.4826 * np.median(np.abs(values - median), axis=0)).astype(np.float32)
        std = np.std(values, axis=0).astype(np.float32)
        scale = np.where(np.isfinite(mad) & (mad > 1e-6), mad, std)
        scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32)
        return median, scale
    morph_median, morph_scale = fit(morph_values)
    floor_median, floor_scale = fit(data.floor)
    return {"morph_median": morph_median, "morph_scale": morph_scale,
            "floor_median": floor_median, "floor_scale": floor_scale}


def apply_scalers(data: AlignedData, scalers: Mapping[str, np.ndarray],
                  clip: float = 12.0) -> AlignedData:
    morph = (data.morph - scalers["morph_median"]) / scalers["morph_scale"]
    floor = (data.floor - scalers["floor_median"]) / scalers["floor_scale"]
    morph = np.clip(np.nan_to_num(morph), -clip, clip).astype(np.float32)
    floor = np.clip(np.nan_to_num(floor), -clip, clip).astype(np.float32)
    morph[~data.prn_mask] = 0.0
    return AlignedData(data.times.copy(), data.available_times.copy(), morph, floor,
                       data.prn_mask.copy(), data.morph_features, data.floor_features)

SCHEMA = "gnss-doppler-lab.da-peak-floor-relational-transformer.v1"
SCORE_SCHEMA = "gnss-doppler-lab.da-peak-floor-relational-transformer-score.v1"
DEFAULT_TRAIN_RANGES = {
    "clean": [(None, 330.0)],
    "attack": [(None, 90.0), (130.0, 360.0)],
}
DEFAULT_VALIDATION_RANGES = {
    "clean": [(340.0, 410.0)],
    "attack": [(370.0, None)],
}
DEFAULT_CALIBRATION_RANGES = {
    "clean": [(420.0, None)],
    "attack": [(95.0, 119.0)],
}


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, alpha: float):
        ctx.alpha = float(alpha)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.alpha * gradient, None


def gradient_reverse(value: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return _GradientReverse.apply(value, alpha)


@dataclass
class LabeledSequenceData:
    times: np.ndarray
    available_times: np.ndarray
    morph: np.ndarray
    floor: np.ndarray
    prn_mask: np.ndarray
    labels: np.ndarray
    domains: np.ndarray
    sample_weights: np.ndarray
    scenarios: np.ndarray

    def __len__(self):
        return int(len(self.labels))


class LabeledDataset(Dataset):
    def __init__(self, data: LabeledSequenceData):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (
            torch.from_numpy(self.data.morph[index]),
            torch.from_numpy(self.data.floor[index]),
            torch.from_numpy(self.data.prn_mask[index]),
            torch.tensor(self.data.labels[index], dtype=torch.float32),
            torch.tensor(self.data.domains[index], dtype=torch.long),
            torch.tensor(self.data.sample_weights[index], dtype=torch.float32),
            torch.from_numpy(self.data.times[index]),
            torch.from_numpy(self.data.available_times[index]),
            str(self.data.scenarios[index]),
        )


@dataclass
class DAPFRTConfig:
    morph_dim: int
    floor_dim: int
    domain_count: int
    seq_len: int = 6
    hidden_dim: int = 96
    token_layers: int = 2
    token_heads: int = 4
    dropout: float = 0.15
    modality_dropout_probability: float = 0.0


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fixed_prn_encoding(hidden_dim: int) -> torch.Tensor:
    position = torch.arange(1, MAX_PRNS + 1, dtype=torch.float32)[:, None]
    index = torch.arange(hidden_dim, dtype=torch.float32)[None, :]
    rates = torch.pow(10_000.0, -2.0 * torch.floor(index / 2.0) / hidden_dim)
    angle = position * rates
    return torch.where((index.long() % 2) == 0, torch.sin(angle), torch.cos(angle))


def build_fold_contract(scenarios: Iterable[str], held_out: str) -> dict:
    names = list(dict.fromkeys(str(x) for x in scenarios))
    if "cleanStatic" not in names:
        raise ValueError("fold contract requires cleanStatic")
    if held_out == "cleanStatic" or held_out not in names:
        raise ValueError("held_out must be a non-clean listed scenario")
    training = [name for name in names if name != held_out]
    attack_training = [name for name in training if name != "cleanStatic"]
    if not attack_training:
        raise ValueError("fold requires at least one training attack scenario")
    return {
        "held_out": held_out,
        "training_scenarios": training,
        "validation_scenarios": training.copy(),
        "threshold_scenarios": training.copy(),
        "held_scenario_used_for_scaling": False,
        "held_scenario_used_for_training": False,
        "held_scenario_used_for_threshold": False,
    }


def _range_mask(times: np.ndarray, ranges: Iterable[tuple[float | None, float | None]]) -> np.ndarray:
    keep = np.zeros(len(times), dtype=bool)
    for start, end in ranges:
        part = np.ones(len(times), dtype=bool)
        if start is not None:
            part &= times >= float(start)
        if end is not None:
            part &= times <= float(end)
        keep |= part
    return keep


def subset_aligned(data: AlignedData, ranges: Iterable[tuple[float | None, float | None]]) -> AlignedData:
    keep = _range_mask(data.times, ranges)
    if not keep.any():
        raise ValueError("aligned subset is empty")
    return AlignedData(
        data.times[keep], data.available_times[keep], data.morph[keep], data.floor[keep],
        data.prn_mask[keep], data.morph_features, data.floor_features,
    )


def concatenate_aligned(items: list[AlignedData]) -> AlignedData:
    if not items:
        raise ValueError("no aligned datasets to concatenate")
    morph_features, floor_features = items[0].morph_features, items[0].floor_features
    if any(x.morph_features != morph_features or x.floor_features != floor_features for x in items):
        raise ValueError("feature contract mismatch across datasets")
    return AlignedData(
        np.concatenate([x.times for x in items]), np.concatenate([x.available_times for x in items]),
        np.concatenate([x.morph for x in items]), np.concatenate([x.floor for x in items]),
        np.concatenate([x.prn_mask for x in items]), morph_features, floor_features,
    )


def make_labeled_sequences(
    data: AlignedData,
    seq_len: int,
    domain_id: int,
    onset_s: float | None,
    endpoint_ranges: Iterable[tuple[float | None, float | None]],
    scenario: str = "scenario",
    transition_end_s: float | None = None,
) -> LabeledSequenceData:
    base = make_sequences(data, seq_len=seq_len, stride=1)
    endpoint = base.times[:, -1]
    keep = _range_mask(endpoint, endpoint_ranges)
    if not keep.any():
        raise ValueError(f"no sequence endpoints selected for {scenario}")
    labels = np.zeros(int(keep.sum()), dtype=np.float32)
    selected_endpoint = endpoint[keep]
    selected_available = sequence_score_available_times(base)[keep]
    if onset_s is not None:
        # Causal support contract: normal only if every input is available before onset;
        # attack only after one full 1 s morphology support interval. Boundary samples
        # are marked uncertain (-1) and excluded from losses/metrics.
        normal = selected_available < float(onset_s)
        attack = selected_endpoint >= float(onset_s) + 1.0
        labels = np.full(len(selected_endpoint), -1.0, dtype=np.float32)
        labels[normal] = 0.0
        labels[attack] = 1.0
    weights = np.ones_like(labels, dtype=np.float32)
    weights[labels < 0] = 0.0
    if onset_s is not None and transition_end_s is not None:
        transition = (labels == 1) & (selected_endpoint < transition_end_s)
        weights[transition] = 0.5
    return LabeledSequenceData(
        base.times[keep], base.available_times[keep], base.morph[keep], base.floor[keep],
        base.prn_mask[keep], labels, np.full(len(labels), int(domain_id), dtype=np.int64),
        weights, np.full(len(labels), scenario, dtype=object),
    )


def concatenate_sequences(items: list[LabeledSequenceData]) -> LabeledSequenceData:
    if not items:
        raise ValueError("no sequence datasets")
    return LabeledSequenceData(*[
        np.concatenate([getattr(x, field) for x in items])
        for field in ("times", "available_times", "morph", "floor", "prn_mask", "labels",
                      "domains", "sample_weights", "scenarios")
    ])


class DomainAdversarialPeakFloorRelationalTransformer(nn.Module):
    def __init__(self, config: DAPFRTConfig):
        super().__init__()
        if config.hidden_dim % config.token_heads:
            raise ValueError("hidden_dim must be divisible by token_heads")
        self.config = config
        h = config.hidden_dim
        self.morph_encoder = nn.Sequential(nn.Linear(config.morph_dim, h), nn.LayerNorm(h), nn.GELU())
        self.floor_encoder = nn.Sequential(nn.Linear(config.floor_dim, h), nn.LayerNorm(h), nn.GELU())
        self.register_buffer("prn_encoding", _fixed_prn_encoding(h), persistent=True)
        self.modality_embedding = nn.Parameter(torch.randn(2, h) * 0.02)
        layer = nn.TransformerEncoderLayer(
            h, config.token_heads, 4 * h, config.dropout, activation="gelu",
            batch_first=True, norm_first=False,
        )
        self.relation_encoder = nn.TransformerEncoder(layer, config.token_layers, nn.LayerNorm(h))
        self.temporal_encoder = nn.GRU(h, h, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(3 * h, 2 * h), nn.LayerNorm(2 * h), nn.GELU(),
                                    nn.Dropout(config.dropout), nn.Linear(2 * h, h), nn.GELU())
        self.spoof_head = nn.Linear(h, 1)
        self.snapshot_head = nn.Linear(h, 1)
        self.peak_head = nn.Linear(h, 1)
        self.floor_head = nn.Linear(h, 1)
        self.domain_head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Dropout(config.dropout),
                                         nn.Linear(h, config.domain_count))
        self.match_head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))

    def _encode_epochs(self, morph: torch.Tensor, floor: torch.Tensor, prn_mask: torch.Tensor):
        b, t, n, fm = morph.shape
        bt = b * t
        nodes = self.morph_encoder(morph.reshape(bt, n, fm))
        nodes = nodes + self.prn_encoding[None] + self.modality_embedding[0]
        flat_mask = prn_mask.reshape(bt, n)
        valid = flat_mask.unsqueeze(-1)
        peak_epoch = (nodes * valid).sum(1) / valid.sum(1).clamp_min(1)
        floor_epoch = self.floor_encoder(floor.reshape(bt, -1)) + self.modality_embedding[1]
        tokens = torch.cat([nodes, floor_epoch[:, None, :]], dim=1)
        padding = torch.cat([~flat_mask, torch.zeros(bt, 1, dtype=torch.bool, device=morph.device)], dim=1)
        related = self.relation_encoder(tokens, src_key_padding_mask=padding)
        relation_valid = (~padding).unsqueeze(-1)
        relation_epoch = (related * relation_valid).sum(1) / relation_valid.sum(1).clamp_min(1)
        return (relation_epoch.reshape(b, t, -1), peak_epoch.reshape(b, t, -1),
                floor_epoch.reshape(b, t, -1))

    def forward(self, morph, floor, prn_mask, grl_alpha: float = 1.0,
                corrupt_modalities: bool = True, negative_floor: torch.Tensor | None = None):
        b, t, n, fm = morph.shape
        if (t, n, fm) != (self.config.seq_len, MAX_PRNS, self.config.morph_dim):
            raise ValueError("morph input contract violation")
        if floor.shape != (b, t, self.config.floor_dim) or prn_mask.shape != (b, t, n):
            raise ValueError("floor/mask input contract violation")
        morph_in, floor_in = morph, floor
        if self.training and corrupt_modalities and self.config.modality_dropout_probability > 0:
            p = self.config.modality_dropout_probability
            drop_peak = torch.rand(b, 1, 1, 1, device=morph.device) < p
            drop_floor = torch.rand(b, 1, 1, device=floor.device) < p
            drop_floor &= ~drop_peak.reshape(b, 1, 1)
            morph_in = morph.masked_fill(drop_peak, 0.0)
            floor_in = floor.masked_fill(drop_floor, 0.0)
        relation_seq, peak_seq, floor_seq = self._encode_epochs(morph_in, floor_in, prn_mask)
        _, temporal_hidden = self.temporal_encoder(relation_seq)
        temporal = temporal_hidden[-1]
        snapshot = relation_seq[:, -1]
        peak_last, floor_last = peak_seq[:, -1], floor_seq[:, -1]
        fused = self.fusion(torch.cat([snapshot, temporal, torch.abs(snapshot - temporal)], dim=-1))
        if negative_floor is None:
            negative_relation = snapshot.detach()
            negative_logit = torch.full((b,), float("nan"), device=morph.device)
        else:
            negative_relation_seq, _, _ = self._encode_epochs(morph_in, negative_floor, prn_mask)
            negative_relation = negative_relation_seq[:, -1]
            negative_logit = self.match_head(negative_relation).squeeze(-1)
        return {
            "spoof_logit": self.spoof_head(fused).squeeze(-1),
            "snapshot_logit": self.snapshot_head(snapshot).squeeze(-1),
            "peak_logit": self.peak_head(peak_last).squeeze(-1),
            "floor_logit": self.floor_head(floor_last).squeeze(-1),
            "domain_logits": self.domain_head(gradient_reverse(fused, grl_alpha)),
            "match_positive_logit": self.match_head(snapshot).squeeze(-1),
            "match_negative_logit": negative_logit,
            "embedding": fused,
        }


def focal_binary_loss(logits, targets, sample_weights, gamma: float = 2.0):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probability = torch.sigmoid(logits)
    pt = targets * probability + (1 - targets) * (1 - probability)
    return (((1 - pt).pow(gamma) * bce) * sample_weights).sum() / sample_weights.sum().clamp_min(1)


def build_hard_negative_indices(domains: torch.Tensor, labels: torch.Tensor, endpoint_times: torch.Tensor):
    """Closest-time derangement constrained to the same domain and class."""
    size = len(labels)
    indices = torch.arange(size, device=labels.device)
    valid = torch.zeros(size, dtype=torch.bool, device=labels.device)
    for i in range(size):
        candidates = torch.where((domains == domains[i]) & (labels == labels[i]) &
                                 (torch.arange(size, device=labels.device) != i) &
                                 (endpoint_times != endpoint_times[i]))[0]
        if len(candidates):
            distance = torch.abs(endpoint_times[candidates] - endpoint_times[i])
            indices[i] = candidates[torch.argmin(distance)]
            valid[i] = True
    return indices, valid


def multitask_loss(outputs, labels, domains, weights, match_valid,
                   domain_weight=0.20, match_weight=0.20, snapshot_weight=0.10):
    valid_class = labels >= 0
    safe_labels = labels.clamp(0, 1)
    class_weights = weights * valid_class.float()
    primary = focal_binary_loss(outputs["spoof_logit"], safe_labels, class_weights)
    snapshot = focal_binary_loss(outputs["snapshot_logit"], safe_labels, class_weights)
    # Branch heads are diagnostics only: no branch loss enters the training objective.
    peak = focal_binary_loss(outputs["peak_logit"], safe_labels, class_weights)
    floor = focal_binary_loss(outputs["floor_logit"], safe_labels, class_weights)
    normal_domain = labels == 0
    if normal_domain.any():
        domain = nn.functional.cross_entropy(outputs["domain_logits"][normal_domain], domains[normal_domain])
    else:
        domain = outputs["domain_logits"].sum() * 0.0
    if match_valid.any():
        positive = nn.functional.binary_cross_entropy_with_logits(
            outputs["match_positive_logit"][match_valid], torch.ones_like(labels[match_valid]))
        negative = nn.functional.binary_cross_entropy_with_logits(
            outputs["match_negative_logit"][match_valid], torch.zeros_like(labels[match_valid]))
        matching = 0.5 * (positive + negative)
    else:
        matching = outputs["match_positive_logit"].sum() * 0.0
    total = primary + snapshot_weight * snapshot + domain_weight * domain + match_weight * matching
    return {"total": total, "primary": primary, "snapshot": snapshot, "peak": peak.detach(),
            "floor": floor.detach(), "domain": domain, "matching": matching}


def select_fixed_threshold(probabilities: np.ndarray, labels: np.ndarray, target_fpr: float = 0.01) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(labels, dtype=int)
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be between zero and one")
    normal = p[y == 0]
    if len(normal) == 0 or not np.isfinite(normal).all():
        raise ValueError("validation normal probabilities required")
    allowed = int(math.floor(target_fpr * len(normal)))
    descending = np.sort(normal)[::-1]
    if allowed <= 0:
        return float(np.nextafter(descending[0], math.inf))
    candidate = float(descending[allowed - 1])
    if np.count_nonzero(normal >= candidate) <= allowed:
        return candidate
    return float(np.nextafter(candidate, math.inf))


def _loader(data: LabeledSequenceData, batch_size: int, training: bool) -> DataLoader:
    dataset = LabeledDataset(data)
    sampler = None
    shuffle = False
    if training:
        labels = data.labels.astype(int)
        valid = labels >= 0
        counts = np.bincount(labels[valid], minlength=2).astype(float)
        class_weight = np.where(counts > 0, valid.sum() / (2.0 * counts), 0.0)
        weights = np.zeros(len(labels), dtype=float)
        weights[valid] = class_weight[labels[valid]]
        sampler = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), int(valid.sum()), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle, num_workers=0)


def _batch_to_device(batch, device):
    morph, floor, mask, label, domain, weight, times, available, scenarios = batch
    return (morph.to(device), floor.to(device), mask.to(device), label.to(device),
            domain.to(device), weight.to(device), times, available, scenarios)


def _run_epoch(model, loader, optimizer, device, grl_alpha):
    training = optimizer is not None
    model.train(training)
    sums = {k: 0.0 for k in ("total", "primary", "snapshot", "peak", "floor", "domain", "matching")}
    count = 0
    for batch in loader:
        morph, floor, mask, label, domain, weight, times, _available, _scenarios = _batch_to_device(batch, device)
        negative_indices, match_valid = build_hard_negative_indices(
            domain, label, times[:, -1].to(device))
        negative_floor = floor[negative_indices]
        outputs = model(morph, floor, mask, grl_alpha=grl_alpha,
                        corrupt_modalities=training, negative_floor=negative_floor)
        losses = multitask_loss(outputs, label, domain, weight, match_valid)
        if training:
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        size = len(label)
        for key in sums:
            sums[key] += float(losses[key].detach().cpu()) * size
        count += size
    return {k: v / max(count, 1) for k, v in sums.items()} | {"sequences": count}


@torch.no_grad()
def predict(model, data: LabeledSequenceData, batch_size: int, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    for batch in _loader(data, batch_size, False):
        morph, floor, mask, label, domain, _weight, times, available, scenarios = _batch_to_device(batch, device)
        outputs = model(morph, floor, mask, grl_alpha=0.0, corrupt_modalities=False)
        probabilities = {name: torch.sigmoid(outputs[name]).cpu().numpy() for name in
                         ("spoof_logit", "snapshot_logit", "peak_logit", "floor_logit")}
        embeddings = outputs["embedding"].cpu().numpy()
        for index in range(len(label)):
            rows.append({
                "scenario": str(scenarios[index]),
                "sequence_start_s": float(times[index, 0]),
                "window_start_s": float(times[index, -1]),
                "available_time_s": float(torch.max(available[index]).item()),
                "label": int(label[index].cpu()),
                "domain_id": int(domain[index].cpu()),
                "spoof_probability": float(probabilities["spoof_logit"][index]),
                "snapshot_probability": float(probabilities["snapshot_logit"][index]),
                "peak_probability": float(probabilities["peak_logit"][index]),
                "floor_probability": float(probabilities["floor_logit"][index]),
                "embedding_norm": float(np.linalg.norm(embeddings[index])),
            })
    return pd.DataFrame(rows).sort_values(["scenario", "window_start_s"]).reset_index(drop=True)


def _first_three_delay(frame: pd.DataFrame, threshold: float, onset_s: float):
    flags = frame.spoof_probability.to_numpy() >= threshold
    available = frame.available_time_s.to_numpy(float)
    nominal = frame.window_start_s.to_numpy(float)
    for index in range(len(flags) - 2):
        if nominal[index] >= onset_s and flags[index:index + 3].all():
            alarm_time = float(available[index + 2])
            return alarm_time - onset_s, alarm_time
    return None, None


def classification_metrics(frame: pd.DataFrame, threshold: float, onset_s: float | None = None) -> dict:
    valid = frame.label.to_numpy(int) >= 0
    evaluation = frame.loc[valid].copy()
    labels = evaluation.label.to_numpy(int)
    probabilities = evaluation.spoof_probability.to_numpy(float)
    predictions = probabilities >= threshold
    result = {
        "windows": int(len(evaluation)), "uncertain_boundary_windows_excluded": int((~valid).sum()),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(labels, probabilities)) if len(np.unique(labels)) == 2 else None,
        "average_precision": float(average_precision_score(labels, probabilities)) if labels.sum() else None,
        "normal_false_positive_rate": float(predictions[labels == 0].mean()) if np.any(labels == 0) else None,
        "attack_true_positive_rate": float(predictions[labels == 1].mean()) if np.any(labels == 1) else None,
    }
    if onset_s is not None:
        causal_pre = evaluation.available_time_s.to_numpy(float) < onset_s
        post = evaluation.window_start_s.to_numpy(float) >= onset_s + 1.0
        delay, alarm_time = _first_three_delay(evaluation, threshold, onset_s)
        result.update({
            "causal_pre_onset_windows": int(causal_pre.sum()),
            "causal_pre_onset_false_positive_rate": float(predictions[causal_pre].mean()) if causal_pre.any() else None,
            "post_onset_windows": int(post.sum()),
            "post_onset_true_positive_rate": float(predictions[post].mean()) if post.any() else None,
            "first_three_consecutive_delay_s": delay,
            "first_three_consecutive_available_s": alarm_time,
        })
    return result


def _read_manifest(path: Path) -> dict:
    try:
        document = json.loads(Path(path).read_text())
    except Exception as exc:
        raise ValueError("invalid dataset manifest JSON") from exc
    _validate_manifest_contract(document)
    return document


def _validate_manifest_contract(manifest: dict) -> None:
    if not isinstance(manifest, dict) or manifest.get("schema") != DATASET_MANIFEST_SCHEMA:
        raise ValueError("dataset manifest schema mismatch")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or set(datasets) != set(EXPECTED_SCENARIOS):
        raise ValueError("dataset manifest scenario roster mismatch")
    for scenario in EXPECTED_SCENARIOS:
        spec = datasets[scenario]
        if not isinstance(spec, dict) or set(spec) != _DATASET_SPEC_KEYS:
            raise ValueError(f"dataset manifest spec schema mismatch: {scenario}")
        identity = spec.get("identity")
        if not isinstance(identity, dict) or set(identity) != _IDENTITY_KEYS:
            raise ValueError(f"dataset identity schema mismatch: {scenario}")
        expected = {"scenario": scenario, "morph_label": f"oakbat_{scenario}_9tap", "morph_run_id": f"oakbat-{scenario}-method-a-9tap", "floor_scenario": f"oakbat_{scenario}"}
        if any(identity.get(key) != value for key, value in expected.items()):
            raise ValueError(f"morphology-floor identity linkage mismatch: {scenario}")
        if not isinstance(identity.get("morph_source_fingerprint"), str) or not __import__("re").fullmatch(r"[0-9a-f]{64}", identity["morph_source_fingerprint"]):
            raise ValueError(f"dataset identity fingerprint mismatch: {scenario}")
        onset = spec["onset_s"]
        if (scenario == "cleanStatic") != (onset is None):
            raise ValueError(f"onset identity mismatch: {scenario}")
        if onset is not None and (not isinstance(onset, (int, float)) or not math.isfinite(float(onset))):
            raise ValueError(f"non-finite onset: {scenario}")
        for key in ("morph_sha256", "floor_sha256"):
            if not isinstance(spec[key], str) or not __import__("re").fullmatch(r"[0-9a-f]{64}", spec[key]):
                raise ValueError(f"invalid SHA256 pin: {scenario}/{key}")


def _single_identity(frame: pd.DataFrame, columns: Mapping[str, str], scenario: str, modality: str) -> None:
    for column, expected in columns.items():
        if column not in frame or frame[column].isna().any() or frame[column].astype(str).nunique() != 1 or str(frame[column].iloc[0]) != expected:
            raise ValueError(f"{scenario} {modality} identity mismatch: {column}")


def _load_datasets(manifest: dict) -> tuple[dict[str, AlignedData], dict[str, float | None], dict]:
    _validate_manifest_contract(manifest)
    paths, identities = {}, {}
    for scenario in EXPECTED_SCENARIOS:
        spec = manifest["datasets"][scenario]
        morph_path, floor_path = Path(spec["morph_csv"]).resolve(), Path(spec["floor_csv"]).resolve()
        for modality, path in (("morph", morph_path), ("floor", floor_path)):
            if not path.is_file():
                raise ValueError(f"missing {modality} dataset: {scenario}")
            if sha256(path) != spec[f"{modality}_sha256"]:
                raise ValueError(f"{scenario} {modality} SHA256 mismatch")
        paths[scenario] = morph_path, floor_path
        identities[scenario] = {"morph_csv": str(morph_path), "morph_sha256": spec["morph_sha256"], "floor_csv": str(floor_path), "floor_sha256": spec["floor_sha256"], "onset_s": spec["onset_s"], "identity": dict(spec["identity"])}
    for modality in ("morph", "floor"):
        hashes = [identity[f"{modality}_sha256"] for identity in identities.values()]
        if len(hashes) != len(set(hashes)):
            raise ValueError(f"duplicate {modality} dataset identity across scenario names")
    aligned, onsets = {}, {}
    for scenario in EXPECTED_SCENARIOS:
        spec, identity = manifest["datasets"][scenario], manifest["datasets"][scenario]["identity"]
        morph_path, floor_path = paths[scenario]
        morph, floor = pd.read_csv(morph_path), pd.read_csv(floor_path)
        _single_identity(morph, {"label": identity["morph_label"], "run_id": identity["morph_run_id"], "source_fingerprint": identity["morph_source_fingerprint"]}, scenario, "morphology")
        _single_identity(floor, {"scenario": identity["floor_scenario"]}, scenario, "floor")
        aligned[scenario] = align_modalities(morph, floor)
        onsets[scenario] = None if spec["onset_s"] is None else float(spec["onset_s"])
    return aligned, onsets, identities


def _scaled_sequences_for_fold(aligned, onsets, held_out, config: DAPFRTConfig):
    contract = build_fold_contract(aligned.keys(), held_out)
    domain_names = contract["training_scenarios"]
    domain_map = {name: index for index, name in enumerate(domain_names)}
    scaler_parts = []
    for scenario in domain_names:
        ranges = DEFAULT_TRAIN_RANGES["clean" if onsets[scenario] is None else "attack"]
        scaler_parts.append(subset_aligned(aligned[scenario], ranges))
    scalers = fit_robust_scalers(concatenate_aligned(scaler_parts))
    scaled = {name: apply_scalers(value, scalers) for name, value in aligned.items()}
    train_sets, validation_sets, calibration_sets = [], [], []
    for scenario in domain_names:
        attack = onsets[scenario] is not None
        train_ranges = DEFAULT_TRAIN_RANGES["attack" if attack else "clean"]
        val_ranges = DEFAULT_VALIDATION_RANGES["attack" if attack else "clean"]
        calibration_ranges = DEFAULT_CALIBRATION_RANGES["attack" if attack else "clean"]
        train_sets.append(make_labeled_sequences(
            scaled[scenario], config.seq_len, domain_map[scenario], onsets[scenario], train_ranges,
            scenario, transition_end_s=None))
        validation_sets.append(make_labeled_sequences(
            scaled[scenario], config.seq_len, domain_map[scenario], onsets[scenario], val_ranges, scenario))
        calibration_sets.append(make_labeled_sequences(
            scaled[scenario], config.seq_len, domain_map[scenario], onsets[scenario], calibration_ranges, scenario))
    test = make_labeled_sequences(
        scaled[held_out], config.seq_len, -1, onsets[held_out], [(None, None)], held_out)
    return (concatenate_sequences(train_sets), concatenate_sequences(validation_sets),
            concatenate_sequences(calibration_sets), test, scalers, contract, domain_map)


def run_fold(
    dataset_manifest: str | Path,
    held_out: str,
    output_dir: str | Path,
    *, epochs: int = 80, batch_size: int = 64, seq_len: int = 6,
    hidden_dim: int = 96, token_layers: int = 2, token_heads: int = 4,
    dropout: float = 0.15, modality_dropout_probability: float = 0.0,
    lr: float = 3e-4, weight_decay: float = 1e-4, patience: int = 12,
    target_fpr: float = 0.01, device: str | None = None, seed: int = 29,
):
    seed_all(seed)
    manifest_path = Path(dataset_manifest).resolve()
    manifest = _read_manifest(manifest_path)
    aligned, onsets, identities = _load_datasets(manifest)
    contract = build_fold_contract(aligned.keys(), held_out)
    config = DAPFRTConfig(
        len(DEFAULT_MORPH_FEATURES), len(DEFAULT_FLOOR_FEATURES),
        domain_count=len(contract["training_scenarios"]), seq_len=seq_len,
        hidden_dim=hidden_dim, token_layers=token_layers, token_heads=token_heads,
        dropout=dropout, modality_dropout_probability=modality_dropout_probability,
    )
    train_data, val_data, calibration_data, test_data, scalers, contract, domain_map = _scaled_sequences_for_fold(
        aligned, onsets, held_out, config)
    if (held_out in set(train_data.scenarios) or held_out in set(val_data.scenarios) or
            held_out in set(calibration_data.scenarios)):
        raise RuntimeError("held scenario leakage detected")
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = DomainAdversarialPeakFloorRelationalTransformer(config).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_path = output / "model.pt"
    history, best, stale = [], math.inf, 0
    train_loader = _loader(train_data, batch_size, True)
    validation_loader = _loader(val_data, batch_size, False)
    for epoch in range(1, epochs + 1):
        progress = (epoch - 1) / max(epochs - 1, 1)
        grl_alpha = float(2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)
        train_stats = _run_epoch(model, train_loader, optimizer, dev, grl_alpha)
        val_stats = _run_epoch(model, validation_loader, None, dev, 0.0)
        row = {"epoch": epoch, "grl_alpha": grl_alpha,
               **{f"train_{k}": v for k, v in train_stats.items()},
               **{f"validation_{k}": v for k, v in val_stats.items()}}
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if val_stats["primary"] < best:
            best, stale = val_stats["primary"], 0
            torch.save({"model_state_dict": model.state_dict(), "model_config": asdict(config),
                        "feature_contract": {"morph_features": DEFAULT_MORPH_FEATURES,
                                             "floor_features": DEFAULT_FLOOR_FEATURES}}, checkpoint_path)
        else:
            stale += 1
            if stale >= patience:
                break
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    validation_scores = predict(model, val_data, batch_size, dev)
    calibration_scores = predict(model, calibration_data, batch_size, dev)
    threshold = select_fixed_threshold(calibration_scores.spoof_probability.to_numpy(),
                                       calibration_scores.label.to_numpy(), target_fpr)
    test_scores = predict(model, test_data, batch_size, dev)
    validation_scores.to_csv(output / "validation_scores.csv", index=False)
    calibration_scores.to_csv(output / "calibration_scores.csv", index=False)
    test_scores.to_csv(output / "held_scenario_scores.csv", index=False)
    metrics = {
        "validation": classification_metrics(validation_scores, threshold),
        "calibration": classification_metrics(calibration_scores, threshold),
        "held_scenario": classification_metrics(test_scores, threshold, onsets[held_out]),
    }
    achieved_calibration_fpr = float((calibration_scores.loc[calibration_scores.label == 0, "spoof_probability"] >= threshold).mean())
    threshold_doc = {
        "method": "disjoint_calibration_normal_empirical_fpr_control", "target_fpr": target_fpr,
        "achieved_empirical_fpr": achieved_calibration_fpr,
        "threshold": threshold, "calibration_scenarios": sorted(set(calibration_scores.scenario)),
        "held_scenario_used": False, "normal_calibration_windows": int((calibration_scores.label == 0).sum()),
        "calibration_reused_for_early_stopping": False,
    }
    (output / "threshold.json").write_text(json.dumps(threshold_doc, indent=2, sort_keys=True) + "\n")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    scaler_doc = {key: value.tolist() for key, value in scalers.items()}
    (output / "scalers.json").write_text(json.dumps(scaler_doc, indent=2, sort_keys=True) + "\n")
    metadata = {
        "schema": SCHEMA, "architecture": "DomainAdversarialPeakFloorRelationalTransformer",
        "held_out": held_out, "fold_contract": contract, "domain_map": domain_map,
        "model_config": asdict(config), "parameters": sum(p.numel() for p in model.parameters()),
        "best_validation_primary_loss": best, "fixed_threshold": threshold_doc,
        "normal_prefix_required_at_test": False, "test_time_adaptation": False,
        "dataset_manifest": str(manifest_path), "dataset_manifest_sha256": sha256(manifest_path),
        "datasets": identities, "device": str(dev), "seed": seed,
    }
    (output / "model_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    names = ["model.pt", "model_metadata.json", "scalers.json", "training_history.csv",
             "validation_scores.csv", "calibration_scores.csv", "held_scenario_scores.csv",
             "threshold.json", "metrics.json"]
    artifact_manifest = {"schema": SCHEMA, "artifacts": {name: sha256(output / name) for name in names}}
    (output / "campaign_manifest.json").write_text(json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n")
    return {"output_dir": str(output), "held_out": held_out, "metrics": metrics,
            "threshold": threshold, "normal_prefix_required_at_test": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--held-out", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--token-layers", type=int, default=2)
    parser.add_argument("--token-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--modality-dropout-probability", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--target-fpr", type=float, default=0.01)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    result = run_fold(
        args.dataset_manifest, args.held_out, args.output_dir, epochs=args.epochs,
        batch_size=args.batch_size, seq_len=args.seq_len, hidden_dim=args.hidden_dim,
        token_layers=args.token_layers, token_heads=args.token_heads, dropout=args.dropout,
        modality_dropout_probability=args.modality_dropout_probability, lr=args.lr,
        weight_decay=args.weight_decay, patience=args.patience, target_fpr=args.target_fpr,
        device=args.device, seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

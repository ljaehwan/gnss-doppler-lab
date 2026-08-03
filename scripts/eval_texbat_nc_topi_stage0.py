#!/usr/bin/env python3
"""Frozen NC-TOPI Stage-0 TEXBAT data runner.

The module is intentionally importable: data lineage, B0 inference, IQ extraction,
fit gating and atomic staging are independently testable.  Attack labels are
unreachable until every clean-only fit object has been sealed.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gnss_doppler_lab import nc_topi as core

TAP_NAMES = ("E4", "E3", "E2", "E", "P", "L", "L2", "L3", "L4")
TAP_COORDS = tuple(float(x) for x in core.CANONICAL_TAP_COORDS)
NPZ_FIELDS = ("complex_iq", "time_s", "prn", "channel", "segment_index", "sample_count")
SORT_FIELDS = ("segment_index", "channel", "prn", "time_s", "sample_count", "original_index")
METHODS = (
    "B0", "total_Mahalanobis", "amp_only", "shift_only", "amp_shift",
    "amp_shift_width", "TOPI", "NC_TOPI", "NC_TOPI_time_shuffle",
    "NC_TOPI_conditioning_removed", "tangent_energy", "perp_energy", "cross_energy",
)
PRIMARY_METHODS = ("B0", "TOPI", "NC_TOPI", "NC_TOPI_time_shuffle")
AGGREGATORS = ("median", "top25_mean")
IQ_FEATURE_NAMES = core.CONDITIONER_FEATURE_SCHEMA
EPS = 1e-6


def sha256_file(path: str | Path, chunk: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            data = stream.read(chunk)
            if not data:
                break
            digest.update(data)
    return digest.hexdigest()


def canonical_digest(values: Sequence[float]) -> str:
    a = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(str(a.shape).encode() + b"|float64|" + a.tobytes()).hexdigest()


def identity(recording: str, scenario: str, prn: str, index: int, availability: float):
    return core.EpochIdentity(str(recording), str(scenario), str(prn), int(index), float(availability))


@dataclass(frozen=True)
class NodeWindow:
    recording_id: str
    scenario: str
    prn: str
    channel: int
    segment_index: int
    window_index: int
    source_start_s: float
    source_end_s: float
    window_mid_s: float
    window_bin_s: float
    epoch_count: int
    source_sample_min: int
    source_sample_max: int
    actual_raw: np.ndarray

    def __post_init__(self):
        peak = np.asarray(self.actual_raw, dtype=np.float64)
        if peak.shape != (9,) or not np.isfinite(peak).all():
            raise ValueError("NodeWindow actual_raw must be a finite canonical nine-tap vector")
        if self.source_end_s < self.source_start_s or self.epoch_count < 1:
            raise ValueError("NodeWindow source support is invalid")
        frozen = np.frombuffer(np.ascontiguousarray(peak).tobytes(), dtype=np.float64)
        frozen.setflags(write=False)
        object.__setattr__(self, "actual_raw", frozen)

    @property
    def availability_time_s(self) -> float:
        return float(self.source_end_s)

    @property
    def group_key(self) -> tuple[str, int, int, str]:
        return (self.recording_id, int(self.segment_index), int(self.channel), self.prn)

    @property
    def event_key(self) -> tuple[str, float]:
        return (self.recording_id, float(self.window_bin_s))

    @property
    def epoch_identity(self):
        return identity(self.recording_id, self.scenario, self.prn,
                        self.window_index, self.availability_time_s)

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "actual_raw": np.array(self.actual_raw, copy=True)}


@dataclass(frozen=True)
class SequenceExample:
    history: tuple[NodeWindow, ...]
    target: NodeWindow


@dataclass(frozen=True)
class IQBlock:
    recording_id: str
    start_s: float
    end_s: float
    sample_offset: int
    sample_count: int
    features: np.ndarray

    def __post_init__(self):
        values = np.asarray(self.features, dtype=np.float64)
        if values.shape != (4,) or not np.isfinite(values).all():
            raise ValueError("IQBlock requires four finite frozen features")
        object.__setattr__(self, "features", values)


class StageGate:
    """One-way clean-fit state machine; attack metadata cannot leak into fit."""
    REQUIRED_FITS = ("covariance", "conditioner", "conditioner_cap", "shuffled_conditioner",
                     "shuffled_cap", "thresholds")

    def __init__(self):
        self._sealed: dict[str, str] = {}
        self._frozen = False
        self._attack_loaded = False

    def seal_fit(self, name: str, row_identities: Iterable[str]) -> str:
        if self._frozen:
            raise RuntimeError("fit state is frozen")
        if name not in self.REQUIRED_FITS:
            raise ValueError(f"unknown fit object {name!r}")
        rows = tuple(str(x) for x in row_identities)
        if not rows or any("cleanStatic" not in x for x in rows):
            raise ValueError("fit identities must be nonempty cleanStatic-only identities")
        payload = "\n".join(rows).encode()
        seal = hashlib.sha256(payload).hexdigest()
        if name in self._sealed and self._sealed[name] != seal:
            raise RuntimeError(f"fit object {name} was already sealed differently")
        self._sealed[name] = seal
        return seal

    def freeze(self) -> None:
        missing = sorted(set(self.REQUIRED_FITS) - set(self._sealed))
        if missing:
            raise RuntimeError(f"all clean fit objects must be sealed before attack loading: {missing}")
        self._frozen = True

    def load_attack_labels(self, onsets: Mapping[str, float]) -> dict[str, float]:
        if not self._frozen:
            raise RuntimeError("attack loader unavailable until all clean fits are sealed")
        expected = set(core.ATTACK_SCENARIOS)
        if set(onsets) not in (expected, {"DS1"}):
            raise ValueError("attack onset set must be the frozen scenario set")
        result = {str(k): float(v) for k, v in onsets.items()}
        if any(not np.isfinite(x) for x in result.values()):
            raise ValueError("attack onsets must be finite")
        self._attack_loaded = True
        return result

    @property
    def audit(self) -> dict[str, object]:
        return {"state": "attack_labels_loaded" if self._attack_loaded else
                ("fit_frozen" if self._frozen else "clean_fit"),
                "sealed_fit_digests": dict(sorted(self._sealed.items())),
                "attack_fit": False, "scenario_id_in_fit": False,
                "onset_in_fit": False, "prn_id_in_conditioner": False}


def _window_starts(times: np.ndarray, window_s: float, stride_s: float) -> list[float]:
    """Bit-for-contract reproduction of tracking_feature_windows._window_starts."""
    if window_s <= 0 or stride_s <= 0:
        raise ValueError("window and stride must be positive")
    if not len(times):
        return []
    diffs = np.diff(times)
    step = float(np.median(diffs[diffs > 0])) if len(times) > 1 and np.any(diffs > 0) else 0.0
    end = float(times[-1]) + step
    start = float(times[0])
    out: list[float] = []
    while start <= end - window_s + 1e-9:
        out.append(round(start, 10))
        start += stride_s
    return out


def _cadence_runs(times: np.ndarray) -> list[slice]:
    if not len(times):
        return []
    positive = np.diff(times)
    positive = positive[positive > 0]
    cadence = float(np.median(positive)) if len(positive) else 0.0
    cuts = [0]
    if cadence > 0:
        cuts.extend((np.flatnonzero(np.diff(times) > cadence * 1.5 + 1e-9) + 1).tolist())
    cuts.append(len(times))
    return [slice(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def build_node_windows(npz_path: str | Path, scenario: str, *, recording_id: str | None = None,
                       window_s: float = 1.0, stride_s: float = .5,
                       min_epochs: int = 4, expected_sha256: str | None = None
                       ) -> tuple[list[NodeWindow], dict[str, object]]:
    """Convert canonical complex tracking epochs to B0 prompt-relative nodes.

    Sorting is stable and explicit.  Windows cannot cross producer segment,
    channel, PRN, or an observed cadence gap.  Ratios are formed at each raw
    epoch before the window mean, exactly matching the frozen B0 feature family.
    """
    path = Path(npz_path)
    actual_hash = sha256_file(path)
    if expected_sha256 and actual_hash != expected_sha256:
        raise ValueError(f"canonical NPZ SHA256 mismatch for {scenario}")
    with np.load(path, allow_pickle=False) as archive:
        missing = [x for x in NPZ_FIELDS if x not in archive.files]
        if missing:
            raise ValueError(f"canonical NPZ missing arrays: {missing}")
        arrays = {name: np.asarray(archive[name]) for name in NPZ_FIELDS}
    n = len(arrays["time_s"])
    if any(len(value) != n for value in arrays.values()):
        raise ValueError("canonical NPZ arrays have unequal row count")
    iq = arrays["complex_iq"]
    if iq.shape != (n, 9, 2) or not np.issubdtype(iq.dtype, np.floating):
        raise ValueError("complex_iq must have schema (N,9,2) floating")
    for name in ("time_s", "prn", "channel", "segment_index", "sample_count"):
        if np.asarray(arrays[name]).ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
    times = arrays["time_s"].astype(np.float64)
    if not np.isfinite(iq).all() or not np.isfinite(times).all():
        raise ValueError("canonical NPZ values must be finite")
    original = np.arange(n, dtype=np.int64)
    order = np.lexsort((original, arrays["sample_count"], times, arrays["prn"],
                        arrays["channel"], arrays["segment_index"]))
    rec = recording_id or scenario
    nodes: list[NodeWindow] = []
    grouped: dict[tuple[int, int, int], list[int]] = {}
    for index in order:
        key = (int(arrays["segment_index"][index]), int(arrays["channel"][index]),
               int(arrays["prn"][index]))
        grouped.setdefault(key, []).append(int(index))
    gap_splits = 0
    for (segment, channel, prn), indices in sorted(grouped.items()):
        ix = np.asarray(indices, dtype=np.int64)
        group_times = times[ix]
        runs = _cadence_runs(group_times)
        gap_splits += max(0, len(runs) - 1)
        window_index = 0
        for run in runs:
            run_ix = ix[run]
            run_times = times[run_ix]
            magnitude = np.hypot(iq[run_ix, :, 0].astype(np.float64),
                                 iq[run_ix, :, 1].astype(np.float64))
            ratios = magnitude / (magnitude[:, 4, None] + EPS)
            for start in _window_starts(run_times, window_s, stride_s):
                end = start + window_s
                mask = (run_times >= start - 1e-9) & (run_times <= end + 1e-9)
                count = int(mask.sum())
                if count < min_epochs:
                    window_index += 1
                    continue
                support = run_ix[mask]
                midpoint = float((start + end) / 2)
                window_bin = round(round(midpoint / stride_s) * stride_s, 10)
                nodes.append(NodeWindow(
                    rec, str(scenario), f"G{prn:02d}", channel, segment, window_index,
                    float(start), float(end), midpoint, window_bin, count,
                    int(np.min(arrays["sample_count"][support])),
                    int(np.max(arrays["sample_count"][support])),
                    np.mean(ratios[mask], axis=0)))
                window_index += 1
    nodes.sort(key=lambda x: (x.recording_id, x.segment_index, x.channel, x.prn,
                              x.window_bin_s, x.window_index))
    audit = {
        "path": str(path), "sha256_recomputed": actual_hash, "rows": n,
        "node_count": len(nodes), "group_count": len(grouped), "cadence_gap_splits": gap_splits,
        "stable_sort": list(SORT_FIELDS), "tap_feature_order": list(TAP_NAMES),
        "tap_coordinates_chips": list(TAP_COORDS), "normalization":
        "mean_epoch(abs(complex_iq_tap)/(abs(complex_iq_prompt)+1e-6))",
        "window_seconds": float(window_s), "stride_seconds": float(stride_s),
        "minimum_raw_epochs": int(min_epochs), "endpoint_tolerance_seconds": 1e-9,
    }
    return nodes, audit


def build_sequence_examples(nodes: Sequence[NodeWindow], *, sequence_length: int = 12,
                            stride_s: float = .5) -> tuple[list[SequenceExample], dict[str, object]]:
    if sequence_length < 1 or stride_s <= 0:
        raise ValueError("sequence contract invalid")
    groups: dict[tuple[str, int, int, str], list[NodeWindow]] = {}
    for node in nodes:
        groups.setdefault(node.group_key, []).append(node)
    examples: list[SequenceExample] = []
    rejected = 0
    for key in sorted(groups):
        values = sorted(groups[key], key=lambda x: (x.window_bin_s, x.window_index))
        for target_pos in range(sequence_length, len(values)):
            seq = values[target_pos-sequence_length:target_pos+1]
            bins = np.asarray([x.window_bin_s for x in seq])
            indices = np.asarray([x.window_index for x in seq])
            contiguous = (np.allclose(np.diff(bins), stride_s, rtol=0, atol=1e-8)
                          and np.all(np.diff(indices) == 1))
            if not contiguous:
                rejected += 1
                continue
            examples.append(SequenceExample(tuple(seq[:-1]), seq[-1]))
    return examples, {"sequence_length": sequence_length, "stride_seconds": stride_s,
                      "valid_examples": len(examples), "rejected_noncontiguous": rejected,
                      "cross_segment_examples": 0}


class _FrozenGRUModel:
    @staticmethod
    def construct(torch, feature_dim: int, cfg: Mapping[str, object]):
        nn = torch.nn
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                emb = int(cfg["emb_dim"]); hidden = int(cfg["hidden_dim"])
                self.encoder = nn.Sequential(nn.Linear(feature_dim, emb), nn.LayerNorm(emb),
                    nn.GELU(), nn.Dropout(float(cfg["dropout"])), nn.Linear(emb, emb), nn.GELU())
                self.gru = nn.GRU(input_size=emb, hidden_size=hidden, batch_first=True)
                self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, feature_dim))
            def forward(self, x):
                b, t, f = x.shape
                z = self.encoder(x.reshape(b*t, f)).reshape(b, t, -1)
                out, _ = self.gru(z)
                return self.head(out[:, -1])
        return Model()


class FrozenB0:
    EXPECTED_FEATURE_COLUMNS = tuple(f"tap_{x}_rel_prompt_mean" for x in TAP_NAMES)

    def __init__(self, checkpoint_path: str | Path, config: Mapping[str, object], *,
                 device: str | None = None):
        import torch
        path = Path(checkpoint_path)
        expected = str(config["checkpoint_sha256"])
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError("frozen B0 checkpoint hash mismatch")
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(f"safe checkpoint load failed: {exc}") from exc
        required = {"config", "node_feature_columns", "standardizer", "model_state_dict"}
        if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
            raise ValueError("frozen B0 checkpoint schema is incomplete")
        features = tuple(checkpoint["node_feature_columns"])
        if features != self.EXPECTED_FEATURE_COLUMNS or tuple(config["feature_order"]) != TAP_NAMES:
            raise ValueError("frozen B0 feature schema/order mismatch or PRN feature present")
        cfg = checkpoint["config"]
        if int(cfg.get("seq_len", -1)) != int(config["sequence_length"]):
            raise ValueError("frozen B0 sequence length mismatch")
        standardizer = checkpoint["standardizer"]
        self.mean = np.asarray(standardizer["node_mean"], dtype=np.float32)
        self.std = np.asarray(standardizer["node_std"], dtype=np.float32)
        if self.mean.shape != (9,) or self.std.shape != (9,) or np.any(self.std <= 0) or not (
                np.isfinite(self.mean).all() and np.isfinite(self.std).all()):
            raise ValueError("frozen B0 float32 standardizer is invalid")
        self.feature_order = TAP_NAMES
        self.sequence_length = int(cfg["seq_len"])
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("requested CUDA but CUDA is unavailable")
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        self.model = _FrozenGRUModel.construct(torch, 9, cfg).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.training_calls = 0
        self.checkpoint_sha256 = actual
        self._torch = torch

    def train(self, *args, **kwargs):
        self.training_calls += 1
        raise RuntimeError("frozen B0 cannot train")

    def standardize(self, values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=np.float32)
        if x.shape[-1] != 9 or not np.isfinite(x).all():
            raise ValueError("B0 input must be finite (...,9)")
        return ((x - self.mean) / self.std).astype(np.float32)

    def predict(self, raw_sequences: np.ndarray, *, batch_size: int | None = None) -> np.ndarray:
        x = np.asarray(raw_sequences, dtype=np.float32)
        if x.ndim != 3 or x.shape[1:] != (self.sequence_length, 9):
            raise ValueError(f"B0 input must be (N,{self.sequence_length},9)")
        z = self.standardize(x)
        # Full-scenario batch is the production contract; chunking is an explicit smoke aid.
        size = len(z) if batch_size is None else int(batch_size)
        output = []
        self.model.eval()
        with self._torch.inference_mode():
            for offset in range(0, len(z), max(1, size)):
                tensor = self._torch.from_numpy(z[offset:offset+size]).to(self.device)
                output.append(self.model(tensor).cpu().numpy().astype(np.float32))
        return np.concatenate(output, axis=0) if output else np.empty((0, 9), np.float32)


def make_peak_pairs(targets_raw: np.ndarray, predicted_standardized: np.ndarray,
                    b0: FrozenB0, identities: Sequence[core.EpochIdentity]
                    ) -> list[core.PeakPredictionPair]:
    targets = np.asarray(targets_raw, dtype=np.float64)
    predictions_z = np.asarray(predicted_standardized, dtype=np.float64)
    if targets.shape != predictions_z.shape or targets.shape != (len(identities), 9):
        raise ValueError("B0 target/prediction/identity alignment mismatch")
    predicted_raw = predictions_z * b0.std.astype(np.float64) + b0.mean.astype(np.float64)
    pairs = []
    for actual, predicted, ident in zip(targets, predicted_raw, identities):
        pairs.append(core.PeakPredictionPair(
            actual_raw=actual, predicted_raw=predicted,
            residual_standardized=(actual-predicted)/b0.std.astype(np.float64),
            standardizer_std=b0.std.astype(np.float64), identity=ident,
            actual_space=core.RAW_SPACE, predicted_space=core.RAW_SPACE,
            residual_space=core.STANDARDIZED_SPACE, coordinates=core.CANONICAL_TAP_COORDS))
    return pairs


def infer_b0_pairs(examples: Sequence[SequenceExample], b0: FrozenB0):
    histories = np.asarray([[node.actual_raw for node in item.history] for item in examples],
                           dtype=np.float32)
    targets = np.asarray([item.target.actual_raw for item in examples], dtype=np.float64)
    predicted_z = b0.predict(histories)
    identities = [item.target.epoch_identity for item in examples]
    pairs = make_peak_pairs(targets, predicted_z, b0, identities)
    rmse = np.asarray([math.sqrt(float(np.mean(x.residual_standardized**2))) for x in pairs])
    return pairs, rmse, {"pairs": len(pairs), "full_scenario_batch": True,
                         "availability": "target_source_end", "PRN_ID_input": False,
                         "prediction_raw_inverse": "predicted_z*checkpoint_std+checkpoint_mean"}


def assert_same_epoch_mask(method_pairs: Mapping[str, Sequence[object]]) -> tuple[object, ...]:
    expected = None
    for method, rows in method_pairs.items():
        keys = tuple(getattr(x, "identity", x) for x in rows)
        if len(keys) != len(set(keys)):
            raise ValueError(f"{method} has duplicate epoch identities")
        if expected is None:
            expected = keys
        elif keys != expected:
            raise ValueError("all methods must use the same exact ordered epoch mask")
    return expected or ()


def iq_feature_vector(iq: np.ndarray, *, spectral_samples: int = 65536,
                      epsilon: float = 1e-12) -> np.ndarray:
    x = np.asarray(iq)
    if x.ndim != 2 or x.shape[1] != 2 or len(x) < 2:
        raise ValueError("IQ feature block must be (N,2) with at least two samples")
    z = x[:, 0].astype(np.float64) + 1j*x[:, 1].astype(np.float64)
    power = np.mean(np.abs(z)**2)
    diff = np.abs(np.diff(z))
    noise = np.median(np.abs(diff - np.median(diff))) / 0.67448975
    spectrum_input = z[:min(int(spectral_samples), len(z))]
    spectrum = np.abs(np.fft.fft(spectrum_input))**2 / max(1, len(spectrum_input))
    flatness = math.exp(float(np.mean(np.log(spectrum + epsilon)))) / float(np.mean(spectrum + epsilon))
    lag = abs(np.vdot(z[:-1], z[1:])) / math.sqrt(
        float(np.vdot(z[:-1], z[:-1]).real * np.vdot(z[1:], z[1:]).real) + epsilon)
    return np.asarray([math.log(power + epsilon), math.log(noise + epsilon),
                       flatness, lag], dtype=np.float64)


def extract_iq_blocks(raw_path: str | Path, recording_id: str, *, sample_rate_hz: int = 25_000_000,
                      block_duration_s: float = .01, block_stride_s: float = .5,
                      max_end_s: float | None = None) -> list[IQBlock]:
    path = Path(raw_path)
    if path.stat().st_size % 4:
        raise ValueError("raw int16 interleaved IQ size must be divisible by four")
    samples = path.stat().st_size // 4
    block = int(round(sample_rate_hz*block_duration_s))
    stride = int(round(sample_rate_hz*block_stride_s))
    if block < 2 or stride < 1:
        raise ValueError("IQ block geometry invalid")
    mem = np.memmap(path, mode="r", dtype="<i2", shape=(samples, 2))
    limit = samples if max_end_s is None else min(samples, int(math.floor(max_end_s*sample_rate_hz)))
    result = []
    for offset in range(0, max(0, limit-block+1), stride):
        end = offset+block
        result.append(IQBlock(str(recording_id), offset/sample_rate_hz, end/sample_rate_hz,
                              offset, block, iq_feature_vector(mem[offset:end])))
    del mem
    return result


def raw_iq_witness(path: str | Path, witness_bytes: int = 1 << 20) -> dict[str, object]:
    p = Path(path); size = p.stat().st_size
    with p.open("rb") as stream:
        first = stream.read(min(witness_bytes, size))
        stream.seek(max(0, size-witness_bytes)); last = stream.read(min(witness_bytes, size))
    return {"path": str(p), "size_bytes": size, "full_sha256_recomputed": False,
            "witness_bytes": witness_bytes,
            "first_1MiB_sha256": hashlib.sha256(first).hexdigest(),
            "last_1MiB_sha256": hashlib.sha256(last).hexdigest()}


def causal_iq_context(target_source_start: Sequence[float], target_groups: Sequence[str],
                      blocks: Sequence[IQBlock], *, history: int = 4, cadence: float = .5
                      ) -> tuple[np.ndarray, dict[str, object]]:
    if not blocks:
        raise ValueError("IQ history blocks are unavailable")
    ordered = sorted(blocks, key=lambda x: (x.recording_id, x.end_s))
    context = core.build_causal_iq_context(
        target_source_start, [x.end_s for x in ordered], np.stack([x.features for x in ordered]),
        history=history, target_groups=target_groups,
        block_groups=[x.recording_id for x in ordered], cadence=cadence)
    if not np.all(context.valid):
        bad = np.flatnonzero(~context.valid).tolist()
        raise ValueError(f"insufficient contiguous strictly causal IQ history for rows {bad[:8]}")
    selected_ends = [[float(ordered[i].end_s) for i in ix] for ix in context.block_indices]
    targets = np.asarray(target_source_start, float)
    if any(any(end > targets[row]+1e-12 for end in values) for row, values in enumerate(selected_ends)):
        raise RuntimeError("causal IQ contract violated")
    # Four blocks describe one event context; temporal mean preserves the exact four-feature schema.
    predictors = np.mean(context.contexts, axis=1)
    audit = dict(context.audit)
    audit.update({"strict_causal": True, "asof_operator": "block_end <= min_target_source_start",
                  "history_blocks": history, "history_reducer": "arithmetic_mean_per_feature",
                  "selected_block_ends": selected_ends, "feature_schema": list(IQ_FEATURE_NAMES)})
    return predictors, audit


def aggregate_event(prn_scores: Mapping[str, float] | Sequence[float], aggregator: str) -> float:
    values = np.asarray(list(prn_scores.values()) if isinstance(prn_scores, Mapping) else prn_scores,
                        dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("aggregation requires finite nonempty PRN scores")
    if aggregator == "median":
        return float(np.median(values))
    if aggregator == "top25_mean":
        count = int(math.ceil(.25*len(values)))
        return float(np.mean(np.sort(values)[-count:]))
    raise ValueError("aggregator must be median or top25_mean")


def calibrate_all_thresholds(calibration: Mapping[str, Mapping[str, Sequence[float]]],
                             quantiles: Sequence[float] = (.99, .995)) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for method in sorted(calibration):
        for aggregator in sorted(calibration[method]):
            values = np.asarray(calibration[method][aggregator], dtype=np.float64)
            if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
                raise ValueError(f"empty/nonfinite calibration for {method}/{aggregator}")
            digest = canonical_digest(values)
            for q in quantiles:
                suffix = f"q{int(round(q*1000))}" if q != .99 else "q99"
                result[f"{method}/{aggregator}/{suffix}"] = {
                    "detector": method, "aggregator": aggregator, "quantile": float(q),
                    "method": "higher", "comparison": "strict >",
                    "value": float(np.quantile(values, q, method="higher")),
                    "rows": len(values), "score_digest_sha256": digest,
                    "clean_scenario": "cleanStatic", "clean_role": "normal_calibration",
                }
    return result


def source_role(node: NodeWindow) -> str:
    if node.scenario != "cleanStatic":
        return "evaluation"
    if node.source_end_s <= 300:
        return "normal_train"
    if node.source_start_s >= 320 and node.source_end_s <= 400:
        return "normal_calibration"
    if node.source_start_s >= 420:
        return "normal_holdout"
    return "excluded_boundary_crossing"


def event_phase(scenario: str, source_start: float, source_end: float,
                onsets: Mapping[str, float] | None) -> tuple[str, object]:
    if scenario == "cleanStatic" or scenario == "cleanDynamic":
        return "normal", ""
    if onsets is None:
        return "labels_not_loaded", ""
    onset = float(onsets[scenario])
    if source_start >= onset:
        return "post", 1
    if source_start >= 30 and source_end <= onset-20:
        return "stable_pre", 0
    return "transition_excluded", ""


class ArtifactStage:
    """PID-scoped no-overwrite stage with verifier-gated atomic publication."""
    def __init__(self, final_path: str | Path):
        self.final = Path(final_path)
        self.path = self.final.parent / f".{self.final.name}.tmp.{os.getpid()}"
        self._published = False

    def __enter__(self):
        if self.final.exists():
            raise FileExistsError(f"artifact output already exists: {self.final}")
        if self.path.exists():
            raise FileExistsError(f"artifact stage already exists: {self.path}")
        self.path.mkdir(parents=True)
        return self

    def publish(self, verifier: Callable[[Path], Mapping[str, object]]) -> None:
        if self._published:
            raise RuntimeError("artifact already published")
        result = verifier(self.path)
        if not result.get("ok"):
            raise RuntimeError(f"independent artifact verifier failed: {result.get('errors')}")
        if self.final.exists():
            raise FileExistsError(f"artifact output appeared before publication: {self.final}")
        os.replace(self.path, self.final)
        self._published = True

    def __exit__(self, kind, value, traceback):
        if kind is not None and self.path.exists():
            marker = {"schema": "gnss-doppler-lab.nc-topi-stage0.failed.v1",
                      "exception_type": kind.__name__, "message": str(value),
                      "unix_time": time.time(), "published": False}
            (self.path/"FAILED.json").write_text(json.dumps(marker, sort_keys=True, indent=2)+"\n")
        return False


def _seed_everything(seed: int = core.DEFAULT_SEED):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def smoke(npz_path: Path, scenario: str, checkpoint: Path, config_path: Path,
          device: str | None = None) -> dict[str, object]:
    """Small real/synthetic lineage smoke; never loads attack labels or fits."""
    started = time.monotonic(); config = core.load_config(config_path)
    nodes, node_audit = build_node_windows(npz_path, scenario, window_s=1., stride_s=.5,
                                            min_epochs=4)
    examples, sequence_audit = build_sequence_examples(nodes, sequence_length=12, stride_s=.5)
    if not examples:
        raise RuntimeError("smoke input produced no contiguous B0 sequence examples")
    b0 = FrozenB0(checkpoint, config["b0"], device=device)
    # Keep a smoke bounded while preserving one deterministic batch.
    subset = examples[:min(256, len(examples))]
    pairs, rmse, inference_audit = infer_b0_pairs(subset, b0)
    return {"schema": "gnss-doppler-lab.nc-topi-stage0.smoke.v1", "scenario": scenario,
            "node_windows": len(nodes), "sequence_examples": len(examples),
            "inference_pairs": len(pairs), "b0_rmse_mean": float(np.mean(rmse)),
            "b0_rmse_max": float(np.max(rmse)), "elapsed_seconds": time.monotonic()-started,
            "node_audit": node_audit, "sequence_audit": sequence_audit,
            "inference_audit": inference_audit, "attack_labels_loaded": False,
            "attack_scores": 0, "device": str(b0.device),
            "checkpoint_sha256": b0.checkpoint_sha256}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/nc_topi_stage0.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT/"artifacts/ai_morph_gru_cleanStatic_q70_frame/prn_local_gru_predictor.pt")
    parser.add_argument("--smoke-npz", type=Path, help="run bounded no-fit/no-label lineage smoke")
    parser.add_argument("--scenario", default="cleanStatic")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--json-output", type=Path, help="optional smoke report under /tmp")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.smoke_npz:
        raise SystemExit("full campaign is intentionally explicit and not launched by default; supply --smoke-npz")
    if args.json_output and not str(args.json_output).startswith("/tmp/"):
        raise SystemExit("smoke output must be under /tmp; final artifact paths are verifier-gated")
    _seed_everything()
    report = smoke(args.smoke_npz, args.scenario, args.checkpoint, args.config, args.device)
    encoded = json.dumps(report, sort_keys=True, indent=2)+"\n"
    if args.json_output:
        args.json_output.write_text(encoded)
    print(encoded, end="")
    return 0



# External-control and legacy lineage diagnostics are import APIs; neither can fit.
def load_external_normal_nodes(csv_path: str | Path, *,
        expected_sha256: str="48fc7d2e4f527917fa1f6c2286562a60c3fff2c09d4ffb54cd215a2a33f2bf49"):
    """Load frozen cleanDynamic nodes or explicitly mark exact schema unavailable."""
    path=Path(csv_path)
    audit={"path":str(path),"expected_sha256":expected_sha256,
           "evaluation_only_never_fit":True,"substitution":False}
    try:
        actual=sha256_file(path)
        if actual!=expected_sha256: raise ValueError("cleanDynamic node CSV SHA256 mismatch")
        with path.open(encoding="utf-8",newline="") as stream: rows=list(csv.DictReader(stream))
        columns=tuple(f"tap_{name}_rel_prompt_mean" for name in TAP_NAMES)
        required={"run_id","source_fingerprint","prn","channel","segment_index","window_index",
                  "window_bin_s","window_start_s","window_end_s","window_mid_s","epoch_count",
                  "tap_count","tap_layout",*columns}
        if not rows or not required.issubset(rows[0]): raise ValueError("cleanDynamic exact node schema failed")
        nodes=[]
        for row in rows:
            if int(float(row["tap_count"]))!=9 or row["tap_layout"]!=",".join(TAP_NAMES):
                raise ValueError("cleanDynamic tap layout is not exact canonical 9-tap order")
            nodes.append(NodeWindow(str(row["run_id"]),"cleanDynamic",str(row["prn"]),
                int(float(row["channel"])),int(float(row["segment_index"])),int(float(row["window_index"])),
                float(row["window_start_s"]),float(row["window_end_s"]),float(row["window_mid_s"]),
                float(row["window_bin_s"]),int(float(row["epoch_count"])),-1,-1,
                np.asarray([float(row[column]) for column in columns],dtype=np.float64)))
        audit.update({"available":True,"sha256_recomputed":actual,"rows":len(nodes),
                      "feature_columns":list(columns),"source_sample_support":
                      "not present in frozen matched node CSV; exact source time support retained"})
        return nodes,audit
    except Exception as exc:
        audit.update({"available":False,"reason":str(exc),"rows":0})
        return [],audit


def legacy_b0_diagnostic(regenerated_rows: Sequence[Mapping[str,object]],legacy_csv: str|Path,*,
        regenerated_lineage_digest: str,legacy_lineage_digest: str|None=None)->dict[str,object]:
    """Bit-exact DS7/DS8 diagnostic only when regenerated lineage is identical."""
    if not legacy_lineage_digest or regenerated_lineage_digest!=legacy_lineage_digest:
        return {"comparable":False,"reason":"regenerated lineage does not exactly match legacy lineage","primary_use":False}
    with Path(legacy_csv).open(encoding="utf-8",newline="") as stream: legacy=list(csv.DictReader(stream))
    keys=("run_id","prn","window_bin_s","target_window_index");score="prn_node_rmse"
    if not legacy or any(field not in legacy[0] for field in (*keys,score)):
        return {"comparable":False,"reason":"legacy score schema mismatch","primary_use":False}
    old={tuple(str(row[field]) for field in keys):np.float32(row[score]).tobytes() for row in legacy}
    try:new={tuple(str(row[field]) for field in keys):np.float32(row[score]).tobytes() for row in regenerated_rows}
    except KeyError:return {"comparable":False,"reason":"regenerated key schema mismatch","primary_use":False}
    equal=set(old)==set(new)
    return {"comparable":equal,"key_sets_equal":equal,"legacy_rows":len(old),"regenerated_rows":len(new),
            "covered_keys":len(set(old).intersection(new)),"bit_exact_scores":bool(equal and all(old[k]==new[k] for k in old)),
            "primary_use":False}


def _write_csv_rows(path: Path, rows: Sequence[Mapping[str,object]]) -> None:
    if not rows:
        path.write_text("scenario,method,aggregator,value\n")
        return
    fields=[]
    for row in rows:
        for field in row:
            if field not in fields: fields.append(field)
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n")
        writer.writeheader();writer.writerows(rows)


def render_stage0_plots(per_epoch_rows: Sequence[Mapping[str,object]],synthetic: Mapping[str,object],
                        output_dir: str|Path)->list[str]:
    """Render the preregistered deterministic PNG inventory using Agg only."""
    import matplotlib
    matplotlib.use("Agg",force=True)
    import matplotlib.pyplot as plt
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    scenarios=("cleanStatic","cleanDynamic","DS1","DS2","DS3","DS7","DS8")
    event=[row for row in per_epoch_rows if row.get("row_level")=="event"]
    prn=[row for row in per_epoch_rows if row.get("row_level")=="prn"]
    made=[]
    def save(name):
        path=out/name;plt.tight_layout();plt.savefig(path,dpi=120,metadata={"Software":"gnss-doppler-lab"});plt.close();made.append(name)
    for scenario in scenarios:
        rows=sorted((x for x in event if x.get("scenario")==scenario),key=lambda x:float(x["availability_time_s"]))
        plt.figure(figsize=(8,3.2))
        for method in ("B0","TOPI","NC_TOPI"):
            column=f"{method}_median"
            if rows and column in rows[0]: plt.plot([float(x["availability_time_s"]) for x in rows],[float(x[column]) for x in rows],label=method,lw=.8)
        plt.xlabel("availability source_end (s)");plt.ylabel("median score");plt.title(f"{scenario}: B0 / TOPI / NC-TOPI");plt.legend(loc="best")
        save(f"score_comparison_{scenario}.png")
        groups=sorted({str(x.get("prn","")) for x in prn if x.get("scenario")==scenario})
        times=sorted({float(x["availability_time_s"]) for x in prn if x.get("scenario")==scenario})
        matrix=np.full((max(1,len(groups)),max(1,len(times))),np.nan)
        gi={x:i for i,x in enumerate(groups)};ti={x:i for i,x in enumerate(times)}
        for row in prn:
            if row.get("scenario")==scenario and row.get("TOPI","")!="": matrix[gi[str(row["prn"])],ti[float(row["availability_time_s"])]]=float(row["TOPI"])
        plt.figure(figsize=(8,3.2));plt.imshow(matrix,aspect="auto",interpolation="nearest",origin="lower");plt.colorbar(label="PRN orthogonal TOPI");plt.xlabel("event index");plt.ylabel("PRN index");plt.title(scenario)
        save(f"prn_orth_heatmap_{scenario}.png")
    plt.figure(figsize=(5,4))
    if prn:
        tangent=np.asarray([float(x.get("tangent_energy",0)) for x in prn]);orth=np.asarray([float(x.get("perp_energy",x.get("TOPI",0))) for x in prn]);plt.scatter(tangent,orth,s=2,alpha=.4)
    plt.xlabel("tangent energy");plt.ylabel("orthogonal energy");save("tangent_orth_energy.png")
    plt.figure(figsize=(6,3.5))
    for method in ("B0_median","TOPI_median","NC_TOPI_median"):
        values=[float(x[method]) for x in event if x.get("scenario") in ("cleanStatic","cleanDynamic") and x.get(method,"")!=""]
        if values: plt.hist(values,bins=40,histtype="step",label=method)
    plt.legend(loc="best");plt.xlabel("clean score");save("clean_distributions.png")
    for name,xlimit in (("roc.png",(0,1)),("roc_low_fpr.png",(0,.05))):
        plt.figure(figsize=(4,4));plt.plot([0,1],[0,1],"k--",lw=.7);plt.xlim(*xlimit);plt.ylim(0,1);plt.xlabel("FPR");plt.ylabel("TPR");save(name)
    grid=synthetic.get("second_peak_grid",[]);powers=sorted({float(x["relative_power"]) for x in grid});seps=sorted({float(x["separation_chips"]) for x in grid})
    heat=np.full((max(1,len(powers)),max(1,len(seps))),np.nan);pi={x:i for i,x in enumerate(powers)};si={x:i for i,x in enumerate(seps)}
    for row in grid:heat[pi[float(row["relative_power"])],si[float(row["separation_chips"])]]=float(row["topi"])
    plt.figure(figsize=(5,4));plt.imshow(heat,aspect="auto",origin="lower",interpolation="nearest");plt.colorbar(label="TOPI");plt.xlabel("separation grid index");plt.ylabel("power grid index");save("second_peak_heatmap.png")
    return made


def publish_artifact(final_path: str|Path, *, config: Mapping[str,object],
        data_manifest: Mapping[str,object], thresholds: Mapping[str,object],
        per_epoch_rows: Sequence[Mapping[str,object]], scenario_metrics: Sequence[Mapping[str,object]],
        ablation_metrics: Sequence[Mapping[str,object]], synthetic: Mapping[str,object],
        bootstrap: Mapping[str,object], decision: Mapping[str,object], provenance: Mapping[str,object],
        fit_audit: Mapping[str,object], extras: Mapping[str,object]|None=None) -> Path:
    """Write, independently verify, then atomically publish one complete artifact."""
    import importlib.util
    allowed={"iq_context.csv","iq_context.npz","test_summary.json","model_lineage_audit.json"}
    extras=dict(extras or {})
    unknown=set(extras)-allowed
    if unknown: raise ValueError(f"unapproved artifact extras: {sorted(unknown)}")
    with ArtifactStage(final_path) as stage:
        payloads={"config.json":config,"data_manifest.json":data_manifest,"thresholds.json":thresholds,
                  "synthetic_physics_tests.json":synthetic,"bootstrap_results.json":bootstrap,
                  "decision.json":decision,"provenance.json":provenance,"fit_audit.json":fit_audit}
        for name,data in payloads.items():(stage.path/name).write_text(json.dumps(data,sort_keys=True,indent=2)+"\n")
        _write_csv_rows(stage.path/"per_epoch_scores.csv",per_epoch_rows)
        _write_csv_rows(stage.path/"scenario_metrics.csv",scenario_metrics)
        _write_csv_rows(stage.path/"ablation_metrics.csv",ablation_metrics)
        for name,data in extras.items():
            path=stage.path/name
            if isinstance(data,bytes):path.write_bytes(data)
            elif isinstance(data,str):path.write_text(data)
            else:path.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n")
        render_stage0_plots(per_epoch_rows,synthetic,stage.path/"plots")
        spec=importlib.util.spec_from_file_location("nc_topi_independent_verifier",ROOT/"scripts"/"summarize_nc_topi_stage0.py")
        verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
        preliminary=verifier.compute_summary(stage.path,verify_hashes=False,verify_source=True)
        preliminary["ok"]=True;preliminary["errors"]=[]
        (stage.path/"README.md").write_text(verifier.render_readme(preliminary))
        verifier.write_hash_inventory(stage.path)
        # README numeric hash count is known only after inventory construction.
        summary=verifier.compute_summary(stage.path,verify_hashes=True,verify_source=True)
        summary["ok"]=True;summary["errors"]=[]
        (stage.path/"README.md").write_text(verifier.render_readme(summary));verifier.write_hash_inventory(stage.path)
        stage.publish(lambda path:verifier.verify_artifact(path,verify_source=True))
    return Path(final_path)



def run_synthetic_physics(pairs: Sequence[core.PeakPredictionPair], covariance,
                          *, trials: int=100, seed: int=core.DEFAULT_SEED)->dict[str,object]:
    """Execute frozen threshold-free equal-RMSE, second-peak and nuisance physics grids."""
    if trials!=100: raise ValueError("Stage-0 equal-RMSE trial count is frozen at 100")
    if not pairs: raise ValueError("synthetic physics requires clean-train predicted peaks")
    rng=np.random.default_rng(seed);raw=[]
    for index in range(trials):
        pair=pairs[index%len(pairs)];basis=core.primary_tangent_basis(pair,core.CANONICAL_TAP_COORDS,covariance)
        tangent=basis.matrix@rng.normal(size=2)
        orth=core.w_orthogonal_vector(basis.matrix,covariance.W,seed=seed+index)
        tangent_b0=math.sqrt(float(np.mean((tangent/pair.standardizer_std)**2)))
        orth_b0=math.sqrt(float(np.mean((orth/pair.standardizer_std)**2)))
        if tangent_b0<=0 or orth_b0<=0: raise RuntimeError("degenerate synthetic perturbation")
        tangent=tangent/tangent_b0;orth=orth/orth_b0
        tb=math.sqrt(float(np.mean((tangent/pair.standardizer_std)**2)))
        ob=math.sqrt(float(np.mean((orth/pair.standardizer_std)**2)))
        tq=core.weighted_project(tangent,basis.matrix,covariance.W).perp_energy
        oq=core.weighted_project(orth,basis.matrix,covariance.W).perp_energy
        total=float(orth@covariance.W@orth)
        raw.append({"trial":index,"pair_identity":pair.identity.canonical_payload(),"b0_tangent":tb,
                    "b0_orthogonal":ob,"b0_relative_difference":abs(tb-ob)/max(tb,ob,1e-15),
                    "tangent_topi":tq,"orthogonal_topi":oq,
                    "tangent_to_orthogonal_topi_ratio":tq/max(oq,1e-15),
                    "orthogonal_preserved_fraction":oq/max(total,1e-15),
                    "tangent_raw":tangent.tolist(),"orthogonal_raw":orth.tolist()})
    reference=pairs[0];basis=core.primary_tangent_basis(reference,core.CANONICAL_TAP_COORDS,covariance)
    grid=[]
    for power in core.SECOND_PEAK_POWERS:
        for separation in core.SECOND_PEAK_SEPARATIONS:
            changed=core.second_peak_perturbation(reference.predicted_raw,core.CANONICAL_TAP_COORDS,power,separation)
            residual=changed-reference.predicted_raw
            projection=core.weighted_project(residual,basis.matrix,covariance.W)
            grid.append({"relative_power":power,"separation_chips":separation,
                         "b0":math.sqrt(float(np.mean((residual/reference.standardizer_std)**2))),
                         "topi":float(projection.perp_energy),"residual_raw":residual.tolist()})
    nuisance=[]
    for kind,values in (("amplitude",(.02,.05,.1)),("shift",(.01,.025,.05))):
        for signed in values:
            for sign in (-1,1):
                amount=sign*signed
                residual=(amount*reference.predicted_raw if kind=="amplitude" else
                    core.shift_peak(reference.predicted_raw,core.CANONICAL_TAP_COORDS,amount)-reference.predicted_raw)
                projection=core.weighted_project(residual,basis.matrix,covariance.W)
                nuisance.append({"kind":kind,"amount":amount,"noise_scale":1.,
                    "b0":math.sqrt(float(np.mean((residual/reference.standardizer_std)**2))),
                    "topi":float(projection.perp_energy),"residual_raw":residual.tolist()})
    for scale in (1.,1.25,1.5):
        residual=rng.normal(size=9)*reference.standardizer_std*scale
        projection=core.weighted_project(residual,basis.matrix,covariance.W)
        nuisance.append({"kind":"noise","amount":0.,"noise_scale":scale,
            "b0":math.sqrt(float(np.mean((residual/reference.standardizer_std)**2))),
            "topi":float(projection.perp_energy),"residual_raw":residual.tolist()})
    from scipy.stats import spearmanr
    power_pass=all(float(spearmanr([x["relative_power"] for x in grid if x["separation_chips"]==sep],
        [x["topi"] for x in grid if x["separation_chips"]==sep]).statistic)>=.8 for sep in (.25,.375,.5))
    separation_pass=all(float(spearmanr([x["separation_chips"] for x in grid if x["relative_power"]==power],
        [x["topi"] for x in grid if x["relative_power"]==power]).statistic)>=.8 for power in (.2,.4,.8))
    equal_pass=bool(max(x["b0_relative_difference"] for x in raw)<=1e-8 and
                    np.median([x["tangent_to_orthogonal_topi_ratio"] for x in raw])<=.05 and
                    np.median([x["orthogonal_preserved_fraction"] for x in raw])>=.95)
    return {"schema":"gnss-doppler-lab.nc-topi-stage0.synthetic.v1","seed":seed,
            "amplitude_semantics":"prompt-normalized shape-scale, not physical global gain",
            "attack_thresholds_used":False,"raw_trials":raw,"second_peak_grid":grid,
            "nuisance_grid":nuisance,"criteria":{"equal_rmse_pass":equal_pass,
            "second_peak_pass":bool(power_pass and separation_pass),"second_peak_power_pass":bool(power_pass),
            "second_peak_separation_pass":bool(separation_pass)},
            "summaries":{"equal_rmse_trials":len(raw),"second_peak_rows":len(grid),
            "nuisance_rows":len(nuisance)}}


if __name__ == "__main__":
    raise SystemExit(main())

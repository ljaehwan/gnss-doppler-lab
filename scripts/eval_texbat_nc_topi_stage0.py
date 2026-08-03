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
    "B0", "total", "amp_only", "shift_only", "amp_shift",
    "amp_shift_width", "TOPI", "NC_TOPI", "NC_TOPI_time_shuffle",
    "NC_TOPI_conditioning_removed",
)
PRIMARY_METHODS = ("B0", "TOPI", "NC_TOPI", "NC_TOPI_time_shuffle",
                   "NC_TOPI_conditioning_removed")
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
    target_index: int = -1


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
        self._sealed: dict[str, dict[str, object]] = {}
        self._frozen = False
        self._attack_loaded = False
        self._trace: list[str] = []

    def record_phase(self, phase: str) -> None:
        if not isinstance(phase,str) or not phase.strip(): raise ValueError("invalid phase")
        self._trace.append(phase)

    def seal_fit(self, name: str, fit_object: object,
                 row_identities: Iterable[core.EpochIdentity]) -> str:
        if self._frozen: raise RuntimeError("fit state is frozen")
        if name not in self.REQUIRED_FITS: raise ValueError(f"unknown fit object {name!r}")
        rows=tuple(row_identities)
        if not rows or any(not isinstance(x,core.EpochIdentity) or x.scenario!="cleanStatic" for x in rows):
            raise TypeError("fit identities must be ordered typed cleanStatic EpochIdentity objects")
        if len(rows)!=len(set(rows)): raise ValueError("fit identities must be unique")
        if name=="covariance":
            if not isinstance(fit_object,core.CovarianceFit): raise TypeError("actual CovarianceFit required")
            core._validate_covariance_fit(fit_object); object_seal=fit_object._construction_seal
        elif name in ("conditioner","conditioner_cap","shuffled_conditioner","shuffled_cap"):
            if not isinstance(fit_object,core.RobustConditioner): raise TypeError("actual RobustConditioner required")
            fit_object.validate_seal(require_calibrated=name.endswith("cap"))
            manifest=fit_object.cap_manifest_ if name.endswith("cap") else fit_object.fit_manifest_
            object_seal=hashlib.sha256(json.dumps(dict(manifest),default=str,sort_keys=True).encode()).hexdigest()
        else:
            if not isinstance(fit_object,Mapping) or not fit_object: raise TypeError("typed threshold mapping required")
            if any(not isinstance(x,core.ThresholdCalibration) for x in fit_object.values()):
                raise TypeError("every threshold must be a ThresholdCalibration")
            for x in fit_object.values(): core._validate_threshold(x,primary=False)
            object_seal=hashlib.sha256("".join(x._construction_seal for _,x in sorted(fit_object.items())).encode()).hexdigest()
        identity_seal=hashlib.sha256(json.dumps([x.canonical_payload() for x in rows],sort_keys=True,separators=(",",":")).encode()).hexdigest()
        seal=hashlib.sha256((object_seal+identity_seal).encode()).hexdigest()
        entry={"seal":seal,"object_seal":object_seal,"identity_digest_sha256":identity_seal,
               "identity_count":len(rows),"identities":[x.canonical_payload() for x in rows],
               "object_type":type(fit_object).__name__}
        if name in self._sealed and self._sealed[name] != entry:
            raise RuntimeError(f"fit object {name} was already sealed differently")
        self._sealed[name]=entry;self._trace.append(f"seal:{name}");return seal

    def freeze(self) -> None:
        missing = sorted(set(self.REQUIRED_FITS) - set(self._sealed))
        if missing:
            raise RuntimeError(f"all clean fit objects must be sealed before attack loading: {missing}")
        self._frozen = True
        self._trace.append("freeze_sealed")

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
        self._trace.append("attack_loader_called")
        return result

    @property
    def audit(self) -> dict[str, object]:
        return {"state": "attack_labels_loaded" if self._attack_loaded else
                ("fit_frozen" if self._frozen else "clean_fit"),
                "sealed_fits": dict(sorted(self._sealed.items())),
                "phase_trace":list(self._trace), "attack_loader_calls":int(self._attack_loaded),
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
    # Reproduce tracking_feature_windows + CSV + legacy scorer lineage exactly.
    path=Path(npz_path);actual_hash=sha256_file(path)
    if expected_sha256 and actual_hash!=expected_sha256:
        raise ValueError(f"canonical NPZ SHA256 mismatch for {scenario}")
    with np.load(path,allow_pickle=False) as archive:
        missing=[x for x in NPZ_FIELDS if x not in archive.files]
        if missing: raise ValueError(f"canonical NPZ missing arrays: {missing}")
        arrays={name:np.asarray(archive[name]) for name in NPZ_FIELDS}
    n=len(arrays["time_s"])
    if any(len(value)!=n for value in arrays.values()): raise ValueError("canonical NPZ arrays have unequal row count")
    iq=arrays["complex_iq"]
    if iq.shape!=(n,9,2) or not np.issubdtype(iq.dtype,np.floating):
        raise ValueError("complex_iq must have schema (N,9,2) floating")
    times=arrays["time_s"].astype(np.float64)
    if not np.isfinite(iq).all() or not np.isfinite(times).all(): raise ValueError("canonical NPZ values must be finite")
    original=np.arange(n,dtype=np.int64)
    order=np.lexsort((original,arrays["sample_count"],times,arrays["prn"],arrays["channel"],arrays["segment_index"]))
    grouped={}
    for index in order:
        key=(int(arrays["segment_index"][index]),int(arrays["channel"][index]),int(arrays["prn"][index]))
        grouped.setdefault(key,[]).append(int(index))
    rec=recording_id or scenario;nodes=[]
    for (segment,channel,prn),indices in sorted(grouped.items()):
        ix=np.asarray(indices,dtype=np.int64);group_times=times[ix]
        magnitude=np.hypot(iq[ix,:,0].astype(np.float64),iq[ix,:,1].astype(np.float64))
        ratios=magnitude/(magnitude[:,4,None]+EPS)
        for wi,start in enumerate(_window_starts(group_times,window_s,stride_s)):
            end=start+window_s;mask=(group_times>=start-1e-9)&(group_times<=end+1e-9)
            count=int(mask.sum())
            if count<min_epochs: continue
            support=ix[mask];midpoint=float((start+end)/2)
            window_bin=round(round(midpoint/stride_s)*stride_s,10)
            # The historical CSV producer formatted every feature at 12 significant digits;
            # pandas then converted it to float32 before checkpoint standardization.
            csv_roundtrip=np.asarray([float(format(float(x),".12g")) for x in np.mean(ratios[mask],axis=0)],dtype=np.float64)
            nodes.append(NodeWindow(rec,str(scenario),f"G{prn:02d}",channel,segment,wi,
                float(start),float(end),midpoint,window_bin,count,
                int(np.min(arrays["sample_count"][support])),int(np.max(arrays["sample_count"][support])),csv_roundtrip))
    nodes.sort(key=lambda x:(x.recording_id,x.prn,x.window_bin_s,x.segment_index,x.channel,x.window_index))
    audit={"path":str(path),"sha256_recomputed":actual_hash,"rows":n,"node_count":len(nodes),
      "group_count":len(grouped),"cadence_gap_splits":0,"cadence_run_prepartition":False,
      "legacy_node_grouping":["segment_index","channel","prn"],
      "legacy_scorer_grouping":["run_id","prn"],"stable_sort":list(SORT_FIELDS),
      "tap_feature_order":list(TAP_NAMES),"tap_coordinates_chips":list(TAP_COORDS),
      "normalization":"mean_epoch(abs(complex_iq_tap)/(abs(complex_iq_prompt)+1e-6))",
      "csv_roundtrip":"format(.12g)->parse float->float32 before standardization",
      "window_seconds":float(window_s),"stride_seconds":float(stride_s),
      "minimum_raw_epochs":int(min_epochs),"endpoint_tolerance_seconds":1e-9}
    return nodes,audit


def build_sequence_examples(nodes: Sequence[NodeWindow], *, sequence_length: int = 12,
                            stride_s: float = .5) -> tuple[list[SequenceExample], dict[str, object]]:
    if sequence_length<1 or stride_s<=0: raise ValueError("sequence contract invalid")
    # Frozen legacy scorer groups only by run_id/prn and sorts window_bin_s.  It does
    # not invent cadence runs or reject gaps; target_index is positional in this group.
    groups={}
    for node in nodes: groups.setdefault((node.recording_id,node.prn),[]).append(node)
    examples=[]
    for key in sorted(groups):
        values=sorted(groups[key],key=lambda x:x.window_bin_s)
        for start in range(0,len(values)-sequence_length):
            target_index=start+sequence_length
            examples.append(SequenceExample(tuple(values[start:target_index]),values[target_index],target_index))
    return examples,{"sequence_length":sequence_length,"stride_seconds":stride_s,
      "valid_examples":len(examples),"rejected_noncontiguous":0,"cross_segment_examples":
      sum(len({x.segment_index for x in e.history+(e.target,)})>1 for e in examples),
      "legacy_grouping":["run_id","prn"],"legacy_sort":"window_bin_s",
      "target_index":"zero-based positional index after per-run/per-PRN sort"}


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
    identities = [identity(item.target.recording_id,item.target.scenario,item.target.prn,
                           item.target_index,item.target.availability_time_s) for item in examples]
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/nc_topi_stage0.json")
    parser.add_argument("--checkpoint",type=Path,default=ROOT/"artifacts/ai_morph_gru_cleanStatic_q70_frame/prn_local_gru_predictor.pt")
    parser.add_argument("--out",type=Path,default=ROOT/"artifacts/nc_topi_stage0")
    parser.add_argument("--stop-after-freeze",action="store_true")
    parser.add_argument("--verify-full-raw-hash",action="store_true")
    parser.add_argument("--device",choices=("cpu","cuda"))
    parser.add_argument("--smoke-npz",type=Path,help="explicit bounded lineage diagnostic; never the default")
    parser.add_argument("--scenario",default="cleanStatic")
    parser.add_argument("--json-output",type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args=parse_args(argv);_seed_everything()
    if args.smoke_npz:
        if args.json_output and not str(args.json_output).startswith("/tmp/"):
            raise SystemExit("smoke output must be under /tmp")
        report=smoke(args.smoke_npz,args.scenario,args.checkpoint,args.config,args.device)
        encoded=json.dumps(report,sort_keys=True,indent=2)+"\n"
        if args.json_output: args.json_output.write_text(encoded)
        print(encoded,end="");return 0
    result=run_campaign(args.config,args.out,checkpoint=args.checkpoint,
      stop_after_freeze=args.stop_after_freeze,verify_full_raw_hash=args.verify_full_raw_hash,device=args.device)
    print(json.dumps({"ok":True,"output":str(result),"stop_after_freeze":args.stop_after_freeze},sort_keys=True))
    return 0



# External-control and legacy lineage diagnostics are import APIs; neither can fit.
def load_external_normal_nodes(csv_path: str | Path, *, scenario: str="cleanDynamic",
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
            nodes.append(NodeWindow(str(row["run_id"]),str(scenario),str(row["prn"]),
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


def legacy_b0_positive_control(node_csv: str|Path, score_csv: str|Path, b0: FrozenB0, *,
        node_sha256: str, score_sha256: str, gpu_atol: float=3e-4) -> dict[str,object]:
    if sha256_file(node_csv)!=node_sha256 or sha256_file(score_csv)!=score_sha256:
        raise ValueError("legacy positive-control input hash mismatch")
    nodes,audit=load_external_normal_nodes(node_csv,scenario="DS7",expected_sha256=node_sha256)
    if not audit.get("available"): raise ValueError(audit.get("reason"))
    examples,_=build_sequence_examples(nodes,sequence_length=b0.sequence_length,stride_s=.5)
    pairs,rmse,_=infer_b0_pairs(examples,b0)
    generated={(p.identity.recording_id,p.identity.prn,p.identity.target_index,
                format(e.target.window_bin_s,".12g")):float(value)
               for e,p,value in zip(examples,pairs,rmse)}
    with Path(score_csv).open(encoding="utf-8",newline="") as stream:stored=list(csv.DictReader(stream))
    expected={(x["run_id"],x["prn"],int(x["target_window_index"]),format(float(x["window_bin_s"]),".12g")):
              float(x["prn_node_rmse"]) for x in stored}
    common=set(generated)&set(expected);equal_keys=set(generated)==set(expected)
    errors=np.asarray([abs(generated[k]-expected[k]) for k in sorted(common)])
    tolerance=float(gpu_atol)
    matched=bool(equal_keys and len(common) and np.all(errors<=tolerance))
    return {"primary_use":False,"comparable":True,"positive_control_pass":matched,
      "node_sha256":node_sha256,"score_sha256":score_sha256,"legacy_rows":len(expected),
      "regenerated_rows":len(generated),"covered_keys":len(common),"key_sets_equal":equal_keys,
      "device":str(b0.device),"legacy_producer_device":"CUDA","defined_tolerance":tolerance,
      "tolerance_contract":"absolute RMSE tolerance for historical CUDA producer vs deterministic replay",
      "max_abs_rmse_error":float(errors.max()) if len(errors) else None,
      "lineage_note":"actual legacy node rows, not regenerated canonical NPZ epochs"}


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
    import matplotlib
    matplotlib.use("Agg",force=True)
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    scenarios=("cleanStatic","cleanDynamic","DS1","DS2","DS3","DS7","DS8")
    event=[row for row in per_epoch_rows if row.get("row_level")=="event"]
    prn=[row for row in per_epoch_rows if row.get("row_level")=="prn"]
    made=[];provenance={"schema":"gnss-doppler-lab.nc-topi-stage0.plot-provenance.v1","plots":{}}
    def save(name,series,points):
        if series<1 or points<1: raise RuntimeError(f"plot {name} has no real data series")
        path=out/name;plt.tight_layout();plt.savefig(path,dpi=120,metadata={"Software":"gnss-doppler-lab"});plt.close();made.append(name)
        provenance["plots"][name]={"data_series":int(series),"data_points":int(points),"source":"stored scores/labels or stored raw physics"}
    for scenario in scenarios:
        rows=sorted((x for x in event if x.get("scenario")==scenario),key=lambda x:float(x["availability_time_s"]))
        if not rows: raise RuntimeError(f"required scenario plot data absent: {scenario}")
        plt.figure(figsize=(8,3.2));series=0
        for method in ("B0","TOPI","NC_TOPI"):
            column=f"{method}_median";plt.plot([float(x["availability_time_s"]) for x in rows],[float(x[column]) for x in rows],label=method,lw=.8);series+=1
        plt.xlabel("availability source_end (s)");plt.ylabel("median score");plt.title(f"{scenario}: B0 / TOPI / NC-TOPI");plt.legend(loc="best")
        save(f"score_comparison_{scenario}.png",series,len(rows)*series)
        subset=[x for x in prn if x.get("scenario")==scenario];groups=sorted({str(x["prn"]) for x in subset});times=sorted({float(x["availability_time_s"]) for x in subset})
        if not groups or not times: raise RuntimeError(f"required PRN heatmap data absent: {scenario}")
        matrix=np.full((len(groups),len(times)),np.nan);gi={x:i for i,x in enumerate(groups)};ti={x:i for i,x in enumerate(times)}
        for row in subset:matrix[gi[str(row["prn"])]][ti[float(row["availability_time_s"])]]=float(row["TOPI"])
        finite=int(np.isfinite(matrix).sum());plt.figure(figsize=(8,3.2));plt.imshow(matrix,aspect="auto",interpolation="nearest",origin="lower");plt.colorbar(label="PRN orthogonal TOPI");plt.xlabel("event index");plt.ylabel("PRN index");plt.title(scenario)
        save(f"prn_orth_heatmap_{scenario}.png",1,finite)
    plt.figure(figsize=(5,4));tangent=np.asarray([float(x["amp_shift"]) for x in prn]);orth=np.asarray([float(x["TOPI"]) for x in prn]);plt.scatter(tangent,orth,s=2,alpha=.4);plt.xlabel("amp+shift tangent energy");plt.ylabel("orthogonal energy");save("tangent_orth_energy.png",1,len(prn))
    plt.figure(figsize=(6,3.5));series=0;points=0
    for method in ("B0_median","TOPI_median","NC_TOPI_median"):
        values=[float(x[method]) for x in event if x.get("scenario") in ("cleanStatic","cleanDynamic")]
        if values:plt.hist(values,bins=40,histtype="step",label=method);series+=1;points+=len(values)
    plt.legend(loc="best");plt.xlabel("clean score");save("clean_distributions.png",series,points)
    classified=[x for x in event if x.get("scenario") in core.ATTACK_SCENARIOS and str(x.get("label")) in ("0","1")]
    labels=np.asarray([int(x["label"]) for x in classified])
    if set(labels)!={0,1}: raise RuntimeError("ROC requires actual stored attack labels with both classes")
    for name,xlimit in (("roc.png",(0,1)),("roc_low_fpr.png",(0,.05))):
        plt.figure(figsize=(4,4));series=0;points=0
        for method in ("B0","TOPI","NC_TOPI"):
            fpr,tpr,_=roc_curve(labels,[float(x[f"{method}_median"]) for x in classified]);plt.plot(fpr,tpr,label=method);series+=1;points+=len(fpr)
        plt.xlim(*xlimit);plt.ylim(0,1);plt.xlabel("FPR");plt.ylabel("TPR");plt.legend(loc="best");save(name,series,points)
    grid=synthetic.get("second_peak_grid",[]);powers=sorted({float(x["relative_power"]) for x in grid});seps=sorted({float(x["separation_chips"]) for x in grid})
    if not powers or not seps: raise RuntimeError("second-peak plot data absent")
    heat=np.full((len(powers),len(seps)),np.nan);pi={x:i for i,x in enumerate(powers)};si={x:i for i,x in enumerate(seps)}
    for row in grid:heat[pi[float(row["relative_power"])]][si[float(row["separation_chips"])]]=float(row["topi"])
    plt.figure(figsize=(5,4));plt.imshow(heat,aspect="auto",origin="lower",interpolation="nearest");plt.colorbar(label="TOPI");plt.xlabel("separation grid index");plt.ylabel("power grid index");save("second_peak_heatmap.png",1,int(np.isfinite(heat).sum()))
    (out.parent/"plot_provenance.json").write_text(json.dumps(provenance,sort_keys=True,indent=2)+"\n")
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
                    "tangent_raw":tangent.tolist(),"orthogonal_raw":orth.tolist(),
                     "basis_matrix":basis.matrix.tolist(),"predicted_raw":pair.predicted_raw.tolist(),
                     "standardizer_std":pair.standardizer_std.tolist()})
    reference=pairs[0];basis=core.primary_tangent_basis(reference,core.CANONICAL_TAP_COORDS,covariance)
    grid=[]
    for power in core.SECOND_PEAK_POWERS:
        for separation in core.SECOND_PEAK_SEPARATIONS:
            changed=core.second_peak_perturbation(reference.predicted_raw,core.CANONICAL_TAP_COORDS,power,separation)
            residual=changed-reference.predicted_raw
            projection=core.weighted_project(residual,basis.matrix,covariance.W)
            grid.append({"relative_power":power,"separation_chips":separation,
                         "b0":math.sqrt(float(np.mean((residual/reference.standardizer_std)**2))),
                         "topi":float(projection.perp_energy),"residual_raw":residual.tolist(),
                          "changed_raw":changed.tolist()})
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
                    "topi":float(projection.perp_energy),"residual_raw":residual.tolist(),
                    "changed_raw":(reference.predicted_raw+residual).tolist()})
    for scale in (1.,1.25,1.5):
        residual=rng.normal(size=9)*reference.standardizer_std*scale
        projection=core.weighted_project(residual,basis.matrix,covariance.W)
        nuisance.append({"kind":"noise","amount":0.,"noise_scale":scale,
            "b0":math.sqrt(float(np.mean((residual/reference.standardizer_std)**2))),
            "topi":float(projection.perp_energy),"residual_raw":residual.tolist(),
                    "changed_raw":(reference.predicted_raw+residual).tolist()})
    from scipy.stats import spearmanr
    power_pass=all(float(spearmanr([x["relative_power"] for x in grid if x["separation_chips"]==sep],
        [x["topi"] for x in grid if x["separation_chips"]==sep]).statistic)>=.8 for sep in (.25,.375,.5))
    separation_pass=all(float(spearmanr([x["separation_chips"] for x in grid if x["relative_power"]==power],
        [x["topi"] for x in grid if x["relative_power"]==power]).statistic)>=.8 for power in (.2,.4,.8))
    equal_pass=bool(max(x["b0_relative_difference"] for x in raw)<=1e-8 and
                    np.median([x["tangent_to_orthogonal_topi_ratio"] for x in raw])<=.05 and
                    np.median([x["orthogonal_preserved_fraction"] for x in raw])>=.95)
    clean_b0=max(float(np.median([core.b0_rmse(x.residual_standardized) for x in pairs])),1e-12)
    clean_topi=max(float(np.median([core.produce_topi_scores(x,core.primary_tangent_basis(x,core.CANONICAL_TAP_COORDS,covariance),covariance).topi for x in pairs])),1e-12)
    for row in nuisance:
        row["b0_normalized"]=row["b0"]/clean_b0;row["topi_normalized"]=row["topi"]/clean_topi
    nuisance_kind={kind:bool(np.median([x["topi_normalized"] for x in nuisance if x["kind"]==kind]) <
                              np.median([x["b0_normalized"] for x in nuisance if x["kind"]==kind]))
                   for kind in ("amplitude","shift","noise")}
    nuisance_pass=bool(all(nuisance_kind.values()))
    reference_state={"pair_identity":reference.identity.canonical_payload(),"actual_raw":reference.actual_raw.tolist(),
      "predicted_raw":reference.predicted_raw.tolist(),"residual_raw":reference.residual_raw.tolist(),
      "standardizer_std":reference.standardizer_std.tolist(),"coordinates":core.CANONICAL_TAP_COORDS.tolist(),
      "basis_matrix":basis.matrix.tolist(),"Sigma":covariance.Sigma.tolist(),"W":covariance.W.tolist(),
      "covariance_input_digest_sha256":covariance.input_digest_sha256,
      "covariance_fit_identity_digest_sha256":covariance.audit["identity_digest_sha256"],
      "clean_reference_b0_median":clean_b0,"clean_reference_topi_median":clean_topi}
    return {"schema":"gnss-doppler-lab.nc-topi-stage0.synthetic.v2","seed":seed,
            "amplitude_semantics":"prompt-normalized shape-scale, not physical global gain",
            "attack_thresholds_used":False,"reference_state":reference_state,"raw_trials":raw,"second_peak_grid":grid,
            "nuisance_grid":nuisance,"criteria":{"equal_rmse_pass":equal_pass,
            "second_peak_pass":bool(power_pass and separation_pass),"second_peak_power_pass":bool(power_pass),
            "second_peak_separation_pass":bool(separation_pass),"nuisance_pass":nuisance_pass,
             "nuisance_kind_pass":nuisance_kind},
            "summaries":{"equal_rmse_trials":len(raw),"second_peak_rows":len(grid),
            "nuisance_rows":len(nuisance)}}


def _git_output(*args: str) -> str:
    import subprocess
    return subprocess.run(["git",*args],cwd=ROOT,check=True,text=True,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout.strip()


def _provenance_manifest(*,allow_dirty: bool=False) -> dict[str,object]:
    commit=_git_output("rev-parse","HEAD")
    status=_git_output("status","--porcelain=v1","--untracked-files=all")
    if status and not allow_dirty: raise RuntimeError("production execution requires a clean exact source worktree")
    tracked=_git_output("ls-files").splitlines()
    nc_files=sorted(x for x in tracked if ("nc_topi" in x.lower() or x=="docs/NC_TOPI_STAGE0.md"))
    return {"schema":"gnss-doppler-lab.nc-topi-stage0.provenance.v2",
      "source_commit":commit,"execution_code_commit":commit,"source_root":str(ROOT),
      "source_file_sha256":{name:sha256_file(ROOT/name) for name in nc_files},
      "source_file_inventory":nc_files,"worktree_clean":not bool(status),"diff_inventory":status.splitlines(),
      "attack_fit":False,"fit_scenarios":["cleanStatic"],
      "lineage":"single canonical regenerated Stage-0 lineage; frozen B0; exact algorithm and common pair mask"}


def verify_campaign_inputs(config: Mapping[str,object], checkpoint: Path, *,
                           verify_full_raw_hash: bool=False) -> dict[str,object]:
    core.validate_config(config)
    canonical={}
    expected_scenarios={"cleanStatic","DS1","DS2","DS3","DS7","DS8"}
    if set(config.get("canonical_inputs",{}))!=expected_scenarios:
        raise ValueError("canonical input inventory must be exactly cleanStatic,DS1,DS2,DS3,DS7,DS8")
    for scenario,item in config["canonical_inputs"].items():
        path=Path(item["path"]);actual=sha256_file(path)
        if actual!=item["sha256"]: raise ValueError(f"canonical hash mismatch: {scenario}")
        canonical[scenario]={"path":str(path),"size_bytes":path.stat().st_size,
            "expected_sha256":item["sha256"],"sha256_recomputed":actual}
    checkpoint_hash=sha256_file(checkpoint)
    if checkpoint_hash!=config["b0"]["checkpoint_sha256"]: raise ValueError("checkpoint hash mismatch")
    raw={}
    required_raw={"cleanStatic","cleanDynamic","DS1","DS2","DS3","DS7","DS8"}
    if set(config.get("raw_iq_inputs",{}))!=required_raw:
        raise ValueError("raw IQ inventory must contain the exact seven Stage-0 recordings")
    for scenario,item in config["raw_iq_inputs"].items():
        witness=raw_iq_witness(item["path"])
        if witness["size_bytes"]!=int(item["size_bytes"]): raise ValueError(f"raw IQ size mismatch: {scenario}")
        for key in ("first_1MiB_sha256","last_1MiB_sha256"):
            if witness[key]!=item[key]: raise ValueError(f"raw IQ witness mismatch: {scenario}/{key}")
        witness.update({"expected_full_sha256":item["sha256"],
          "full_hash_status":"expected_not_recomputed"})
        if verify_full_raw_hash:
            actual=sha256_file(item["path"])
            if actual!=item["sha256"]: raise ValueError(f"raw IQ full hash mismatch: {scenario}")
            witness.update({"full_sha256_recomputed":True,"full_hash_status":"verified","sha256_recomputed":actual})
        raw[scenario]=witness
    dynamic=config["clean_dynamic_nodes"];dynamic_path=Path(dynamic["path"])
    actual_dynamic=sha256_file(dynamic_path)
    if actual_dynamic!=dynamic["sha256"]: raise ValueError("cleanDynamic node hash mismatch")
    return {"schema":"gnss-doppler-lab.nc-topi-stage0.data-manifest.v2",
      "canonical_npz":canonical,"checkpoint":{"path":str(checkpoint),"size_bytes":checkpoint.stat().st_size,
      "expected_sha256":config["b0"]["checkpoint_sha256"],"sha256_recomputed":checkpoint_hash},
      "raw_iq":raw,"cleanDynamic_nodes":{"path":str(dynamic_path),"size_bytes":dynamic_path.stat().st_size,
      "expected_sha256":dynamic["sha256"],"sha256_recomputed":actual_dynamic,"available":True},
      "legacy_predictions":{"primary_use":False,"positive_control":"only exact key+value match",
      "regenerated_lineage_byte_identical_claim":False}}


def build_event_iq_context(examples: Sequence[SequenceExample], blocks: Sequence[IQBlock],
                           scenario: str) -> tuple[np.ndarray,list[dict[str,object]],dict[str,object]]:
    groups={}
    for index,item in enumerate(examples): groups.setdefault(item.target.event_key,[]).append(index)
    event_keys=sorted(groups);starts=[min(examples[i].target.source_start_s for i in groups[k]) for k in event_keys]
    recordings=[k[0] for k in event_keys]
    predictors,audit=causal_iq_context(starts,recordings,blocks,history=4,cadence=.5)
    ordered=sorted(blocks,key=lambda x:(x.recording_id,x.end_s));pair_x=np.empty((len(examples),4),float);rows=[]
    selected=audit["selected_block_indices"] if "selected_block_indices" in audit else None
    # Reconstruct selected indices exactly from the same strict as-of rule for stored raw evidence.
    for event_pos,(key,start) in enumerate(zip(event_keys,starts)):
        eligible=[i for i,x in enumerate(ordered) if x.recording_id==key[0] and x.end_s<=start]
        chosen=eligible[-4:]
        if len(chosen)!=4: raise RuntimeError("event lacks history4 IQ context")
        context=np.mean(np.stack([ordered[i].features for i in chosen]),axis=0)
        if not np.allclose(context,predictors[event_pos],rtol=0,atol=1e-15): raise RuntimeError("IQ history reduction mismatch")
        for pair_index in groups[key]: pair_x[pair_index]=context
        rows.append({"scenario":scenario,"physical_recording_id":key[0],"event_id":f"{key[0]}@{key[1]:.10g}",
          "window_bin_s":key[1],"target_source_start_s":start,"history_blocks":4,"cadence_seconds":.5,
          "block_end_s":";".join(format(ordered[i].end_s,".17g") for i in chosen),
          "block_start_s":";".join(format(ordered[i].start_s,".17g") for i in chosen),
          "sample_offset":";".join(str(ordered[i].sample_offset) for i in chosen),
          "sample_count":";".join(str(ordered[i].sample_count) for i in chosen),
          "block_features_json":json.dumps([ordered[i].features.tolist() for i in chosen],separators=(",",":")),
          "context_features_json":json.dumps(context.tolist(),separators=(",",":")),
          "linked_prns":";".join(sorted({examples[i].target.prn for i in groups[key]})),
          "linked_pair_count":len(groups[key]),"history_reducer":"arithmetic_mean_per_feature"})
    audit.update({"event_contexts":len(rows),"pair_contexts":len(examples),"broadcast_same_event_vector":True})
    return pair_x,rows,audit


def _scenario_lineage(config: Mapping[str,object], scenario: str, b0: FrozenB0):
    if scenario=="cleanDynamic":
        nodes,audit=load_external_normal_nodes(config["clean_dynamic_nodes"]["path"],
            expected_sha256=config["clean_dynamic_nodes"]["sha256"])
        if not audit.get("available"): raise RuntimeError(f"cleanDynamic nodes unavailable: {audit.get(reason)}")
    else:
        item=config["canonical_inputs"][scenario]
        nodes,audit=build_node_windows(item["path"],scenario,recording_id=scenario,
            window_s=float(config["b0"]["window_seconds"]),stride_s=float(config["b0"]["stride_seconds"]),
            min_epochs=int(config["b0"]["minimum_raw_epochs"]),expected_sha256=item["sha256"])
    examples,seq_audit=build_sequence_examples(nodes,sequence_length=int(config["b0"]["sequence_length"]),
        stride_s=float(config["b0"]["stride_seconds"]))
    if not examples: raise RuntimeError(f"{scenario} has no B0 examples")
    pairs,rmse,infer_audit=infer_b0_pairs(examples,b0)
    assert_same_epoch_mask({"B0":pairs,"canonical_pairs":pairs})
    return examples,pairs,{"nodes":audit,"sequences":seq_audit,"inference":infer_audit,
                           "b0_rmse_mean":float(np.mean(rmse))}


def _iq_for_scenario(config: Mapping[str,object], scenario: str, examples: Sequence[SequenceExample]):
    raw=config["raw_iq_inputs"][scenario];iqcfg=config["iq_conditioner"]
    max_end=max(min(item.target.source_start_s for item in examples if item.target.event_key==key)
                for key in {x.target.event_key for x in examples})
    blocks=extract_iq_blocks(raw["path"],scenario,sample_rate_hz=int(iqcfg["sample_rate_hz"]),
      block_duration_s=float(iqcfg["block_duration_seconds"]),block_stride_s=float(iqcfg["block_stride_seconds"]),
      max_end_s=max_end+1e-9)
    return build_event_iq_context(examples,blocks,scenario)


def _fit_geometry(config, examples, pairs, gate: StageGate):
    starts=np.asarray([x.target.source_start_s for x in examples]);ends=np.asarray([x.target.source_end_s for x in examples])
    masks=core.source_support_split(starts,ends,scenario="cleanStatic")
    indices={"normal_train":np.flatnonzero(masks.train),"normal_calibration":np.flatnonzero(masks.calibration),
             "normal_holdout":np.flatnonzero(masks.holdout),"excluded_boundary_crossing":np.flatnonzero(masks.unassigned)}
    if any(len(indices[x])==0 for x in ("normal_train","normal_calibration","normal_holdout")):
        raise RuntimeError("cleanStatic source-support split has an empty mandatory role")
    train_pairs=[pairs[i] for i in indices["normal_train"]];train_ids=tuple(x.identity for x in train_pairs)
    train_prov=core.FitProvenance("cleanStatic","normal_train",train_ids)
    covariance=core.fit_shrinkage_covariance(train_pairs,provenance=train_prov)
    gate.seal_fit("covariance",covariance,train_ids);gate.record_phase("fit_covariance_clean_train")
    topi=np.empty(len(pairs))
    for i,pair in enumerate(pairs):
        basis=core.primary_tangent_basis(pair,core.CANONICAL_TAP_COORDS,covariance)
        topi[i]=core.produce_topi_scores(pair,basis,covariance).topi
    gate.record_phase("compute_topi_all_clean")
    return {"covariance":covariance,"split_indices":indices,"topi":topi,"train_provenance":train_prov}


def _fit_conditioners(config,pairs,iq_x,state,gate: StageGate):
    indices=state["split_indices"];topi=state["topi"];train_prov=state["train_provenance"]
    train_ids=train_prov.identities
    conditioner=core.RobustConditioner().fit(iq_x[indices["normal_train"]],topi[indices["normal_train"]],provenance=train_prov)
    gate.seal_fit("conditioner",conditioner,train_ids)
    shuffled_target=core.shuffled_control_target(topi[indices["normal_train"]],provenance=train_prov,
        feature_names=IQ_FEATURE_NAMES,seed=int(config["iq_conditioner"]["shuffle_control"]["seed"]))
    shuffled=core.RobustConditioner().fit(iq_x[indices["normal_train"]],shuffled_target,provenance=train_prov)
    gate.seal_fit("shuffled_conditioner",shuffled,train_ids);gate.record_phase("fit_conditioners_clean_train")
    cal_ids=tuple(pairs[i].identity for i in indices["normal_calibration"])
    cal_prov=core.FitProvenance("cleanStatic","normal_calibration",cal_ids)
    conditioner.calibrate_cap(iq_x[indices["normal_calibration"]],provenance=cal_prov,q=.995)
    shuffled.calibrate_cap(iq_x[indices["normal_calibration"]],provenance=cal_prov,q=.995)
    gate.seal_fit("conditioner_cap",conditioner,cal_ids);gate.seal_fit("shuffled_cap",shuffled,cal_ids)
    gate.record_phase("calibrate_conditioner_caps")
    state.update({"conditioner":conditioner,"shuffled":shuffled,"calibration_provenance":cal_prov})
    return state


def _fit_state(config: Mapping[str,object], examples, pairs, iq_x, gate: StageGate):
    starts=np.asarray([x.target.source_start_s for x in examples]);ends=np.asarray([x.target.source_end_s for x in examples])
    masks=core.source_support_split(starts,ends,scenario="cleanStatic")
    indices={"normal_train":np.flatnonzero(masks.train),"normal_calibration":np.flatnonzero(masks.calibration),
             "normal_holdout":np.flatnonzero(masks.holdout),"excluded_boundary_crossing":np.flatnonzero(masks.unassigned)}
    if any(len(indices[x])==0 for x in ("normal_train","normal_calibration","normal_holdout")):
        raise RuntimeError("cleanStatic source-support split has an empty mandatory role")
    train_pairs=[pairs[i] for i in indices["normal_train"]];train_ids=tuple(x.identity for x in train_pairs)
    train_prov=core.FitProvenance("cleanStatic","normal_train",train_ids)
    covariance=core.fit_shrinkage_covariance(train_pairs,provenance=train_prov)
    gate.seal_fit("covariance",covariance,train_ids);gate.record_phase("compute_topi_all_clean")
    topi=np.empty(len(pairs))
    for i,pair in enumerate(pairs):
        basis=core.primary_tangent_basis(pair,core.CANONICAL_TAP_COORDS,covariance)
        topi[i]=core.produce_topi_scores(pair,basis,covariance).topi
    conditioner=core.RobustConditioner().fit(iq_x[indices["normal_train"]],topi[indices["normal_train"]],provenance=train_prov)
    gate.seal_fit("conditioner",conditioner,train_ids)
    shuffled_target=core.shuffled_control_target(topi[indices["normal_train"]],provenance=train_prov,
        feature_names=IQ_FEATURE_NAMES,seed=int(config["iq_conditioner"]["shuffle_control"]["seed"]))
    shuffled=core.RobustConditioner().fit(iq_x[indices["normal_train"]],shuffled_target,provenance=train_prov)
    gate.seal_fit("shuffled_conditioner",shuffled,train_ids);gate.record_phase("fit_conditioners_clean_train")
    cal_ids=tuple(pairs[i].identity for i in indices["normal_calibration"])
    cal_prov=core.FitProvenance("cleanStatic","normal_calibration",cal_ids)
    conditioner.calibrate_cap(iq_x[indices["normal_calibration"]],provenance=cal_prov,q=.995)
    shuffled.calibrate_cap(iq_x[indices["normal_calibration"]],provenance=cal_prov,q=.995)
    gate.seal_fit("conditioner_cap",conditioner,cal_ids);gate.seal_fit("shuffled_cap",shuffled,cal_ids)
    gate.record_phase("calibrate_conditioner_caps")
    return {"covariance":covariance,"conditioner":conditioner,"shuffled":shuffled,
      "split_indices":indices,"topi":topi,"train_provenance":train_prov,"calibration_provenance":cal_prov}


def _role_interval(scenario: str,start: float,end: float) -> str:
    if scenario!="cleanStatic": return "evaluation"
    if end<=300: return "normal_train"
    if start>=320 and end<=400: return "normal_calibration"
    if start>=420: return "normal_holdout"
    return "excluded_boundary_crossing"


def score_scenario(examples, pairs, iq_x, state, *, onsets=None):
    covariance=state["covariance"];conditioner=state["conditioner"];shuffled=state["shuffled"]
    scored=[];method_ids={name:[] for name in METHODS}
    for pair_sequence_index,(item,pair,features) in enumerate(zip(examples,pairs,iq_x)):
        basis=core.primary_tangent_basis(pair,core.CANONICAL_TAP_COORDS,covariance)
        top=core.produce_topi_scores(pair,basis,covariance)
        nc=core.produce_nc_topi_scores(pair,basis,covariance,conditioner=conditioner,iq_features=features[None,:])
        sh=core.produce_nc_topi_scores(pair,basis,covariance,conditioner=shuffled,iq_features=features[None,:])
        width_basis=core.build_width_ablation_basis(pair,core.CANONICAL_TAP_COORDS,covariance)
        width=core.produce_width_ablation_scores(pair,width_basis,covariance).score
        amp=core.weighted_project(pair.residual_raw,basis.matrix[:,[0]],covariance.W)
        shift=core.weighted_project(pair.residual_raw,basis.matrix[:,[1]],covariance.W)
        values={"B0":top.b0,"total":top.total,"amp_only":amp.tangent_energy,
          "shift_only":shift.tangent_energy,"amp_shift":top.tangent,
          "amp_shift_width":width.tangent,"TOPI":top.topi,"NC_TOPI":nc.nc_topi,
          "NC_TOPI_time_shuffle":sh.nc_topi,"NC_TOPI_conditioning_removed":top.topi}
        if set(values)!=set(METHODS) or not np.isfinite(list(values.values())).all(): raise RuntimeError("method score inventory invalid")
        for name in METHODS: method_ids[name].append(pair.identity)
        phase,label=event_phase(pair.identity.scenario,item.target.source_start_s,item.target.source_end_s,onsets)
        scored.append({"example":item,"pair":pair,"scores":values,"phase":phase,"label":label,
                        "pair_sequence_index":pair_sequence_index})
    assert_same_epoch_mask(method_ids)
    return scored


def aggregate_scored(scored):
    groups={}
    for row in scored: groups.setdefault(row["example"].target.event_key,[]).append(row)
    prn_rows=[];event_rows=[];event_internal=[]
    for event_pos,key in enumerate(sorted(groups)):
        group=groups[key];scenario=group[0]["pair"].identity.scenario
        start=min(x["example"].target.source_start_s for x in group);end=max(x["example"].target.source_end_s for x in group)
        event_id=f"{key[0]}@{key[1]:.10g}";role=_role_interval(scenario,start,end)
        phases={x["phase"] for x in group};labels={str(x["label"]) for x in group}
        if len(phases)!=1 or len(labels)!=1: raise RuntimeError("event PRNs disagree on phase/label")
        phase=next(iter(phases));label=group[0]["label"]
        base={"scenario":scenario,"physical_recording_id":key[0],"event_id":event_id,
          "target_index":event_pos,"availability_time_s":end,"source_start_s":start,"source_end_s":end,
          "role":role,"phase":phase,"label":label,"valid":phase!="transition_excluded","tracked_prn_count":len(group)}
        event=dict(base,row_level="event",prn="")
        for method in METHODS:
            ids=[x["pair"].identity.prn for x in group];values=[x["scores"][method] for x in group]
            for agg in AGGREGATORS:
                event[f"{method}_{agg}"]=core.aggregate_prn_scores(ids,values,method=agg).score
        event_rows.append(event);event_internal.append({"meta":base,"scores":{m:{a:event[f"{m}_{a}"] for a in AGGREGATORS} for m in METHODS}})
        for x in group:
            row=dict(base,row_level="prn",prn=x["pair"].identity.prn,prn_target_index=x["pair"].identity.target_index,
                      pair_sequence_index=x["pair_sequence_index"],
                     availability_time_s=x["pair"].identity.availability_time_s,
                     source_start_s=x["example"].target.source_start_s,source_end_s=x["example"].target.source_end_s,
                     role=_role_interval(scenario,x["example"].target.source_start_s,x["example"].target.source_end_s))
            row.update(x["scores"]);prn_rows.append(row)
    return prn_rows+event_rows,event_internal


def calibrate_typed_thresholds(clean_events, clean_pairs, state, gate):
    cal=[x for x in clean_events if x["meta"]["role"]=="normal_calibration"]
    if not cal: raise RuntimeError("no clean calibration events")
    identities=tuple(core.EpochIdentity(x["meta"]["physical_recording_id"],"cleanStatic","EVENT",
       int(x["meta"]["target_index"]),float(x["meta"]["availability_time_s"])) for x in cal)
    provenance=core.FitProvenance("cleanStatic","normal_calibration",identities);typed={};serialized={}
    detector_name=lambda m:"NC-TOPI" if m=="NC_TOPI" else m
    for method in METHODS:
      for agg in AGGREGATORS:
       values=[x["scores"][method][agg] for x in cal]
       for q,suffix in ((.99,"q99"),(.995,"q995")):
        obj=core.calibrate_threshold(values,q,provenance=provenance,detector=detector_name(method),aggregator=agg)
        key=f"{method}/{agg}/{suffix}";typed[key]=obj
        serialized[key]={"detector":detector_name(method),"aggregator":agg,"quantile":q,"method":"higher",
          "comparison":"strict >","value":obj.value,"rows":len(values),"score_digest_sha256":obj.score_digest_sha256,
          "identity_digest_sha256":obj.identity_digest_sha256,"clean_scenario":"cleanStatic","clean_role":"normal_calibration",
          "typed_seal":obj._construction_seal}
    gate.seal_fit("thresholds",typed,identities);gate.record_phase("calibrate_typed_thresholds")
    return typed,serialized,identities


def _write_iq_evidence(root: Path, rows):
    _write_csv_rows(root/"iq_context.csv",rows)
    np.savez(root/"iq_context.npz",
      target_source_start_s=np.asarray([float(x["target_source_start_s"]) for x in rows]),
      block_end_s=np.asarray([[float(y) for y in x["block_end_s"].split(";")] for x in rows]),
      sample_offset=np.asarray([[int(y) for y in x["sample_offset"].split(";")] for x in rows]),
      context_features=np.asarray([json.loads(x["context_features_json"]) for x in rows]))


def _publish_freeze(out: Path, config, data_manifest, fit_audit, iq_rows, lineage):
    with ArtifactStage(out) as stage:
        payload={"schema":"gnss-doppler-lab.nc-topi-stage0.freeze.v2","stop_after_freeze":True,
          "attack_loader_calls":0,"attack_fit":False,"phase_trace":fit_audit["phase_trace"],
          "sealed_fits":fit_audit["sealed_fits"],"split_counts":fit_audit["split_counts"],
          "pair_count":fit_audit["pair_count"],"iq_context_count":len(iq_rows)}
        for name,value in (("freeze_manifest.json",payload),("config.json",config),
          ("data_manifest.json",data_manifest),("fit_audit.json",fit_audit),("model_lineage_audit.json",lineage)):
            (stage.path/name).write_text(json.dumps(value,sort_keys=True,indent=2)+"\n")
        _write_iq_evidence(stage.path,iq_rows)
        import importlib.util
        spec=importlib.util.spec_from_file_location("freeze_verify",ROOT/"scripts/summarize_nc_topi_stage0.py")
        verifier=importlib.util.module_from_spec(spec);spec.loader.exec_module(verifier)
        verifier.verify_freeze_bundle(stage.path)
        if out.exists(): raise FileExistsError(out)
        os.replace(stage.path,out);stage._published=True
    return out


def load_attack_evaluation(config, b0, scenarios=core.ATTACK_SCENARIOS+("cleanDynamic",)):
    result={}
    for scenario in scenarios: result[scenario]=_scenario_lineage(config,scenario,b0)
    return result


def run_campaign(config, out, *, checkpoint: str|Path|None=None, stop_after_freeze: bool=False,
                 verify_full_raw_hash: bool=False, device: str|None=None,
                 attack_loader: Callable|None=None) -> Path:
    cfg=core.load_config(config) if isinstance(config,(str,Path)) else dict(config)
    core.validate_config(cfg);out=Path(out)
    checkpoint=Path(checkpoint or cfg.get("checkpoint_path",ROOT/"artifacts/ai_morph_gru_cleanStatic_q70_frame/prn_local_gru_predictor.pt"))
    gate=StageGate();_seed_everything();gate.record_phase("verify_inputs_source")
    data_manifest=verify_campaign_inputs(cfg,checkpoint,verify_full_raw_hash=verify_full_raw_hash)
    provenance=_provenance_manifest(allow_dirty=bool(cfg.get("test_fixture")));b0=FrozenB0(checkpoint,cfg["b0"],device=device)
    gate.record_phase("load_build_clean_canonical_nodes_b0_pairs")
    examples,pairs,lineage=_scenario_lineage(cfg,"cleanStatic",b0)
    gate.record_phase("assign_source_support_split")
    state=_fit_geometry(cfg,examples,pairs,gate)
    iq_x,iq_rows,iq_audit=_iq_for_scenario(cfg,"cleanStatic",examples);gate.record_phase("extract_all_clean_iq")
    state=_fit_conditioners(cfg,pairs,iq_x,state,gate)
    clean_scored=score_scenario(examples,pairs,iq_x,state,onsets=None);gate.record_phase("score_all_clean_methods")
    clean_rows,clean_events=aggregate_scored(clean_scored);gate.record_phase("aggregate_clean_events")
    typed_thresholds,thresholds,threshold_ids=calibrate_typed_thresholds(clean_events,pairs,state,gate)
    gate.freeze()
    split_counts={k:len(v) for k,v in state["split_indices"].items()}
    fit_audit={**gate.audit,"schema":"gnss-doppler-lab.nc-topi-stage0.fit-audit.v2",
      "fit_scenarios":["cleanStatic"],"attack_fit":False,"pair_count":len(pairs),"split_counts":split_counts,
      "covariance_audit":dict(state["covariance"].audit),"conditioner_fit":dict(state["conditioner"].fit_manifest_),
      "conditioner_cap":dict(state["conditioner"].cap_manifest_),"shuffle_fit":dict(state["shuffled"].fit_manifest_),
      "shuffle_cap":dict(state["shuffled"].cap_manifest_),"threshold_identity_count":len(threshold_ids),
      "iq_audit":iq_audit}
    lineage.update({"primary_lineage":"canonical regenerated","same_pair_mask":True,
      "legacy_predictions_primary":False,"legacy_positive_control":"unavailable unless exact key+value match"})
    if stop_after_freeze:
        if gate.audit["attack_loader_calls"]!=0: raise RuntimeError("stop-after-freeze loaded attack data")
        return _publish_freeze(out,cfg,data_manifest,fit_audit,iq_rows,lineage)
    loader=attack_loader or load_attack_evaluation
    loaded=loader(cfg,b0)
    gate.load_attack_labels(cfg["attacks"]["onsets_seconds"])
    gate.record_phase("evaluate_attacks_and_cleanDynamic")
    all_rows=list(clean_rows);all_events=list(clean_events);all_iq=list(iq_rows);lineages={"cleanStatic":lineage}
    for scenario in (*core.ATTACK_SCENARIOS,"cleanDynamic"):
        examples_s,pairs_s,lineage_s=loaded[scenario]
        x_s,iq_s,iq_audit_s=_iq_for_scenario(cfg,scenario,examples_s)
        scored=score_scenario(examples_s,pairs_s,x_s,state,onsets=cfg["attacks"]["onsets_seconds"])
        rows,events=aggregate_scored(scored);all_rows.extend(rows);all_events.extend(events);all_iq.extend(iq_s)
        lineage_s["iq_audit"]=iq_audit_s;lineages[scenario]=lineage_s
    scenario_metrics,ablation_metrics,bootstrap,decision=_campaign_statistics(all_events,typed_thresholds,cfg,state,pairs)
    synthetic=run_synthetic_physics([pairs[i] for i in state["split_indices"]["normal_train"]],state["covariance"])
    decision=_derive_decision(scenario_metrics,bootstrap,synthetic)
    gate.record_phase("metrics_bootstrap_physics_decision")
    fit_audit.update(gate.audit)
    provenance["attack_fit"]=False;provenance["fit_scenarios"]=["cleanStatic"]
    extras={"model_lineage_audit.json":lineages,"iq_context.csv":_csv_bytes(all_iq),
            "iq_context.npz":_iq_npz_bytes(all_iq)}
    return publish_artifact(out,config=cfg,data_manifest=data_manifest,thresholds=thresholds,
      per_epoch_rows=all_rows,scenario_metrics=scenario_metrics,ablation_metrics=ablation_metrics,
      synthetic=synthetic,bootstrap=bootstrap,decision=decision,provenance=provenance,
      fit_audit=fit_audit,extras=extras)


def _csv_bytes(rows):
    import io
    stream=io.StringIO();fields=[]
    for row in rows:
      for key in row:
       if key not in fields: fields.append(key)
    writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n");writer.writeheader();writer.writerows(rows)
    return stream.getvalue()


def _iq_npz_bytes(rows):
    import io
    stream=io.BytesIO();np.savez(stream,target_source_start_s=np.asarray([float(x["target_source_start_s"]) for x in rows]),
      block_end_s=np.asarray([[float(y) for y in x["block_end_s"].split(";")] for x in rows]),
      sample_offset=np.asarray([[int(y) for y in x["sample_offset"].split(";")] for x in rows]),
      context_features=np.asarray([json.loads(x["context_features_json"]) for x in rows]))
    return stream.getvalue()


def _campaign_statistics(events,thresholds,cfg,state,pairs):
    # Exact finite inventory; every scalar carries reconstructable numerator/denominator or labels/scores.
    rows=[]
    for scenario in ("cleanStatic","cleanDynamic",*core.ATTACK_SCENARIOS):
      scenario_events=[x for x in events if x["meta"]["scenario"]==scenario]
      for method in METHODS:
       detector="NC-TOPI" if method=="NC_TOPI" else method
       for agg in AGGREGATORS:
        for q,suffix in ((.99,"q99"),(.995,"q995")):
         threshold=thresholds[f"{method}/{agg}/{suffix}"].value
         if scenario=="cleanStatic": eligible=[x for x in scenario_events if x["meta"]["role"]=="normal_holdout"]
         elif scenario=="cleanDynamic": eligible=scenario_events
         else: eligible=[x for x in scenario_events if x["meta"]["phase"]=="stable_pre"]
         scores=[x["scores"][method][agg] for x in eligible];alarms=[x>threshold for x in scores]
         rows.append({"scenario":scenario,"method":method,"aggregator":agg,"quantile":q,"metric":"fpr",
           "phase":"normal_holdout" if scenario=="cleanStatic" else ("normal" if scenario=="cleanDynamic" else "stable_pre"),
           "value":sum(alarms)/len(alarms),"numerator":sum(alarms),"denominator":len(alarms)})
         if scenario in core.ATTACK_SCENARIOS:
          classify=[x for x in scenario_events if x["meta"]["phase"] in ("stable_pre","post")]
          labels=[int(x["meta"]["label"]) for x in classify];values=[x["scores"][method][agg] for x in classify]

          if set(labels)!={0,1}: raise RuntimeError(f"{scenario} classification inventory lacks both classes: {sorted(set(labels))}")
          pauc=core.standardized_pauc(labels,values,max_fpr=.05)
          rows.append({"scenario":scenario,"method":method,"aggregator":agg,"quantile":q,"metric":"pauc","phase":"stable_pre+post",
            "value":pauc,"labels_json":json.dumps(labels,separators=(",",":")),"scores_json":json.dumps(values,separators=(",",":"))})
    ab=[]
    for method in METHODS:
      for agg in AGGREGATORS:
       for q in (.99,.995):
        values=[float(x["value"]) for x in rows if x["scenario"] in core.ATTACK_SCENARIOS and x["method"]==method and x["aggregator"]==agg and float(x["quantile"])==q and x["metric"]=="pauc"]
        ab.append({"method":method,"aggregator":agg,"quantile":q,"metric":"mean_attack_pauc","value":float(np.mean(values)),
                   "numerator":float(np.sum(values)),"denominator":len(values)})
    comparisons=[];pairs_cmp=(("NC-B0","NC_TOPI","B0"),("TOPI-B0","TOPI","B0"),
      ("NC-TOPI","NC_TOPI","TOPI"),("shuffleNC-TOPI","NC_TOPI_time_shuffle","TOPI"))
    for scenario in core.ATTACK_SCENARIOS:
      subset=[x for x in events if x["meta"]["scenario"]==scenario and x["meta"]["phase"] in ("stable_pre","post")]
      labels=[int(x["meta"]["label"]) for x in subset];times=[x["meta"]["availability_time_s"] for x in subset];recs=[x["meta"]["physical_recording_id"] for x in subset]
      for agg in AGGREGATORS:
       for name,a,b in pairs_cmp:
        sa=[x["scores"][a][agg] for x in subset];sb=[x["scores"][b][agg] for x in subset]
        result=core.paired_pauc_delta_block_bootstrap(labels,sa,sb,recs,times,reps=2000,seed=core.DEFAULT_SEED)
        comparisons.append({"scenario":scenario,"aggregator":agg,"comparison":name,"labels":labels,"score_a":sa,"score_b":sb,
          "recording_ids":recs,"availability_time_s":times,"seed":core.DEFAULT_SEED,"available":result.available,
          "reason":result.reason,"point_estimate":result.point_estimate if np.isfinite(result.point_estimate) else None,
          "ci":list(result.ci) if result.available else [None,None],"valid_reps":result.valid_reps,
          "replicates_sha256":hashlib.sha256(np.ascontiguousarray(result.replicates,dtype=np.float64).tobytes()).hexdigest()})
    return rows,ab,{"schema":"gnss-doppler-lab.nc-topi-stage0.bootstrap.v2","comparisons":comparisons,
      "expected_inventory":{"scenarios":list(core.ATTACK_SCENARIOS),"aggregators":list(AGGREGATORS),
      "comparisons":[x[0] for x in pairs_cmp],"repetitions":2000}},{}


def _derive_decision(metrics,bootstrap,synthetic):
    def metric(s,m,a,q,phase):
      hits=[x for x in metrics if x["scenario"]==s and x["method"]==m and x["aggregator"]==a and x["quantile"]==q and x["metric"]=="fpr" and x["phase"]==phase]
      if len(hits)!=1: raise RuntimeError("decision metric inventory is not unique")
      return float(hits[0]["value"])
    pauc=lambda s,m: next(float(x["value"]) for x in metrics if x["scenario"]==s and x["method"]==m and x["aggregator"]=="median" and x["quantile"]==.99 and x["metric"]=="pauc")
    cmp={(x["scenario"],x["aggregator"],x["comparison"]):x for x in bootstrap["comparisons"]}
    nc={s:pauc(s,"NC_TOPI") for s in core.ATTACK_SCENARIOS};b0={s:pauc(s,"B0") for s in core.ATTACK_SCENARIOS}
    lower={s:(cmp[(s,"median","NC-B0")]["ci"][0] if cmp[(s,"median","NC-B0")]["available"] else -1.) for s in core.ATTACK_SCENARIOS}
    upper={s:(cmp[(s,"median","NC-B0")]["ci"][1] if cmp[(s,"median","NC-B0")]["available"] else 1.) for s in core.ATTACK_SCENARIOS}
    evidence={"clean_nc_fpr":metric("cleanStatic","NC_TOPI","median",.99,"normal_holdout"),
      "clean_b0_fpr":metric("cleanStatic","B0","median",.99,"normal_holdout"),
      "stable_pre_fpr":{s:metric(s,"NC_TOPI","median",.99,"stable_pre") for s in core.ATTACK_SCENARIOS},
      "nc_pauc":nc,"b0_pauc":b0,"pauc_delta":{s:nc[s]-b0[s] for s in core.ATTACK_SCENARIOS},
      "nc_delay":{s:None for s in core.ATTACK_SCENARIOS},"b0_delay":{s:None for s in core.ATTACK_SCENARIOS},
      "pauc_ci_lower":lower,"pauc_ci_upper":upper,
      "equal_rmse_pass":bool(synthetic["criteria"]["equal_rmse_pass"] and synthetic["criteria"].get("nuisance_pass",False)),
      "second_peak_pass":bool(synthetic["criteria"]["second_peak_pass"]),
      "actual_nc_mean_pauc":float(np.mean(list(nc.values()))),
      "topi_mean_pauc":float(np.mean([pauc(s,"TOPI") for s in core.ATTACK_SCENARIOS])),
      "shuffled_nc_mean_pauc":float(np.mean([pauc(s,"NC_TOPI_time_shuffle") for s in core.ATTACK_SCENARIOS]))}
    result=core.evaluate_stage0_decision(**evidence)
    return {"schema":"gnss-doppler-lab.nc-topi-stage0.decision.v2","status":result.status,
      "criteria":result.criteria,"counts":result.counts,"missing_evidence":result.missing_evidence,
      "no_go_triggers":result.no_go_triggers,"validation_errors":result.validation_errors,"evidence":evidence,
      "evidence_source":"independent-reconstructable metrics/bootstrap/physics only"}


if __name__ == "__main__":
    raise SystemExit(main())

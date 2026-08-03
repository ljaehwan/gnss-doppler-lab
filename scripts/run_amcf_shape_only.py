#!/usr/bin/env python3
"""Leakage-safe AMCF Shape-Only campaign runner.

Primary mode consumes only hash-pinned canonical NPZs and writes the one
preregistered artifact directory atomically.  ``--smoke`` exercises the same
serialization, provenance, alarm, metric, and summarizer contracts with a
small synthetic fixture outside the final artifact path; it never runs an
attack campaign.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from typing import NamedTuple
from dataclasses import dataclass
from types import MappingProxyType
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FINAL_ARTIFACT = Path("artifacts/amcf_r1_shape_only")
TAP_NAMES = ("E4", "E3", "E2", "E", "P", "L", "L2", "L3", "L4")
SIDE_INDICES = (0, 1, 2, 3, 5, 6, 7, 8)
COMPLEX_SCHEMA = ("median_real", "median_imag", "mad_real", "mad_imag")
MAGNITUDE_SCHEMA = ("median_abs", "mad_abs")
SEEDS = (101, 202, 303)
PRIMARY_BASE_SHA = "ef36f265fca856914b7c79b77fd0a2f85f89b4d1"
PRIMARY_MIN_VALID_ROWS = 5
B0_FROZEN_Q99 = 1.7035537524611113
B0_FROZEN_Q995 = 1.7035537524611113
B0_FROZEN_INTERVAL = (300.0, 330.0)
B0_CONTRACT_HASHES = {
    "thresholds.json": "cf013a78a9ec91ba764d1ca86ce2582962466e3e70a0bcb10400836852b00ce2",
    "config.json": "b65a00ba4c8681875e6a8969512ba0c68f2ed91f1ccd70984e05b69ee35f65ab",
    "provenance.json": "55ea998a18d6caed76c8cca2ab01a4dbad94fd1e96d68602c2b659e4dcbd3ce7",
}
PRIMARY_FIT_CONFIG = MappingProxyType({
    "optimizer": "AdamW", "learning_rate": 1e-3, "df": 4.0,
    "tuple_batch_size": 256, "validation_chunk_size": 4096,
    "validation_weighting": "exact tuple sum/count", "max_epochs": 200,
    "patience": 20, "early_stop_tolerance": 1e-12,
    "gradient_rule": "every trainable parameter finite gradient every update",
    "targets_all9": list(range(8)), "targets_EPL": [3, 4],
})
ONSETS = {"DS1": 100.0, "DS2": 100.0, "DS3": 100.0, "DS7": 110.0, "DS8": 110.0}
CANONICAL = {
    "cleanStatic": (Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-clean-graph-input-v2/exports/cleanStatic.npz"), "fcd1d378c28e79fe4a550b65fc1208cde3c8fb334db11406a07fed4d90fba237"),
    "DS1": (Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds123-graph-input/exports/ds1.npz"), "b24d947c83890dbfa1c801bfbcb72e1fd192dd66509e927eb5afb8118902b072"),
    "DS2": (Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds123-graph-input/exports/ds2.npz"), "dae0f245cbb107febd220c6de33b9a279a2bad356cb0ba772daf9418bc75d7c9"),
    "DS3": (Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-ds123-graph-input/exports/ds3.npz"), "38eb5842dfec306d99bf0c5d61df6cffcb6faa25ed63721cafa8e3c3776f9b3e"),
    "DS7": (Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/ds7-sealed-input/exports/ds7.npz"), "d0e6da4e27d51e3e96abf2ef7786501124072f28667671e4e40da756eb35f3c8"),
    "DS8": (Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/cmte-a2-ds8-complex-d67f813/exports/ds8.npz"), "d1973fa150b7b4e7359df4827f36ce60289f206e9db11c1ac2bc1fd33a0df533"),
}
B0_DIR = ROOT / "artifacts/cmte_a2_texbat_epochfix/per_epoch"
REQUIRED_INVENTORY = (
    "README.md", "config.json", "feature_schema.json", "provenance.json",
    "input_hashes.json", "training_history.csv", "convergence_audit.json",
    "thresholds.json", "scenario_metrics.csv", "seed_metrics.csv",
    "paired_comparisons.csv", "decision.json", "feature_audit.json", "window_qa.json",
    "feature_cache", "per_epoch", "plots", "models", "hashes.json",
)
PRIMARY_REQUIRED_INVENTORY = REQUIRED_INVENTORY + ("calibration_evidence.json",)
SHAPE_ONLY_ALLOWED_FILES = frozenset({
    "docs/AMCF_SHAPE_ONLY.md", "src/gnss_doppler_lab/amcf_shape_only.py",
    "tests/test_amcf_shape_only.py", "tests/test_amcf_shape_only_runner.py",
    "scripts/run_amcf_shape_only.py", "scripts/summarize_amcf_shape_only.py",
})
SOURCE_FILES = (
    "src/gnss_doppler_lab/amcf_shape_only.py", "docs/AMCF_SHAPE_ONLY.md",
    "scripts/run_amcf_shape_only.py", "scripts/summarize_amcf_shape_only.py",
    "tests/test_amcf_shape_only.py", "tests/test_amcf_shape_only_runner.py",
)
ALLOWED_NPZ_FIELDS = ("complex_iq", "time_s", "prn", "channel", "segment_index", "sample_count")


class GateEvidence(NamedTuple):
    minimum: float
    raw_quantile: float
    positive_floor: float
    epsilon: float = 1e-12
    quantile: float = .005
    method: str = "higher"


def sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def canonical_json(value: Any) -> bytes:
    return (json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(_jsonable(value), sort_keys=True, indent=2,
                                ensure_ascii=True, allow_nan=False).encode() + b"\n")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Iterable[str] | None = None) -> None:
    rows = list(rows); path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fields or [])
    if not names:
        for row in rows:
            for key in row:
                if key not in names:
                    names.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        if not names:
            return
        writer = csv.DictWriter(f, names, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k, "")) for k in names})


def _csv_value(value: Any) -> Any:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"))
    return value


def _digest_value(value: Any) -> str:
    h = hashlib.sha256()
    def add(x: Any, name: str = "root") -> None:
        h.update(name.encode() + b"\0")
        if isinstance(x, np.ndarray):
            a = np.ascontiguousarray(x)
            h.update(str(a.dtype).encode() + b"\0")
            h.update(canonical_json(list(a.shape)))
            h.update(a.tobytes(order="C"))
        elif isinstance(x, Mapping):
            for key in sorted(x):
                add(x[key], str(key))
        else:
            h.update(canonical_json(x))
    add(value)
    return h.hexdigest()


def load_canonical_npz(path: Path | str, expected_hash: str, scenario: str,
                       *, tap_order: Iterable[str]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    path = Path(path)
    actual = sha256(path)
    if actual != expected_hash:
        raise ValueError(f"{scenario} SHA-256 mismatch: expected {expected_hash}, got {actual}")
    if tuple(tap_order) != TAP_NAMES:
        raise ValueError(f"tap order metadata mismatch: required {TAP_NAMES}")
    # Intentionally index only the allowlist.  In particular, f.files is not
    # traversed and cn0_db_hz is never read or copied.
    with np.load(path, allow_pickle=False) as f:
        try:
            data = {name: np.array(f[name], copy=True) for name in ALLOWED_NPZ_FIELDS}
        except KeyError as exc:
            raise ValueError(f"missing canonical field {exc}") from exc
    n = len(data["time_s"])
    if data["complex_iq"].shape != (n, 9, 2):
        raise ValueError("complex_iq must have shape [N,9,2]")
    if any(np.asarray(data[k]).shape != (n,) for k in ALLOWED_NPZ_FIELDS[1:]):
        raise ValueError("canonical metadata fields must be aligned [N]")
    if not (np.isfinite(data["complex_iq"]).all() and np.isfinite(data["time_s"]).all()):
        raise ValueError("canonical IQ and timestamps must be finite")
    original = np.arange(n, dtype=np.int64)
    order = np.lexsort((original, data["sample_count"], data["time_s"],
                        data["prn"], data["channel"],
                        data["segment_index"]))
    data = {k: np.asarray(v)[order] for k, v in data.items()}
    audit = {"scenario": str(scenario), "path": str(path), "sha256": actual,
             "rows": n, "shape": list(data["complex_iq"].shape),
             "tap_order": list(TAP_NAMES), "stable_sort":
             ["segment_index", "channel", "prn", "time_s", "sample_count", "original_index"],
             "loaded_fields": list(ALLOWED_NPZ_FIELDS)}
    return data, audit


def clean_role(times: np.ndarray) -> np.ndarray:
    t = np.asarray(times, dtype=float)
    role = np.full(t.shape, "excluded", dtype="U16")
    role[(t >= 0) & (t < 240)] = "train"
    role[(t >= 250) & (t < 330)] = "validation"
    role[(t >= 340) & (t < 410)] = "calibration"
    role[t >= 420] = "clean_test"
    return role


def fit_gate_from_clean_train(data: Mapping[str, np.ndarray]) -> GateEvidence:
    iq = np.asarray(data["complex_iq"])
    role = clean_role(np.asarray(data["time_s"]))
    p = np.hypot(iq[:, 4, 0].astype(np.float64), iq[:, 4, 1].astype(np.float64))
    train = p[role == "train"]
    if not len(train) or not np.isfinite(train).all():
        raise ValueError("finite cleanStatic train Prompt rows required")
    raw = float(np.quantile(train, .005, method="higher"))
    floor = float(np.finfo(np.float64).tiny)
    return GateEvidence(max(raw, floor), raw, floor)


def _features(z: np.ndarray, representation: str) -> np.ndarray:
    if representation == "complex":
        mr = np.median(z.real, axis=0); mi = np.median(z.imag, axis=0)
        return np.stack((mr, mi, np.median(np.abs(z.real - mr), axis=0),
                         np.median(np.abs(z.imag - mi), axis=0)), axis=1).astype("f4")
    mag = np.abs(z); mm = np.median(mag, axis=0)
    return np.stack((mm, np.median(np.abs(mag - mm), axis=0)), axis=1).astype("f4")


def build_feature_bundle(data: Mapping[str, np.ndarray], scenario: str, gate: GateEvidence,
                         *, min_valid_rows: int = 5) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    if int(min_valid_rows) < 1:
        raise ValueError("minimum valid-row rule must be positive and frozen")
    iq = np.asarray(data["complex_iq"]); t = np.asarray(data["time_s"], float)
    z = iq[..., 0].astype(np.float64) + 1j * iq[..., 1].astype(np.float64)
    pmag = np.abs(z[:, 4]); finite = np.isfinite(z.real).all(1) & np.isfinite(z.imag).all(1)
    valid = finite & (pmag >= gate.minimum) & (pmag > 0)
    normalized = np.full((len(z), 8), np.nan + 1j * np.nan, dtype=np.complex128)
    den = pmag[valid] ** 2 + gate.epsilon
    normalized[valid] = z[valid][:, SIDE_INDICES] * np.conjugate(z[valid, 4, None]) / den[:, None]
    roles = clean_role(t) if scenario == "cleanStatic" else np.full(len(t), "attack", dtype="U16")
    rows: list[dict[str, Any]] = []
    keys = np.stack((np.asarray(data["segment_index"]), np.asarray(data["channel"]),
                     np.asarray(data["prn"])), axis=1)
    _, starts = np.unique(keys, axis=0, return_index=True)
    starts = np.sort(starts)
    bounds = list(starts) + [len(t)]
    for a, b in zip(bounds[:-1], bounds[1:]):
        tt = t[a:b]
        first = math.ceil((float(tt.min()) + 1.0) * 2.0 - 1e-12) / 2.0
        last = math.floor(float(tt.max()) * 2.0 + 1e-12) / 2.0
        for end in np.arange(first, last + .25, .5):
            local = np.flatnonzero((tt > end - 1.0) & (tt <= end)) + a
            if not len(local):
                continue
            unique_role = np.unique(roles[local])
            if len(unique_role) != 1 or unique_role[0] == "excluded":
                continue
            good = local[valid[local]]
            if len(good) < int(min_valid_rows):
                continue
            rows.append({"segment_index": data["segment_index"][a], "channel": data["channel"][a],
                         "prn": data["prn"][a], "role": unique_role[0],
                         "source_start": end - 1.0, "source_end": end,
                         "valid_count": len(good), "raw_count": len(local), "ids": good})
    rows.sort(key=lambda r: (float(r["source_end"]), str(r["segment_index"]),
                             str(r["channel"]), str(r["prn"])))
    if not rows:
        raise ValueError(f"{scenario}: no feature windows survived frozen minimum-valid-row rule")
    base = {
        "segment_index": np.asarray([r["segment_index"] for r in rows]),
        "channel": np.asarray([r["channel"] for r in rows]),
        "prn": np.asarray([r["prn"] for r in rows]),
        "role": np.asarray([r["role"] for r in rows]),
        "source_start": np.asarray([r["source_start"] for r in rows], dtype="f8"),
        "source_end": np.asarray([r["source_end"] for r in rows], dtype="f8"),
        "valid_count": np.asarray([r["valid_count"] for r in rows], dtype="i4"),
        "raw_count": np.asarray([r["raw_count"] for r in rows], dtype="i4"),
    }
    bundle = {}
    for rep in ("complex", "magnitude"):
        part = {k: np.array(v, copy=True) for k, v in base.items()}
        part["features"] = np.stack([_features(normalized[r["ids"]], rep) for r in rows])
        bundle[rep] = part
    qa = {"scenario": scenario, "raw_rows": int(len(t)), "valid_rows": int(valid.sum()),
          "rejected_rows": int((~valid).sum()), "zero_prompt_rejected": int(np.sum(pmag == 0)),
          "windows": len(rows), "minimum_valid_rows": int(min_valid_rows),
          "low_prompt_rule": "finite and abs(P)>=max(clean-train q.005 higher, positive float64 floor) and abs(P)>0",
          "gaps_excluded": True, "prompt_is_qa_only": True}
    return bundle, qa


def feature_schema_document(min_valid_rows: int = 5) -> dict[str, Any]:
    return {"schema": "gnss-doppler-lab.amcf-shape-only-feature.v1",
            "tap_order": list(TAP_NAMES), "prompt_index": 4,
            "side_source_indices": list(SIDE_INDICES), "tensor_side_count": 8,
            "representations": {
                "complex": {"dimensions": list(COMPLEX_SCHEMA), "shape": [8, 4]},
                "magnitude": {"dimensions": list(MAGNITUDE_SCHEMA), "shape": [8, 2]}},
            "source_support": "(T-1.0,T]", "endpoint_stride_s": .5,
            "history": "strictly previous 12 at exact 0.5 s cadence within identity and role",
            "minimum_valid_rows": int(min_valid_rows), "literal_mad_multiplier": 1.0,
            "model_tensor_fields": ["side_shape_features"]}


def bind_feature_provenance(scenario: str, input_audit: Mapping[str, Any], gate: GateEvidence,
                            schema: Mapping[str, Any], bundle: Mapping[str, Any],
                            qa: Mapping[str, Any]) -> dict[str, Any]:
    components = {"scenario": scenario, "input_sha256": input_audit["sha256"],
                  "gate": dict(gate._asdict()), "schema_sha256": _digest_value(schema),
                  "bundle_sha256": _digest_value(bundle), "qa_sha256": _digest_value(qa)}
    return {**components, "feature_provenance_sha256": _digest_value(components)}


def verify_feature_provenance(evidence: Mapping[str, Any], input_audit: Mapping[str, Any],
                              gate: GateEvidence, schema: Mapping[str, Any],
                              bundle: Mapping[str, Any], qa: Mapping[str, Any]) -> None:
    actual = bind_feature_provenance(str(evidence.get("scenario")), input_audit, gate, schema, bundle, qa)
    if dict(evidence) != actual:
        raise ValueError("feature provenance mismatch; fit/threshold/GO path is closed")


def build_examples(part: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    n = len(part["features"])
    identity = [(str(part["segment_index"][i]), str(part["channel"][i]), str(part["prn"][i]),
                 str(part["role"][i])) for i in range(n)]
    groups: dict[tuple[str, ...], list[int]] = {}
    for i, key in enumerate(identity): groups.setdefault(key, []).append(i)
    current=[]; history=[]; h_end=[]; role=[]; h_role=[]; source_start=[]; source_end=[]; prn=[]; segment=[]; channel=[]
    for key in sorted(groups):
        ids = sorted(groups[key], key=lambda i: (float(part["source_end"][i]), i))
        ends = np.asarray([part["source_end"][i] for i in ids], float)
        if len(np.unique(ends)) != len(ends):
            raise ValueError("duplicate identity/timestamp feature window")
        for j in range(12, len(ids)):
            use = ids[j-12:j+1]; ee = np.asarray([part["source_end"][i] for i in use], float)
            if not np.allclose(np.diff(ee), .5, rtol=0, atol=1e-9):
                continue
            cur = use[-1]; prev = use[:-1]
            current.append(part["features"][cur]); history.append(part["features"][prev]); h_end.append(ee[:-1])
            role.append(part["role"][cur]); h_role.append([part["role"][i] for i in prev])
            source_start.append(part["source_start"][cur]); source_end.append(part["source_end"][cur]); prn.append(part["prn"][cur]); segment.append(part["segment_index"][cur]); channel.append(part["channel"][cur])
    if not current:
        d = int(np.asarray(part["features"]).shape[-1])
        return {"current": np.empty((0,8,d),"f4"), "history": np.empty((0,12,8,d),"f4"),
                "history_end": np.empty((0,12)), "role": np.empty(0,"U16"),
                "history_role": np.empty((0,12),"U16"), "source_start": np.empty(0),
                "source_end": np.empty(0), "prn": np.empty(0), "segment_index": np.empty(0), "channel": np.empty(0)}
    order = np.lexsort((np.asarray(prn).astype(str), np.asarray(source_end)))
    values = {"current": np.stack(current), "history": np.stack(history), "history_end": np.stack(h_end),
              "role": np.asarray(role), "history_role": np.asarray(h_role),
              "source_start": np.asarray(source_start), "source_end": np.asarray(source_end), "prn": np.asarray(prn),
               "segment_index": np.asarray(segment), "channel": np.asarray(channel)}
    return {k: np.asarray(v)[order] for k,v in values.items()}


def phase_labels(source_start: np.ndarray, source_end: np.ndarray, onset: float) -> dict[str, np.ndarray]:
    start=np.asarray(source_start,float); end=np.asarray(source_end,float)
    if start.shape != end.shape or np.any(end <= start):
        raise ValueError("aligned positive source intervals required")
    return {"stable_pre": (start >= 30.) & (end <= onset-20.),
            "post": start >= onset, "persistent": start >= onset+40.}


def load_b0_exact(path: Path | str, scenario: str) -> dict[str, np.ndarray]:
    with Path(path).open(newline="", encoding="utf-8") as f:
        raw = list(csv.DictReader(f))
    t=[]; score=[]
    for row in raw:
        if row.get("decision_time_s", "") != "" and row.get("score_B0_Exact", "") != "":
            t.append(float(row["decision_time_s"])); score.append(float(row["score_B0_Exact"]))
    if len(set(t)) != len(t):
        raise ValueError(f"{scenario}: duplicate B0 decision timestamp")
    order=np.argsort(t,kind="mergesort")
    return {"decision_time_s": np.asarray(t,float)[order], "score_B0_Exact": np.asarray(score,float)[order]}


def common_timestamp_join(a: Iterable[Mapping[str, Any]], b: Iterable[Mapping[str, Any]],
                          a_key: str, b_key: str) -> list[dict[str, float]]:
    a_rows=list(a); b_rows=list(b)
    aa={float(r["decision_time_s"]):float(r[a_key]) for r in a_rows}
    bb={float(r["decision_time_s"]):float(r[b_key]) for r in b_rows}
    if len(aa) != len(a_rows) or len(bb) != len(b_rows):
        raise ValueError("duplicate comparator timestamp")
    amap={float(r["decision_time_s"]):r for r in a_rows}; bmap={float(r["decision_time_s"]):r for r in b_rows}
    out=[]
    for t in sorted(set(aa).intersection(bb)):
        ar,br=amap[t],bmap[t]
        for field in ("source_start","source_end","phase","label"):
            if field in ar and field in br:
                same=math.isclose(float(ar[field]),float(br[field]),rel_tol=0,abs_tol=1e-12) if field in ("source_start","source_end") else str(ar[field])==str(br[field])
                if not same:raise ValueError(f"common timestamp source interval/label mismatch: {t} {field}")
        row={"decision_time_s":t,"complex":aa[t],"comparator":bb[t]}
        for field in ("source_start","source_end","phase","label"):
            if field in ar: row[field]=ar[field]
        out.append(row)
    return out


def aggregate_prn_scores(scores: Mapping[Any, float]) -> float:
    values=np.asarray(list(scores.values()),float)
    if not len(values) or not np.isfinite(values).all(): raise ValueError("finite PRN scores required")
    return float(np.median(values))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool): return value
    return str(value).strip().lower() in {"1","true","yes"}


def _roc_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    y=np.asarray(y,bool); score=np.asarray(score,float); pos=score[y]; neg=score[~y]
    if not len(pos) or not len(neg): return None
    return float(np.mean((pos[:,None] > neg[None,:]) + .5*(pos[:,None] == neg[None,:])))


def _pr_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    y=np.asarray(y,bool); score=np.asarray(score,float)
    if not y.any() or y.all(): return None
    order=np.argsort(-score,kind="mergesort"); yy=y[order]
    tp=np.cumsum(yy); fp=np.cumsum(~yy); recall=tp/tp[-1]; precision=tp/(tp+fp)
    return float(np.sum((recall-np.r_[0.,recall[:-1]])*precision))


def _sustained_delay(times: np.ndarray, alarms: np.ndarray, stable: np.ndarray,
                     post: np.ndarray, onset: float) -> float | str | None:
    def first(mask: np.ndarray) -> int | None:
        ids=np.flatnonzero(mask)
        for i in range(max(0,len(ids)-2)):
            run=ids[i:i+3]
            if np.array_equal(run,np.arange(run[0],run[0]+3)) and alarms[run].all() and np.allclose(np.diff(times[run]),.5,rtol=0,atol=1e-9):
                return int(run[0])
        return None
    if first(stable) is not None: return "N/A: already alarming in stable-pre"
    found=first(post)
    return None if found is None else float(times[found]-onset)


def recompute_scenario_metrics(scenario: str, rows: Iterable[Mapping[str, Any]], q99: float,
                               q995: float, *, onset_s: float | None) -> dict[str, Any]:
    rows=sorted(list(rows),key=lambda r:float(r["decision_time_s"]))
    if not rows: raise ValueError("nonempty per-epoch rows required")
    t=np.asarray([float(r["decision_time_s"]) for r in rows]); start=np.asarray([float(r["source_start"]) for r in rows]); end=np.asarray([float(r["source_end"]) for r in rows]); score=np.asarray([float(r["score_ensemble"]) for r in rows])
    if len(np.unique(t)) != len(t): raise ValueError("duplicate per-epoch timestamp")
    a99=score>float(q99); a995=score>float(q995)
    if any(_as_bool(r["alarm_q99"]) != bool(a) or _as_bool(r["alarm_q995"]) != bool(b) for r,a,b in zip(rows,a99,a995)):
        raise ValueError("saved alarm does not independently recompute")
    out={"scenario":scenario,"operating_point":"q99","threshold":float(q99),"rows":len(rows)}
    if onset_s is None:
        out.update({"clean_test_fpr":float(a99.mean()),"stable_pre_fpr":None,"post_detection":None,"persistent_detection":None,"roc_auc":None,"pr_auc":None,"sustained3_delay_s":None})
        return out
    masks=phase_labels(start,end,onset_s); selected=masks["stable_pre"]|masks["post"]; labels=masks["post"][selected]
    out.update({"clean_test_fpr":None,
                "stable_pre_fpr":float(a99[masks["stable_pre"]].mean()) if masks["stable_pre"].any() else None,
                "post_detection":float(a99[masks["post"]].mean()) if masks["post"].any() else None,
                "persistent_detection":float(a99[masks["persistent"]].mean()) if masks["persistent"].any() else None,
                "roc_auc":_roc_auc(labels,score[selected]),"pr_auc":_pr_auc(labels,score[selected]),
                "sustained3_delay_s":_sustained_delay(t,a99,masks["stable_pre"],masks["post"],float(onset_s)),
                "roc_definition":"stable-pre label 0 versus wholly-post label 1; transition excluded",
                "phase_definition":"actual source_start/source_end"})
    return out


def gpu_probe(*, torch_module=None, required: bool = True) -> dict[str, Any]:
    if torch_module is None:
        try: import torch as torch_module
        except ImportError as exc:
            if required: raise RuntimeError("CUDA PyTorch is required for a primary run") from exc
            return {"cuda_available":False,"real_tensor_op":False,"device":"cpu","error":str(exc)}
    if not torch_module.cuda.is_available():
        if required: raise RuntimeError("CUDA availability and a real CUDA tensor operation are required")
        return {"cuda_available":False,"real_tensor_op":False,"device":"cpu"}
    try:
        x=torch_module.arange(8,dtype=torch_module.float32,device="cuda"); value=float((x*x).sum().item())
        if value != 140.0: raise RuntimeError("unexpected CUDA tensor result")
        torch_module.cuda.synchronize()
        name=torch_module.cuda.get_device_name(0)
    except Exception as exc:
        if required: raise RuntimeError("real CUDA tensor operation failed") from exc
        return {"cuda_available":True,"real_tensor_op":False,"device":"cuda","error":str(exc)}
    return {"cuda_available":True,"real_tensor_op":True,"device":"cuda","device_name":name,"probe_value":value,
            "torch_version":str(torch_module.__version__),"cuda_runtime":str(torch_module.version.cuda)}


def diff_inventory(repo: Path | str, baseline: str) -> set[str]:
    text=subprocess.check_output(["git","-C",str(repo),"diff","--name-only",f"{baseline}..HEAD"],text=True)
    staged=subprocess.check_output(["git","-C",str(repo),"diff","--name-only","--cached"],text=True)
    unstaged=subprocess.check_output(["git","-C",str(repo),"diff","--name-only"],text=True)
    untracked=subprocess.check_output(["git","-C",str(repo),"ls-files","--others","--exclude-standard"],text=True)
    return {x for x in (text+staged+unstaged+untracked).splitlines() if x}


def verify_primary_source_state(repo: Path | str, baseline: str,
                                *, allowed_files: set[str] | frozenset[str] = SHAPE_ONLY_ALLOWED_FILES) -> dict[str, Any]:
    repo=Path(repo)
    resolved=subprocess.check_output(["git","-C",str(repo),"rev-parse",baseline],text=True).strip()
    dirty=subprocess.check_output(["git","-C",str(repo),"status","--porcelain"],text=True)
    if dirty.strip(): raise RuntimeError("primary run refuses a dirty source tree")
    inventory=diff_inventory(repo,baseline)
    forbidden=inventory-set(allowed_files)
    if forbidden: raise RuntimeError(f"protected baseline diff inventory violation: {sorted(forbidden)}")
    commit=subprocess.check_output(["git","-C",str(repo),"rev-parse","HEAD"],text=True).strip()
    return {"source_commit":commit,"clean_tree":True,"baseline":baseline,"diff_inventory":sorted(inventory),
            "source_hashes":{p:sha256(repo/p) for p in SOURCE_FILES if (repo/p).is_file()}}


def write_hashes(out: Path | str) -> dict[str, str]:
    out=Path(out); files={str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name!="hashes.json"}
    write_json(out/"hashes.json",{"algorithm":"sha256","excludes":["hashes.json"],"files":files})
    return files


def verify_hashes(out: Path | str) -> bool:
    out=Path(out)
    try: manifest=json.loads((out/"hashes.json").read_text(encoding="utf-8"))
    except Exception as exc: raise ValueError("hash manifest missing or invalid") from exc
    expected=manifest.get("files",{}); actual_files={str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and p.name!="hashes.json"}
    if set(expected) != actual_files: raise ValueError("hash inventory does not cover every artifact file")
    for rel,digest in expected.items():
        if sha256(out/rel) != digest: raise ValueError(f"hash mismatch: {rel}")
    return True


def _save_bundle(path: Path, part: Mapping[str,np.ndarray]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez(path, **{k:np.asarray(v) for k,v in sorted(part.items())})


def _load_saved_bundle(path: Path, expected_keys: Iterable[str]) -> dict[str,np.ndarray]:
    with np.load(path,allow_pickle=False) as f:
        return {k:np.array(f[k],copy=True) for k in sorted(expected_keys)}


def _smoke_feature_part(scenario: str, rep: str, seed: int) -> dict[str,np.ndarray]:
    rng=np.random.default_rng(seed + sum(map(ord,scenario)) + (0 if rep=="complex" else 1000)); d=4 if rep=="complex" else 2
    n=42; end=np.arange(n)*.5+7.; onset=ONSETS.get(scenario)
    if onset is not None:
        end=np.r_[np.arange(30.,36.,.5),np.arange(onset+1.,onset+7.,.5),np.arange(onset+41.,onset+50.,.5)]
        n=len(end)
    return {"features":rng.normal(size=(n,8,d)).astype("f4"),"segment_index":np.zeros(n,"i4"),"channel":np.zeros(n,"i2"),
            "prn":np.resize(np.asarray([3,7,11],"i2"),n),"role":np.full(n,"attack" if scenario!="cleanStatic" else "clean_test"),
            "source_start":end-1.,"source_end":end,"valid_count":np.full(n,12,"i4"),"raw_count":np.full(n,20,"i4")}


def _smoke_epoch_rows(scenario: str) -> list[dict[str,Any]]:
    onset=ONSETS.get(scenario)
    if onset is None:
        times=np.arange(420.,426.,.5)
    else:
        times=np.r_[np.arange(30.,32.,.5),np.arange(onset+1.,onset+2.5,.5),np.arange(onset+41.,onset+42.,.5)]
    rows=[]
    for i,t in enumerate(times):
        if onset is None: phase="clean_test"; attack=False
        elif t < onset-20: phase="stable_pre"; attack=False
        elif t >= onset+40: phase="persistent"; attack=True
        else: phase="post"; attack=True
        c=.1+.1*i+(3.0 if attack else 0.); m=.15+.08*i+(2.0 if attack else 0.); b=.05+.05*i+(1.4 if attack else 0.)
        row={"scenario":scenario,"decision_time_s":float(t),"source_start":float(t),"source_end":float(t+1.),"phase":phase,
             "tracked_prn_count":3,"label":int(attack)}
        for prefix,value in (("complex",c),("magnitude",m),("complex_epl",c-.2),("magnitude_epl",m-.2)):
            for j,s in enumerate(SEEDS): row[f"score_{prefix}_seed{s}"]=value+.01*j
            row[f"p_{prefix}_seed101"]=.5 if not attack else .01; row[f"e_{prefix}_seed101"]=-math.log(row[f"p_{prefix}_seed101"])
            row[f"score_{prefix}_ensemble"]=value+.01
        row.update({"score_B0_Exact":b,"alarm_complex_q99":c+.01>1.,"alarm_complex_q995":c+.01>4.5,
                    "alarm_magnitude_q99":m+.01>1.2,"alarm_magnitude_q995":m+.01>3.8,
                    "alarm_B0_q99":b>.8,"alarm_B0_q995":b>2.5})
        rows.append(row)
    return rows


def render_readme(decision: str, status: str, criteria: Mapping[str,Any]) -> str:
    lines=["# AMCF-R1 Shape-Only campaign", "", f"**Primary q99 decision: {decision}**", f"Status: {status}.", "",
           "All DS1/DS2/DS3/DS7/DS8 attack results are exploratory/developmental and post-exposure.",
           "q99 is the sole primary operating point; q995 is diagnostic and cannot affect GO.",
           "AMCF thresholds use cleanStatic [340,410); frozen B0 Exact uses its protected [300,330) calibration.",
           "Because those protocols differ, threshold-dependent AMCF/B0 comparison is limited; B0 ROC-AUC alone is recomputed on common timestamps.",
           "DS4 is explicit NA metadata and is not one of the five attack GO scenarios.", "", "## Verified criteria"]
    for name in sorted(criteria): lines.append(f"- {name}: {'PASS' if bool(criteria[name]) else 'FAIL'}")
    lines += ["", "## Claims", "- Claimable: leakage controls, deterministic artifact recomputation, and the recorded smoke/campaign outcome.",
              "- Not claimable: confirmatory attack performance, independent-clean generalization, causality, or deployment benefit.",
              "- Any core criterion failure means NO-GO and AMCF WCL no-go.", ""]
    return "\n".join(lines)


def _smoke_criteria() -> dict[str,bool]:
    return {"all_required_seeds_converged":False,"stable_pre_fpr_all_below_0.05":True,
            "complex_auc_gt_magnitude_4_of_5":False,"auc_bootstrap_ci_lower_gt_zero_3_of_5":False,
            "same_seed_direction_each_scenario":False,"beats_b0_with_fpr_guard_3_of_5":False,
            "ds7_ds8_no_collapse":True}


def run_smoke(out: Path | str, *, fixture_seed: int = 7) -> dict[str,Any]:
    out=Path(out).resolve(); final=(ROOT/FINAL_ARTIFACT).resolve()
    if out == final: raise ValueError("smoke output must be outside final artifact path")
    if out.exists(): raise FileExistsError(out)
    stage=out.with_name(out.name+f".tmp-{os.getpid()}"); stage.mkdir(parents=True)
    try:
        for d in ("per_epoch","plots","models","feature_cache"): (stage/d).mkdir()
        schema=feature_schema_document(5); write_json(stage/"feature_schema.json",schema)
        features={}; input_hashes={}
        for scenario in ("cleanStatic",*ONSETS):
            input_hashes[scenario]={"synthetic":True,"fixture_seed":fixture_seed,"sha256":hashlib.sha256(f"{scenario}:{fixture_seed}".encode()).hexdigest(),"tap_order":list(TAP_NAMES)}
            for rep in ("complex","magnitude"):
                part=_smoke_feature_part(scenario,rep,fixture_seed); rel=f"feature_cache/{scenario}_{rep}.npz"; _save_bundle(stage/rel,part); features[f"{scenario}:{rep}"]={"file":rel,"tensor_sha256":_digest_value(part)}
        write_json(stage/"input_hashes.json",input_hashes)
        write_json(stage/"window_qa.json",{name:{"scenario":name,"synthetic":True,"fixture_seed":fixture_seed,"minimum_valid_rows":5} for name in ("cleanStatic",*ONSETS)})
        gate=GateEvidence(.01,.01,float(np.finfo(np.float64).tiny)); source_commit=subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip()
        provenance={"schema":"gnss-doppler-lab.amcf-shape-only-provenance.v1","mode":"synthetic-smoke","source_commit":source_commit,
                    "clean_tree_required":False,"execution_source_hashes":{p:sha256(ROOT/p) for p in SOURCE_FILES if (ROOT/p).is_file()},
                    "input_hashes_digest":_digest_value(input_hashes),"gate":dict(gate._asdict()),"gate_scaler_hash":"smoke-"+_digest_value(dict(gate._asdict())),
                    "feature_schema_hash":_digest_value(schema),"features":features,"roles":{"train":[0,240],"validation":[250,330],"calibration":[340,410],"clean_test":[420,None]},
                    "causal_qa":{"history":12,"cadence_s":.5,"source_support":"(T-1,T]","pass":True},"attack_fit":False}
        write_json(stage/"provenance.json",provenance)
        config={"schema":"gnss-doppler-lab.amcf-shape-only-config.v1","mode":"synthetic-smoke","primary":False,"seeds":list(SEEDS),
                "representations":["Complex all9","Magnitude all9","Complex EPL","Magnitude EPL"],"history":12,"stride_s":.5,"source_window_s":1.,
                "minimum_valid_rows":5,"prompt_quantile":.005,"prompt_epsilon":1e-12,"hidden":32,"batch_size":256,"learning_rate":1e-3,
                "max_epochs":200,"patience":20,"bootstrap_reps":2000,"final_artifact":str(FINAL_ARTIFACT),"no_attack_retune":True,
                "scenarios":[*ONSETS,"DS4:NA"],"DS4":{"status":"NA","included_in_attack_go":False}}
        write_json(stage/"config.json",config)
        histories=[]; audits={}
        validation_hash=hashlib.sha256(b"fixed-smoke-validation-sample-target-identities").hexdigest()
        for rep in ("complex","magnitude"):
            for objective in ("all9","EPL"):
                for seed in SEEDS:
                    key=f"{rep}_{objective}_seed{seed}"; histories.append({"representation":rep,"objective":objective,"seed":seed,"epoch":0,"train_loss":1.0,"validation_loss":1.0,"optimizer_updates":2,"tuple_batch_size":256,"validation_bank_hash":validation_hash})
                    model=stage/"models"/f"{key}.pt"; model.write_bytes(f"SMOKE CHECKPOINT {key}\n".encode())
                    audits[key]={"seed":seed,"representation":rep,"objective":objective,"finite":True,"optimizer_updates":2,"best_epoch":0,"patience_early_stop":False,"converged":False,"excluded_from_ensemble":True,"checkpoint_sha256":sha256(model),"validation_bank_hash":validation_hash}
        write_csv(stage/"training_history.csv",histories); write_json(stage/"convergence_audit.json",{"primary_status":"INCOMPLETE: nonconverged smoke models","audits":audits,"exact_three_converged_required":True})
        thresholds={"comparison":"strict_greater","primary":"q99","q995_role":"diagnostic_only","Complex all9":{"q99":1.,"q995":4.5},"Magnitude all9":{"q99":1.2,"q995":3.8},"B0 Exact":{"q99":.8,"q995":2.5},"calibration":"cleanStatic only; synthetic smoke"}
        write_json(stage/"thresholds.json",thresholds)
        metric_rows=[]; seed_rows=[]
        for scenario in ("cleanStatic",*ONSETS):
            rows=_smoke_epoch_rows(scenario); write_csv(stage/"per_epoch"/f"{scenario}.csv",rows)
            generic=[{"decision_time_s":r["decision_time_s"],"source_start":r["source_start"],"source_end":r["source_end"],"score_ensemble":r["score_complex_ensemble"],"alarm_q99":r["alarm_complex_q99"],"alarm_q995":r["alarm_complex_q995"]} for r in rows]
            primary_metric={"model":"Complex all9",**recompute_scenario_metrics(scenario,generic,1.,4.5,onset_s=ONSETS.get(scenario)),"source_file":f"per_epoch/{scenario}.csv"}; metric_rows.append(primary_metric)
            diagnostic=[dict(x,alarm_q99=x["alarm_q995"]) for x in generic]
            diag_metric={"model":"Complex all9",**recompute_scenario_metrics(scenario,diagnostic,4.5,4.5,onset_s=ONSETS.get(scenario)),"source_file":f"per_epoch/{scenario}.csv","operating_point":"q995","diagnostic_only":True}; metric_rows.append(diag_metric)
            for seed in SEEDS: seed_rows.append({"scenario":scenario,"model":"Complex all9","seed":seed,"roc_auc":primary_metric.get("roc_auc"),"converged":False,"excluded":True})
        metric_rows.append(ds4_na_row())
        write_csv(stage/"scenario_metrics.csv",metric_rows); write_csv(stage/"seed_metrics.csv",_append_seed_mean_std(seed_rows))
        paired=[]
        for scenario in ONSETS:
            for comparator in ("Magnitude all9","B0 Exact"):
                for metric in ("roc_auc","post_detection","stable_pre_fpr"):
                    paired.append({"scenario":scenario,"comparator":comparator,"metric":metric,"delta_definition":"Complex-comparator","estimate":0.,"ci_low":0.,"ci_high":0.,"reps":2000,"block_s":10.,"paired_on":"same timestamp/label"})
        write_csv(stage/"paired_comparisons.csv",paired)
        smoke_checks={}
        for scenario in ("cleanStatic","DS7","DS8"):
            reps={}
            for rep,d in (("complex",4),("magnitude",2)):
                rel=f"feature_cache/{scenario}_{rep}.npz"; part=_load_saved_bundle(stage/rel,("features","segment_index","channel","prn","role","source_start","source_end","valid_count","raw_count")); x=part["features"]; iq=np.quantile(x,.75,axis=0)-np.quantile(x,.25,axis=0)
                reps[rep]={"file":rel,"bundle_digest":_digest_value(part),"tensor_digest":_digest_value(x),"schema_hash":_digest_value(schema),"iqr":iq.tolist(),"pass":bool(x.shape[1:]==(8,d) and np.isfinite(x).all() and np.all(iq>1e-8))}
            rows=_smoke_epoch_rows(scenario); stable=rows if scenario=="cleanStatic" else [r for r in rows if r["phase"]=="stable_pre"]
            alarms=[(r["decision_time_s"],r["source_start"],r["source_end"],r["score_complex_ensemble"],1.,r["alarm_complex_q99"]) for r in stable]
            smoke_checks[scenario]={"representations":reps,"stable_alarm_evidence_digest":_digest_value(alarms),"stable_pre_alarm_rate":float(np.mean([x[-1] for x in alarms])) if alarms else 0.,"pass":all(x["pass"] for x in reps.values())}
        smoke_binding={"canonical_fields":list(ALLOWED_NPZ_FIELDS),"schema_hash":_digest_value(schema),"scenarios":smoke_checks}
        write_json(stage/"feature_audit.json",{**smoke_binding,"pass":all(smoke_checks[n]["pass"] for n in ("DS7","DS8")),"audit_digest":_digest_value(smoke_binding)})
        criteria=_smoke_criteria(); decision={"primary_quantile":"q99","primary_decision":"NO-GO","amcf_wcl":"AMCF WCL no-go","criteria":criteria,"source_digests":{"metrics":sha256(stage/"scenario_metrics.csv"),"paired":sha256(stage/"paired_comparisons.csv"),"convergence":sha256(stage/"convergence_audit.json"),"schema":sha256(stage/"feature_schema.json"),"feature_audit":sha256(stage/"feature_audit.json")}}
        write_json(stage/"decision.json",decision); (stage/"README.md").write_text(render_readme("NO-GO","SMOKE-NO-GO",criteria),encoding="utf-8",newline="\n")
        (stage/"plots"/"SMOKE_ONLY.txt").write_text("Synthetic smoke only; no attack campaign executed.\n")
        write_hashes(stage)
        missing=[x for x in REQUIRED_INVENTORY if not (stage/x).exists()]
        if missing: raise RuntimeError(f"missing required artifact inventory: {missing}")
        verify_hashes(stage); os.replace(stage,out)
        return {"status":"SMOKE-NO-GO","out":str(out),"files":len(json.loads((out/"hashes.json").read_text())["files"])}
    except Exception:
        shutil.rmtree(stage,ignore_errors=True); raise



def index_metric_rows(rows: Iterable[Mapping[str,Any]]) -> dict[tuple[str,str,str],dict[str,Any]]:
    """Index metrics without permitting a diagnostic row to shadow q99."""
    out={}
    for raw in rows:
        row=dict(raw); op=str(row.get("operating_point") or "")
        if op not in {"q99","q995"}: raise ValueError("explicit q99/q995 operating_point required")
        if op=="q995" and row.get("diagnostic_only") not in (True,"true","True","1"):
            raise ValueError("q995 must be diagnostic_only")
        key=(str(row.get("scenario")),str(row.get("model")),op)
        if key in out: raise ValueError(f"duplicate metric key {key}")
        out[key]=row
    return out


def select_primary_metric(index: Mapping[tuple[str,str,str],Mapping[str,Any]], scenario: str, model: str) -> Mapping[str,Any]:
    key=(scenario,model,"q99")
    if key not in index: raise ValueError(f"exact q99 primary metric missing: {scenario} {model}")
    return index[key]


def _seed_key(mapping: Mapping[Any,Any], seed: int) -> Any:
    if seed in mapping:return seed
    if str(seed) in mapping:return str(seed)
    raise ValueError(f"exact three seed evidence missing seed {seed}")


def build_calibration_evidence(variant: str, seed_rows: Mapping[Any,Iterable[Mapping[str,Any]]], *,
                               source_commit: str, score_bank_hash: str, index_hash: str) -> dict[str,Any]:
    """Persist sufficient raw evidence; exact row identity/order equality is mandatory."""
    if {int(k) for k in seed_rows} != set(SEEDS): raise ValueError("exact three seed calibration banks required")
    normalized={}
    reference=None
    for seed in SEEDS:
        rows=[dict(r) for r in seed_rows[_seed_key(seed_rows,seed)]]
        identity=[(float(r["decision_time_s"]),float(r["source_start"]),float(r["source_end"]),str(r["identity_hash"])) for r in rows]
        if len(set(identity))!=len(identity): raise ValueError("exact seed calibration identity has duplicates")
        if reference is None: reference=identity
        elif identity != reference: raise ValueError("exact seed calibration timestamp/identity/order mismatch")
        normalized[seed]=rows
    if not reference: raise ValueError("exact seed calibration evidence cannot be empty")
    raw={str(seed):[float(r["score"]) for r in normalized[seed]] for seed in SEEDS}
    loo={str(seed):_loo_evidence(np.asarray(raw[str(seed)],float)).tolist() for seed in SEEDS}
    ensemble=np.mean(np.asarray([loo[str(s)] for s in SEEDS],float),axis=0)
    common={"decision_time_s":[x[0] for x in reference],"source_start":[x[1] for x in reference],
            "source_end":[x[2] for x in reference],"identity":[x[3] for x in reference]}
    binding={"variant":variant,"source_commit":source_commit,"score_bank_hash":score_bank_hash,
             "index_hash":index_hash,"common_rows":common,"per_seed_raw_scores":raw}
    loo_p={str(seed):np.exp(-np.asarray(loo[str(seed)],float)).tolist() for seed in SEEDS}
    result={**binding,"common_identity":common["identity"],"per_seed_loo_p":loo_p,"per_seed_loo_evidence":loo,
            "ensemble_loo_evidence":ensemble.tolist(),"q99":_higher(ensemble,.99),
            "q995":_higher(ensemble,.995),"quantile_method":"higher"}
    result["threshold_digest"]=_digest_value(binding)
    return result


def recompute_calibration_evidence(evidence: Mapping[str,Any]) -> dict[str,Any]:
    required={"variant","source_commit","score_bank_hash","index_hash","common_rows","common_identity",
              "per_seed_raw_scores","per_seed_loo_p","per_seed_loo_evidence","ensemble_loo_evidence","q99","q995","threshold_digest"}
    if not required <= set(evidence): raise ValueError("calibration evidence fields missing")
    rows=evidence["common_rows"]; n=len(rows["decision_time_s"])
    if evidence["common_identity"] != rows["identity"] or any(len(rows[k])!=n for k in rows):
        raise ValueError("calibration common identity/order mismatch")
    raw=evidence["per_seed_raw_scores"]
    if {int(k) for k in raw}!=set(SEEDS) or any(len(raw[_seed_key(raw,s)])!=n for s in SEEDS):
        raise ValueError("exact three seed calibration row mismatch")
    loo={str(s):_loo_evidence(np.asarray(raw[_seed_key(raw,s)],float)).tolist() for s in SEEDS}
    ensemble=np.mean(np.asarray([loo[str(s)] for s in SEEDS],float),axis=0)
    binding={k:evidence[k] for k in ("variant","source_commit","score_bank_hash","index_hash","common_rows","per_seed_raw_scores")}
    actual={"q99":_higher(ensemble,.99),"q995":_higher(ensemble,.995),"threshold_digest":_digest_value(binding)}
    loo_p={str(seed):np.exp(-np.asarray(loo[str(seed)],float)).tolist() for seed in SEEDS}
    if loo != evidence["per_seed_loo_evidence"] or loo_p != evidence["per_seed_loo_p"] or not np.array_equal(ensemble,np.asarray(evidence["ensemble_loo_evidence"],float)):
        raise ValueError("calibration LOO p/e evidence mismatch")
    if actual["q99"]!=float(evidence["q99"]) or actual["q995"]!=float(evidence["q995"]) or actual["threshold_digest"]!=evidence["threshold_digest"]:
        raise ValueError("calibration threshold digest/scalar mismatch")
    return actual


def load_frozen_b0_contract(artifact: Path|str, *, expected_hashes: Mapping[str,str]=B0_CONTRACT_HASHES) -> dict[str,Any]:
    root=Path(artifact); required=("thresholds.json","config.json","provenance.json")
    if set(expected_hashes)!=set(required): raise ValueError("B0 protected contract requires exact hash inventory")
    for name in required:
        if sha256(root/name)!=expected_hashes[name]: raise ValueError(f"B0 protected hash mismatch: {name}")
    th=json.loads((root/"thresholds.json").read_text()); cfg=json.loads((root/"config.json").read_text()); prov=json.loads((root/"provenance.json").read_text())
    row=th.get("primary",{}).get("B0-Exact",{})
    if float(row.get("q99",math.nan))!=B0_FROZEN_Q99 or float(row.get("q995",math.nan))!=B0_FROZEN_Q995:
        raise ValueError("B0 frozen q99/q995 contract mismatch")
    if cfg.get("threshold_fit")!="cleanStatic [300,330) only": raise ValueError("B0 frozen source role mismatch")
    return {"q99":B0_FROZEN_Q99,"q995":B0_FROZEN_Q995,"fit_interval":list(B0_FROZEN_INTERVAL),
            "source_role":"cleanStatic [300,330)","protected_hashes":dict(expected_hashes),
            "protected_source_commit":prov.get("source_commit"),"contract_digest":_digest_value({"hashes":expected_hashes,"row":row,"role":cfg["threshold_fit"]})}


class FitManifest(NamedTuple):
    upstream_digests: Mapping[str,str]
    manifest_digest: str


def _example_component_digests(examples: Mapping[str,np.ndarray]) -> dict[str,str]:
    required=("current","history","source_start","source_end","prn","segment_index","channel","role")
    if any(k not in examples for k in required): raise ValueError("fit manifest examples missing current/history/identity/source interval")
    return {"examples_current_hash":_digest_value(np.asarray(examples["current"])),
            "examples_history_hash":_digest_value(np.asarray(examples["history"])),
            "examples_identity_hash":_digest_value({k:np.asarray(examples[k]) for k in ("prn","segment_index","channel","role")}),
            "examples_source_interval_hash":_digest_value({"s":np.asarray(examples["source_start"]),"e":np.asarray(examples["source_end"])})}


def make_fit_manifest(examples: Mapping[str,np.ndarray], *, canonical_input_hash: str, prompt_gate_hash: str,
                      scaler_hash: str, feature_tensor_hash: str, validation_bank_hash: str,
                      fit_config: Mapping[str,Any], source_interval_hash: str) -> FitManifest:
    components={"canonical_input_hash":canonical_input_hash,"prompt_gate_hash":prompt_gate_hash,
                "scaler_hash":scaler_hash,"transformed_feature_tensor_hash":feature_tensor_hash,
                "validation_bank_hash":validation_bank_hash,"fit_config_hash":_digest_value(dict(fit_config)),
                **_example_component_digests(examples)}
    if components["examples_source_interval_hash"]!=source_interval_hash:
        raise ValueError("fit manifest source interval prebinding mismatch")
    return FitManifest(MappingProxyType(dict(components)),_digest_value(components))


def verify_fit_manifest(manifest: FitManifest, examples: Mapping[str,np.ndarray]) -> None:
    if not isinstance(manifest,FitManifest): raise TypeError("arbitrary dict training input forbidden; FitManifest required")
    actual=dict(manifest.upstream_digests); actual.update(_example_component_digests(examples))
    # update above intentionally replaces all example digests with actual bytes
    if _digest_value(actual)!=manifest.manifest_digest:
        raise ValueError("fit manifest chain mismatch")


def verify_checkpoint_metadata(path: Path|str, manifest: FitManifest) -> dict[str,Any]:
    import torch
    try: obj=torch.load(path,map_location="cpu",weights_only=False)
    except Exception as exc: raise ValueError("checkpoint metadata cannot be loaded") from exc
    if obj.get("fit_manifest_digest")!=manifest.manifest_digest or obj.get("upstream_digests")!=dict(manifest.upstream_digests):
        raise ValueError("checkpoint metadata upstream digest mismatch")
    return obj


def weighted_validation_loss(chunks: Iterable[tuple[float,int]]) -> float:
    rows=[(float(mean),int(count)) for mean,count in chunks]
    total=sum(count for _,count in rows)
    if total<=0 or any(count<=0 or not math.isfinite(mean) for mean,count in rows): raise ValueError("finite nonempty validation chunks required")
    return float(sum(mean*count for mean,count in rows)/total)


def require_finite_trainable_gradients(parameters: Iterable[Any]) -> None:
    for p in parameters:
        if not getattr(p,"requires_grad",False): continue
        grad=getattr(p,"grad",None)
        if grad is None: raise RuntimeError("every trainable parameter must have a gradient in every update")
        try:
            import torch
            finite=bool(torch.isfinite(grad).all().item())
        except Exception: finite=bool(np.isfinite(np.asarray(grad)).all())
        if not finite: raise RuntimeError("every trainable parameter must have a finite gradient in every update")


def ds4_na_row() -> dict[str,Any]:
    return {"scenario":"DS4","model":"NA","operating_point":"NA","status":"NA",
            "reason":"mixed-producer dataset is outside the five-scenario Shape-Only attack GO contract",
            "included_in_attack_go":False}



class AuditedFitInput(NamedTuple):
    train: Mapping[str,np.ndarray]
    validation: Mapping[str,np.ndarray]
    train_manifest: FitManifest
    validation_manifest: FitManifest
    validation_bank_hash: str
    fit_manifest_digest: str


def make_audited_fit_input(train: Mapping[str,np.ndarray], validation: Mapping[str,np.ndarray], *,
                           canonical_input_hash: str, prompt_gate_hash: str, scaler_hash: str,
                           transformed_feature_tensor_hash: str, objective: str, seed: int) -> AuditedFitInput:
    _,bank_hash=_target_bank(len(validation["current"]),objective,seed)
    kwargs={"canonical_input_hash":canonical_input_hash,"prompt_gate_hash":prompt_gate_hash,
            "scaler_hash":scaler_hash,"feature_tensor_hash":transformed_feature_tensor_hash,
            "validation_bank_hash":bank_hash,"fit_config":PRIMARY_FIT_CONFIG}
    tm=make_fit_manifest(train,source_interval_hash=_example_component_digests(train)["examples_source_interval_hash"],**kwargs)
    vm=make_fit_manifest(validation,source_interval_hash=_example_component_digests(validation)["examples_source_interval_hash"],**kwargs)
    digest=_digest_value({"train":tm.manifest_digest,"validation":vm.manifest_digest,"objective":objective,"seed":seed})
    return AuditedFitInput(train,validation,tm,vm,bank_hash,digest)


def verify_audited_fit_input(value: AuditedFitInput, objective: str, seed: int) -> None:
    if not isinstance(value,AuditedFitInput): raise TypeError("arbitrary dict direct training forbidden; AuditedFitInput required")
    verify_fit_manifest(value.train_manifest,value.train); verify_fit_manifest(value.validation_manifest,value.validation)
    _,vh=_target_bank(len(value.validation["current"]),objective,seed)
    actual=_digest_value({"train":value.train_manifest.manifest_digest,"validation":value.validation_manifest.manifest_digest,"objective":objective,"seed":seed})
    if vh!=value.validation_bank_hash or actual!=value.fit_manifest_digest: raise ValueError("audited fit input chain mismatch")


def _fit_scaler(train_features: np.ndarray) -> dict[str,Any]:
    x=np.asarray(train_features,float); med=np.median(x,axis=0); iqr=np.quantile(x,.75,axis=0,method="linear")-np.quantile(x,.25,axis=0,method="linear")
    failed=np.argwhere(iqr<=1e-8)
    if len(failed): raise ValueError(f"clean-train IQR collapse at {failed.tolist()}")
    return {"median":med,"iqr":iqr,"hash":_digest_value({"median":med,"iqr":iqr})}


def _transform_bundle(bundle: Mapping[str,dict[str,np.ndarray]], scalers: Mapping[str,Any]) -> dict[str,dict[str,np.ndarray]]:
    out={}
    for rep,part in bundle.items():
        q={k:np.array(v,copy=True) for k,v in part.items()}; q["features"]=((q["features"]-scalers[rep]["median"])/scalers[rep]["iqr"]).astype("f4"); out[rep]=q
    return out


def _target_bank(n: int, objective: str, seed: int) -> tuple[np.ndarray,str]:
    targets=range(8) if objective=="all9" else (3,4)
    bank=np.asarray([(i,j) for i in range(n) for j in targets],dtype="i8")
    rng=np.random.default_rng(seed); bank=bank[rng.permutation(len(bank))]
    return bank,hashlib.sha256(bank.astype("<i8",copy=False).tobytes()).hexdigest()


def _torch_loss(model, history, current, pairs, objective: str, core, torch):
    si=torch.as_tensor(pairs[:,0],dtype=torch.long,device=current.device); ti=torch.as_tensor(pairs[:,1],dtype=torch.long,device=current.device)
    hh=history[si]; cc=current[si]; mask=torch.ones((len(pairs),8),dtype=torch.bool,device=current.device)
    if objective=="all9": mask[torch.arange(len(pairs),device=current.device),ti]=False
    else:
        mask.zero_(); mask[ti==3,4]=True; mask[ti==4,3]=True
    loc,scale=model(hh,cc,mask); target=cc[torch.arange(len(pairs),device=current.device),ti]
    return core.student_t_nll(target,loc,scale,model.df).mean()


def _train_primary_model(fit_input: AuditedFitInput, rep: str, objective: str, seed: int,
                         model_path: Path) -> tuple[Any,dict[str,Any],list[dict[str,Any]]]:
    """Audited primary fit path; raw dictionaries are rejected at this boundary."""
    import copy
    verify_audited_fit_input(fit_input,objective,seed)
    train,validation=fit_input.train,fit_input.validation
    sys.path.insert(0,str(ROOT/"src")); import torch
    from gnss_doppler_lab import amcf_shape_only as core
    device=torch.device("cuda"); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    model=core.ShapeOnlyModel(train["current"].shape[-1],hidden=32).to(device)
    if float(model.df)!=4.0: raise RuntimeError("primary Student-t df must be exactly 4")
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    tc=torch.as_tensor(train["current"],device=device); th=torch.as_tensor(train["history"],device=device)
    vc=torch.as_tensor(validation["current"],device=device); vh=torch.as_tensor(validation["history"],device=device)
    train_bank,_=_target_bank(len(tc),objective,seed); val_bank,val_hash=_target_bank(len(vc),objective,seed)
    if val_hash!=fit_input.validation_bank_hash: raise RuntimeError("fixed validation target bank binding mismatch")
    rng=np.random.default_rng(seed); best=math.inf; best_state=None; best_opt=None; best_epoch=-1; stale=0; stopped=False; finite=True; updates=0; history_rows=[]; gradient_updates=0
    for epoch in range(int(PRIMARY_FIT_CONFIG["max_epochs"])):
        model.train(); order=rng.permutation(len(train_bank)); train_chunks=[]; epoch_updates=0; batches=[]
        for a in range(0,len(order),int(PRIMARY_FIT_CONFIG["tuple_batch_size"])):
            pairs=train_bank[order[a:a+int(PRIMARY_FIT_CONFIG["tuple_batch_size"])]]; batches.append(len(pairs)); optimizer.zero_grad(set_to_none=True)
            loss=_torch_loss(model,th,tc,pairs,objective,core,torch)
            if not torch.isfinite(loss): finite=False; break
            loss.backward()
            require_finite_trainable_gradients(model.parameters()); gradient_updates+=1
            optimizer.step(); updates+=1; epoch_updates+=1; train_chunks.append((float(loss.item()),len(pairs)))
        if not finite: break
        if epoch_updates != math.ceil(len(train_bank)/int(PRIMARY_FIT_CONFIG["tuple_batch_size"])): raise RuntimeError("tuple minibatch optimizer update audit failed")
        model.eval(); val_chunks=[]
        with torch.no_grad():
            size=int(PRIMARY_FIT_CONFIG["validation_chunk_size"])
            for a in range(0,len(val_bank),size):
                part=val_bank[a:a+size]; val_chunks.append((float(_torch_loss(model,vh,vc,part,objective,core,torch).item()),len(part)))
        val=weighted_validation_loss(val_chunks)
        row={"representation":rep,"objective":objective,"seed":seed,"epoch":epoch,
             "train_loss":weighted_validation_loss(train_chunks),"validation_loss":val,
             "validation_tuple_count":len(val_bank),"validation_chunk_weighting":"exact tuple sum/count",
             "optimizer":"AdamW","df":4.0,"optimizer_updates":epoch_updates,
             "optimizer_updates_cumulative":updates,"all_trainable_gradients_finite_each_update":True,
             "tuple_batch_sizes":batches,"validation_bank_hash":val_hash,"fit_manifest_digest":fit_input.fit_manifest_digest}
        history_rows.append(row)
        if val < best-float(PRIMARY_FIT_CONFIG["early_stop_tolerance"]):
            best=val; best_epoch=epoch; best_state=copy.deepcopy(model.state_dict()); best_opt=copy.deepcopy(optimizer.state_dict()); stale=0
        else:
            stale+=1
            if stale>=int(PRIMARY_FIT_CONFIG["patience"]): stopped=True; break
    if best_state is not None: model.load_state_dict(best_state); optimizer.load_state_dict(best_opt)
    converged=bool(stopped and finite and updates>1 and gradient_updates==updates)
    upstream={"train_manifest":fit_input.train_manifest.manifest_digest,
              "validation_manifest":fit_input.validation_manifest.manifest_digest,
              **dict(fit_input.train_manifest.upstream_digests)}
    audit={"seed":seed,"representation":rep,"objective":objective,"finite":finite,"epochs_run":len(history_rows),
           "optimizer":"AdamW","df":4.0,"optimizer_updates":updates,"gradient_audited_updates":gradient_updates,
           "every_trainable_parameter_finite_gradient_each_update":gradient_updates==updates,
           "best_epoch":best_epoch,"best_validation_loss":best,"patience_early_stop":stopped,"early_stop_tolerance":1e-12,
           "converged":converged,"excluded_from_ensemble":not converged,"validation_bank_hash":val_hash,
           "fit_manifest_digest":fit_input.fit_manifest_digest,"upstream_digests":upstream,
           "best_checkpoint_restored":best_state is not None,"best_optimizer_restored":best_opt is not None,
           "stop_reason":"patience_converged" if converged else ("nonfinite" if not finite else "cap_nonconverged")}
    torch.save({"checkpoint_role":"amcf-shape-only-primary-real-torch","state_dict":model.state_dict(),
                "optimizer":optimizer.state_dict(),"audit":audit,"feature_dim":train["current"].shape[-1],
                "upstream_digests":upstream,"fit_manifest_digest":fit_input.fit_manifest_digest,
                "fit_config":dict(PRIMARY_FIT_CONFIG)},model_path)
    audit["checkpoint_sha256"]=sha256(model_path)
    return model,audit,history_rows

def _score_examples(model, examples: Mapping[str,np.ndarray], objective: str) -> np.ndarray:
    import torch
    from gnss_doppler_lab import amcf_shape_only as core
    h=torch.as_tensor(examples["history"],device="cuda"); c=torch.as_tensor(examples["current"],device="cuda"); targets=list(range(8)) if objective=="all9" else [3,4]; out=[]
    model.eval()
    with torch.no_grad():
        for a in range(0,len(c),512):
            cc=c[a:a+512]; hh=h[a:a+512]; n=len(cc); vals=[]
            for target in targets:
                mask=torch.ones((n,8),dtype=torch.bool,device="cuda")
                if objective=="all9": mask[:,target]=False
                else:
                    mask.zero_(); mask[:,4 if target==3 else 3]=True
                loc,scale=model(hh,cc,mask); y=cc[:,target]; vals.append(core.student_t_nll(y,loc,scale,model.df).mean(dim=1))
            score=torch.stack(vals,dim=1)
            if objective=="all9": score=torch.topk(score,2,dim=1).values.mean(dim=1)
            else: score=score.mean(dim=1)
            out.append(score.cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)


def _epoch_scores(examples: Mapping[str,np.ndarray], raw: np.ndarray) -> dict[float,dict[str,Any]]:
    required=("source_start","source_end","prn","segment_index","channel")
    if any(k not in examples for k in required) or len(raw)!=len(examples["source_end"]):
        raise ValueError("epoch score inputs require aligned source interval and identity")
    groups: dict[float,list[tuple[float,str,float]]]={}
    for start,end,prn,segment,channel,score in zip(*(examples[k] for k in required),raw):
        ident=f"{segment}:{channel}:{prn}"
        groups.setdefault(float(end),[]).append((float(start),ident,float(score)))
    out={}
    for end,part in sorted(groups.items()):
        starts=np.asarray([x[0] for x in part],float)
        if not np.allclose(starts,starts[0],rtol=0,atol=1e-9):
            raise ValueError(f"source interval mismatch within epoch {end}")
        identities=[x[1] for x in part]
        if len(set(identities))!=len(identities): raise ValueError("duplicate epoch identity")
        scores={i:v for _,i,v in part}
        identity_hash=_digest_value({"source_start":float(starts[0]),"source_end":end,"identities":sorted(identities)})
        out[end]={"decision_time_s":end,"source_start":float(starts[0]),"source_end":end,
                  "identity_hash":identity_hash,"identities":sorted(identities),
                  "score":aggregate_prn_scores(scores)}
    return out


def _conformal(cal: np.ndarray, values: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    p=(1+np.sum(cal[None,:]>=values[:,None],axis=1))/(len(cal)+1); return p,-np.log(p)


def _loo_evidence(cal: np.ndarray) -> np.ndarray:
    return -np.log((1+(cal[None,:]>=cal[:,None]).sum(1)-1)/len(cal))


def _higher(x: np.ndarray,q: float) -> float: return float(np.quantile(np.asarray(x,float),q,method="higher"))


def _bootstrap_delta(t,y,a,b,metric,ath,bth,mask,reps=2000,seed=0):
    t=np.asarray(t,float); y=np.asarray(y,bool); a=np.asarray(a,float); b=np.asarray(b,float); use=np.asarray(mask,bool)
    t,y,a,b=t[use],y[use],a[use],b[use]
    population_hash=_digest_value({"t":t,"y":y})
    order=np.argsort(t,kind="mergesort"); t,y,a,b=t[order],y[order],a[order],b[order]; block=np.floor((t-t.min())/10.).astype(int); groups=[np.flatnonzero(block==x) for x in np.unique(block)]
    def stat(v,yy,th):
        return _roc_auc(yy,v) if metric=="roc_auc" else float(np.mean(v>th))
    estimate=stat(a,y,ath)-stat(b,y,bth); rng=np.random.default_rng(seed); draws=[]
    while len(draws)<reps:
        ids=np.concatenate([groups[j] for j in rng.integers(0,len(groups),len(groups))])[:len(t)]
        x=stat(a[ids],y[ids],ath); z=stat(b[ids],y[ids],bth)
        if x is not None and z is not None: draws.append(x-z)
    return {"estimate":float(estimate),"ci_low":float(np.quantile(draws,.025)),"ci_high":float(np.quantile(draws,.975)),"reps":reps,"block_s":10.,"bootstrap_seed":int(seed),"phase_population_hash":population_hash,"delta_definition":"Complex-comparator","paired_on":"same timestamp/label/block"}


def _read_csv(path: Path) -> list[dict[str,str]]:
    with path.open(newline="",encoding="utf-8") as f:return list(csv.DictReader(f))


def _append_seed_mean_std(rows: list[dict[str,Any]]) -> list[dict[str,Any]]:
    out=list(rows); groups={}
    for row in rows: groups.setdefault((row.get("scenario"),row.get("model"),row.get("operating_point")),[]).append(row)
    for (scenario,model,operating_point),part in sorted(groups.items(),key=lambda x:str(x[0])):
        values=[float(r["roc_auc"]) for r in part if r.get("roc_auc") is not None]
        out.append({"scenario":scenario,"model":f"mean_std::{model}","operating_point":operating_point,"seed":"mean/std","seed_count":len(part),
                    "roc_auc_mean":float(np.mean(values)) if values else None,
                    "roc_auc_std":float(np.std(values,ddof=1)) if len(values)>1 else (0. if values else None),
                    "all_members_converged":all(bool(r.get("converged")) for r in part),
                    "nonconverged_excluded":all(bool(r.get("converged")) or bool(r.get("excluded")) for r in part)})
    return out


def _metric_rows_for_model(scenario: str, rows: list[dict[str,Any]], variant: str, model_label: str,
                           threshold: Mapping[str,Any]) -> list[dict[str,Any]]:
    generic=[{"decision_time_s":r["decision_time_s"],"source_start":r["source_start"],"source_end":r["source_end"],
              "score_ensemble":r[f"score_{variant}_ensemble"],"alarm_q99":r[f"alarm_{variant}_q99"],
              "alarm_q995":r[f"alarm_{variant}_q995"]} for r in rows]
    out=[]
    for op in ("q99","q995"):
        q=float(threshold[op]); adjusted=[dict(x,alarm_q99=(float(x["score_ensemble"])>q),alarm_q995=(float(x["score_ensemble"])>q)) for x in generic]
        metric=recompute_scenario_metrics(scenario,adjusted,q,q,onset_s=ONSETS.get(scenario))
        metric.update({"model":model_label,"operating_point":op,"diagnostic_only":op=="q995","source_file":f"per_epoch/{scenario}.csv"})
        out.append(metric)
    return out


def _exact_epoch_rows(epoch: Mapping[tuple[str,str],dict[float,dict[str,Any]]], scenario: str, variant: str) -> tuple[list[float],dict[int,list[dict[str,Any]]]]:
    banks={s:list(epoch[scenario,f"{variant}_seed{s}"].values()) for s in SEEDS}
    ref=[(r["decision_time_s"],r["source_start"],r["source_end"],r["identity_hash"]) for r in banks[SEEDS[0]]]
    for seed in SEEDS[1:]:
        cur=[(r["decision_time_s"],r["source_start"],r["source_end"],r["identity_hash"]) for r in banks[seed]]
        if cur!=ref: raise ValueError(f"{scenario} {variant}: exact three seed timestamp/identity/order mismatch")
    return [x[0] for x in ref],banks


def _write_primary_plots(stage: Path, metric_rows: list[dict[str,Any]]) -> None:
    # Deterministic plot-ready evidence for every representation/objective/op.
    rows=[r for r in metric_rows if r.get("scenario")!="DS4"]
    write_csv(stage/"plots"/"scenario_plot_data.csv",rows)
    (stage/"plots"/"README.txt").write_text(
        "Deterministic plot-ready rows include Complex/Magnitude all9/EPL and B0 Exact at q99/q995.\n",
        encoding="utf-8",newline="\n")


def run_primary(out: Path | str = ROOT/FINAL_ARTIFACT, *, bootstrap_reps: int = 2000) -> dict[str,Any]:
    out=Path(out).resolve()
    if out != (ROOT/FINAL_ARTIFACT).resolve(): raise ValueError(f"primary output must be exactly {FINAL_ARTIFACT}")
    if bootstrap_reps<2000: raise ValueError("primary paired bootstrap requires at least 2000 replicates")
    if out.exists(): raise FileExistsError(out)
    source=verify_primary_source_state(ROOT,PRIMARY_BASE_SHA); gpu=gpu_probe(required=True)
    stage=out.with_name(out.name+f".tmp-{os.getpid()}"); stage.mkdir(parents=True)
    try:
        for d in ("per_epoch","plots","models","feature_cache"): (stage/d).mkdir()
        schema=feature_schema_document(PRIMARY_MIN_VALID_ROWS); datasets={}; inputs={}; bundles={}; qas={}; feature_evidence={}
        for name,(path,digest) in CANONICAL.items(): datasets[name],inputs[name]=load_canonical_npz(path,digest,name,tap_order=TAP_NAMES)
        gate=fit_gate_from_clean_train(datasets["cleanStatic"]); gate_hash=_digest_value(dict(gate._asdict()))
        for name in CANONICAL:
            bundles[name],qas[name]=build_feature_bundle(datasets[name],name,gate,min_valid_rows=PRIMARY_MIN_VALID_ROWS)
            feature_evidence[name]=bind_feature_provenance(name,inputs[name],gate,schema,bundles[name],qas[name])
            verify_feature_provenance(feature_evidence[name],inputs[name],gate,schema,bundles[name],qas[name])
            for rep in ("complex","magnitude"): _save_bundle(stage/"feature_cache"/f"{name}_{rep}.npz",bundles[name][rep])
        for name in CANONICAL:
            stored={rep:_load_saved_bundle(stage/"feature_cache"/f"{name}_{rep}.npz",bundles[name][rep].keys()) for rep in ("complex","magnitude")}
            verify_feature_provenance(feature_evidence[name],inputs[name],gate,schema,stored,qas[name]); bundles[name]=stored
        scalers={}
        for rep in ("complex","magnitude"):
            part=bundles["cleanStatic"][rep]; scalers[rep]=_fit_scaler(part["features"][part["role"]=="train"])
        transformed={name:_transform_bundle(bundle,scalers) for name,bundle in bundles.items()}; examples={}
        for name in CANONICAL:
            for rep in ("complex","magnitude"): examples[name,rep]=build_examples(transformed[name][rep])
        models={}; audits={}; history=[]
        for rep in ("complex","magnitude"):
            ex=examples["cleanStatic",rep]; train={k:v[ex["role"]=="train"] for k,v in ex.items()}; val={k:v[ex["role"]=="validation"] for k,v in ex.items()}
            if len(train["current"])<2 or len(val["current"])<2: raise ValueError("insufficient causal clean train/validation examples")
            transformed_hash=_digest_value(transformed["cleanStatic"][rep]["features"])
            for objective in ("all9","EPL"):
                for seed in SEEDS:
                    key=f"{rep}_{objective}_seed{seed}"
                    fit=make_audited_fit_input(train,val,canonical_input_hash=inputs["cleanStatic"]["sha256"],
                        prompt_gate_hash=gate_hash,scaler_hash=scalers[rep]["hash"],
                        transformed_feature_tensor_hash=transformed_hash,objective=objective,seed=seed)
                    models[key],audits[key],rows=_train_primary_model(fit,rep,objective,seed,stage/"models"/f"{key}.pt"); history.extend(rows)
        for objective in ("all9","EPL"):
            for seed in SEEDS:
                if audits[f"complex_{objective}_seed{seed}"]["validation_bank_hash"] != audits[f"magnitude_{objective}_seed{seed}"]["validation_bank_hash"]:
                    raise RuntimeError("validation bank hash differs across representations")
        exact=all(audits[f"{rep}_{obj}_seed{seed}"]["converged"] for rep in ("complex","magnitude") for obj in ("all9","EPL") for seed in SEEDS)
        convergence={"primary_status":"COMPLETE" if exact else "INCOMPLETE: one or more nonconverged models",
                     "exact_three_converged_per_variant":exact,"audits":audits}
        write_csv(stage/"training_history.csv",history); write_json(stage/"convergence_audit.json",convergence)
        if not exact: raise RuntimeError("primary incomplete: nonconverged model; attack scoring was not started")
        epoch={}
        for name in CANONICAL:
            for rep in ("complex","magnitude"):
                for objective in ("all9","EPL"):
                    for seed in SEEDS:
                        key=f"{rep}_{objective}_seed{seed}"
                        epoch[name,key]=_epoch_scores(examples[name,rep],_score_examples(models[key],examples[name,rep],objective))
        thresholds={"comparison":"strict_greater","primary":"q99","q995_role":"diagnostic_only",
                    "quantile_method":"higher","amcf_calibration_interval":[340.0,410.0]}
        evidence={}; calibration={}
        variants=("complex_all9","magnitude_all9","complex_EPL","magnitude_EPL")
        for variant in variants:
            _,clean_banks=_exact_epoch_rows(epoch,"cleanStatic",variant)
            cal_banks={seed:[r for r in clean_banks[seed] if 340<=float(r["decision_time_s"])<410] for seed in SEEDS}
            cal=build_calibration_evidence(variant,cal_banks,source_commit=source["source_commit"],
                 score_bank_hash=_digest_value({s:[r["score"] for r in clean_banks[s]] for s in SEEDS}),
                 index_hash=_digest_value({s:[(r["decision_time_s"],r["identity_hash"]) for r in clean_banks[s]] for s in SEEDS}))
            recompute_calibration_evidence(cal); calibration[variant]=cal
            thresholds[variant]={"q99":cal["q99"],"q995":cal["q995"],"count":len(cal["common_identity"]),
                                 "required_seeds":list(SEEDS),"all_converged":True,"threshold_digest":cal["threshold_digest"],
                                 "evidence_ref":"calibration_evidence.json"}
            evidence[variant]={}
            cal_raw={s:np.asarray(cal["per_seed_raw_scores"][str(s)],float) for s in SEEDS}
            for name in CANONICAL:
                times,banks=_exact_epoch_rows(epoch,name,variant); seed_e={}
                for seed in SEEDS:
                    vals=np.asarray([r["score"] for r in banks[seed]],float); seed_e[seed]=_conformal(cal_raw[seed],vals)
                evidence[variant][name]={"times":times,"rows":banks,
                    "raw":{s:[r["score"] for r in banks[s]] for s in SEEDS},
                    "p":{s:seed_e[s][0] for s in SEEDS},"e":{s:seed_e[s][1] for s in SEEDS},
                    "ensemble":np.mean([seed_e[s][1] for s in SEEDS],axis=0)}
        write_json(stage/"calibration_evidence.json",calibration)
        b0_contract=load_frozen_b0_contract(ROOT/"artifacts/cmte_a2_texbat_epochfix")
        thresholds["B0 Exact"]={"q99":b0_contract["q99"],"q995":b0_contract["q995"],"count":46,
                                "scale":"frozen protected B0 Exact score","source_role":b0_contract["source_role"],
                                "contract_digest":b0_contract["contract_digest"]}
        b0={}; b0_paths={}
        for name in CANONICAL:
            path=B0_DIR/("cleanStatic_test.csv" if name=="cleanStatic" else f"{name}.csv")
            b0[name]=load_b0_exact(path,name); b0_paths[name]={"path":str(path),"sha256":sha256(path),"read_columns":["decision_time_s","score_B0_Exact"]}
        write_json(stage/"thresholds.json",thresholds)
        metric_rows=[]; seed_rows=[]; paired=[]; saved={}
        model_labels={"complex_all9":"Complex all9","magnitude_all9":"Magnitude all9",
                      "complex_EPL":"Complex EPL","magnitude_EPL":"Magnitude EPL"}
        for name in CANONICAL:
            ref_times=evidence[variants[0]][name]["times"]
            for variant in variants[1:]:
                if evidence[variant][name]["times"]!=ref_times: raise ValueError(f"{name}: representation timestamp/order mismatch")
            bmap=dict(zip(b0[name]["decision_time_s"],b0[name]["score_B0_Exact"])); times=[t for t in ref_times if t in bmap]
            if name=="cleanStatic": times=[t for t in times if t>=420 or 340<=t<410]
            maps={variant:{t:i for i,t in enumerate(evidence[variant][name]["times"])} for variant in variants}; rows=[]
            epoch_maps={variant:{r["decision_time_s"]:r for r in evidence[variant][name]["rows"][SEEDS[0]]} for variant in variants}
            refmap=epoch_maps[variants[0]]
            for t in times:
                actual=refmap[t]; aligned=(actual["source_start"],actual["source_end"],actual["identity_hash"])
                if any((epoch_maps[v][t]["source_start"],epoch_maps[v][t]["source_end"],epoch_maps[v][t]["identity_hash"])!=aligned for v in variants[1:]):raise ValueError(f"{name}: common timestamp source interval/identity mismatch")
                start=float(actual["source_start"]); end=float(actual["source_end"])
                phase="calibration" if name=="cleanStatic" and 340<=start and end<=410 else "clean_test" if name=="cleanStatic" else "persistent" if start>=ONSETS[name]+40 else "post" if start>=ONSETS[name] else "stable_pre" if start>=30 and end<=ONSETS[name]-20 else "transition"
                row={"scenario":name,"decision_time_s":t,"source_start":start,"source_end":end,"phase":phase,
                     "epoch_identity_hash":actual["identity_hash"],"tracked_prn_count":len(actual["identities"])}
                for variant in variants:
                    i=maps[variant][t]
                    for seed in SEEDS:
                        row[f"score_{variant}_seed{seed}"]=evidence[variant][name]["raw"][seed][i]
                        row[f"p_{variant}_seed{seed}"]=evidence[variant][name]["p"][seed][i]
                        row[f"e_{variant}_seed{seed}"]=evidence[variant][name]["e"][seed][i]
                    row[f"score_{variant}_ensemble"]=evidence[variant][name]["ensemble"][i]
                    row[f"alarm_{variant}_q99"]=row[f"score_{variant}_ensemble"]>thresholds[variant]["q99"]
                    row[f"alarm_{variant}_q995"]=row[f"score_{variant}_ensemble"]>thresholds[variant]["q995"]
                row["score_B0_Exact"]=bmap[t]; row["alarm_B0_q99"]=bmap[t]>thresholds["B0 Exact"]["q99"]; row["alarm_B0_q995"]=bmap[t]>thresholds["B0 Exact"]["q995"]
                rows.append(row)
            test_rows=[r for r in rows if r["phase"]!="calibration"]; cal_rows=[r for r in rows if r["phase"]=="calibration"]
            write_csv(stage/"per_epoch"/("cleanStatic.csv" if name=="cleanStatic" else f"{name}.csv"),test_rows); saved[name]=test_rows
            if cal_rows: write_csv(stage/"per_epoch"/"cleanStatic_calibration.csv",cal_rows)
            for variant,label in model_labels.items(): metric_rows.extend(_metric_rows_for_model(name,test_rows,variant,label,thresholds[variant]))
            b_rows=[]
            for r in test_rows:
                q={**r,"score_B0_Exact_ensemble":r["score_B0_Exact"]}; q["alarm_B0_Exact_q99"]=r["alarm_B0_q99"]; q["alarm_B0_Exact_q995"]=r["alarm_B0_q995"]; b_rows.append(q)
            metric_rows.extend(_metric_rows_for_model(name,b_rows,"B0_Exact","B0 Exact",thresholds["B0 Exact"]))
            for seed in SEEDS:
                for variant,label in model_labels.items():
                    vals=np.asarray([r[f"e_{variant}_seed{seed}"] for r in test_rows]); starts=np.asarray([r["source_start"] for r in test_rows]); ends=np.asarray([r["source_end"] for r in test_rows])
                    masks=phase_labels(starts,ends,ONSETS[name]) if name in ONSETS else None
                    use=(masks["stable_pre"]|masks["post"]) if masks is not None else np.zeros(len(vals),bool)
                    auc=_roc_auc(masks["post"][use],vals[use]) if masks is not None else None
                    for op in ("q99","q995"):
                        alarm=vals>thresholds[variant][op]
                        seed_rows.append({"scenario":name,"model":label,"operating_point":op,"diagnostic_only":op=="q995",
                          "seed":seed,"roc_auc":auc,"stable_pre_fpr":float(alarm[masks["stable_pre"]].mean()) if masks is not None and masks["stable_pre"].any() else None,
                          "post_detection":float(alarm[masks["post"]].mean()) if masks is not None and masks["post"].any() else None,
                          "converged":True,"source_checkpoint":f"models/{variant}_seed{seed}.pt","source_file":f"per_epoch/{name}.csv"})
        metric_rows.append(ds4_na_row())
        index=index_metric_rows([r for r in metric_rows if r.get("scenario")!="DS4"])
        for name in ONSETS:
            rows=saved[name]; t=np.asarray([r["decision_time_s"] for r in rows]); starts=np.asarray([r["source_start"] for r in rows]); ends=np.asarray([r["source_end"] for r in rows]); masks=phase_labels(starts,ends,ONSETS[name]); y=masks["post"]; c=np.asarray([r["score_complex_all9_ensemble"] for r in rows])
            for comparator,col,thkey in (("Magnitude all9","score_magnitude_all9_ensemble","magnitude_all9"),("B0 Exact","score_B0_Exact","B0 Exact")):
                b=np.asarray([r[col] for r in rows]); ath=thresholds["complex_all9"]["q99"]; bth=thresholds[thkey]["q99"]
                for metric,mask in (("roc_auc",masks["stable_pre"]|masks["post"]),("post_detection",masks["post"]),("stable_pre_fpr",masks["stable_pre"])):
                    seed=101+sum(map(ord,name+comparator+metric)); paired.append({"scenario":name,"comparator":comparator,"metric":metric,"operating_point":"q99",**_bootstrap_delta(t,y,c,b,metric,ath,bth,mask,bootstrap_reps,seed)})
        write_csv(stage/"scenario_metrics.csv",metric_rows); write_csv(stage/"seed_metrics.csv",_append_seed_mean_std(seed_rows)); write_csv(stage/"paired_comparisons.csv",paired); _write_primary_plots(stage,metric_rows)
        collapse={}
        for name in ("cleanStatic","DS7","DS8"):
            checks={}; ok=True
            for rep,d in (("complex",4),("magnitude",2)):
                x=bundles[name][rep]["features"]; iq=np.quantile(x,.75,axis=0)-np.quantile(x,.25,axis=0)
                good=x.shape[1:]==(8,d) and np.isfinite(x).all() and np.all(iq>1e-8)
                checks[rep]={"canonical_fields":list(ALLOWED_NPZ_FIELDS),"feature_cache":f"feature_cache/{name}_{rep}.npz",
                  "source_feature_digest":_digest_value(bundles[name][rep]),"feature_tensor_digest":_digest_value(x),
                  "schema_hash":_digest_value(schema),"scaler_hash":scalers[rep]["hash"],"shape":list(x.shape),
                  "finite_count":int(np.isfinite(x).sum()),"element_count":int(x.size),"iqr":iq.tolist(),"pass":bool(good)}; ok &= bool(good)
            rows=saved[name]
            stable=[r for r in rows if r["phase"]=="stable_pre"] if name!="cleanStatic" else rows
            alarm_evidence=[(r["decision_time_s"],r["source_start"],r["source_end"],r["score_complex_all9_ensemble"],thresholds["complex_all9"]["q99"],r["alarm_complex_all9_q99"]) for r in stable]
            rate=float(np.mean([x[-1] for x in alarm_evidence])) if alarm_evidence else 0.; tracked=float(np.median([r["tracked_prn_count"] for r in rows])) if rows else 0.
            ok &= bool(name=="cleanStatic" or (tracked>1 and rate<1))
            collapse[name]={"representation_checks":checks,"tracked_prn_median":tracked,"stable_pre_alarm_rate":rate,
                            "stable_pre_score_threshold_alarm_digest":_digest_value(alarm_evidence),"pass":bool(ok)}
        feature_binding={"canonical_fields":list(ALLOWED_NPZ_FIELDS),"forbidden_fields":["cn0_db_hz","context"],"scenarios":collapse}
        feature_audit={**feature_binding,"pass":all(collapse[n]["pass"] for n in ("DS7","DS8")),"audit_digest":_digest_value(feature_binding)}; write_json(stage/"feature_audit.json",feature_audit)
        attacks=[select_primary_metric(index,n,"Complex all9") for n in ONSETS]; mags=[select_primary_metric(index,n,"Magnitude all9") for n in ONSETS]; bzeros=[select_primary_metric(index,n,"B0 Exact") for n in ONSETS]
        cis={r["scenario"]:r["ci_low"] for r in paired if r["comparator"]=="Magnitude all9" and r["metric"]=="roc_auc" and r["operating_point"]=="q99"}
        seed_dir={n:sum(next(x for x in seed_rows if x["scenario"]==n and x["model"]=="Complex all9" and x["seed"]==seed and x["operating_point"]=="q99")["roc_auc"]>next(x for x in seed_rows if x["scenario"]==n and x["model"]=="Magnitude all9" and x["seed"]==seed and x["operating_point"]=="q99")["roc_auc"] for seed in SEEDS) for n in ONSETS}
        criteria={"stable_pre_fpr_all_below_0.05":all(r["stable_pre_fpr"]<.05 for r in attacks),"complex_auc_gt_magnitude_4_of_5":sum(a["roc_auc"]>m["roc_auc"] for a,m in zip(attacks,mags))>=4,"auc_bootstrap_ci_lower_gt_zero_3_of_5":sum(v>0 for v in cis.values())>=3,"same_seed_direction_each_scenario":all(v>=2 for v in seed_dir.values()),"beats_b0_with_fpr_guard_3_of_5":sum(((a["roc_auc"]-b["roc_auc"]>=.02 or a["post_detection"]-b["post_detection"]>=.05) and a["stable_pre_fpr"]-b["stable_pre_fpr"]<=.01) for a,b in zip(attacks,bzeros))>=3,"all_required_seeds_converged":exact,"ds7_ds8_no_collapse":feature_audit["pass"]}
        final="GO" if all(criteria.values()) else "NO-GO"
        write_json(stage/"feature_schema.json",schema); write_json(stage/"input_hashes.json",inputs); write_json(stage/"window_qa.json",qas)
        provenance={"schema":"gnss-doppler-lab.amcf-shape-only-provenance.v2",**source,"execution_tree_clean":True,"gpu_execution":gpu,
          "inputs":inputs,"B0_inputs":b0_paths,"B0_frozen_contract":b0_contract,"threshold_protocol_limitation":"AMCF [340,410) and frozen B0 [300,330) differ; threshold-dependent comparisons are limited",
          "gate":dict(gate._asdict()),"scalers":scalers,"gate_hash":gate_hash,"gate_scaler_hash":_digest_value({"gate":dict(gate._asdict()),"scalers":scalers}),
          "feature_schema_hash":_digest_value(schema),"feature_provenance":feature_evidence,"feature_audit_digest":feature_audit["audit_digest"],
          "fit_config":dict(PRIMARY_FIT_CONFIG),"fit_audits":{k:{"fit_manifest_digest":v["fit_manifest_digest"],"upstream_digests":v["upstream_digests"]} for k,v in audits.items()},"attack_fit":False,"all_attack_results":"exploratory/developmental"}; write_json(stage/"provenance.json",provenance)
        config={"schema":"gnss-doppler-lab.amcf-shape-only-config.v2","mode":"primary-full","base_sha":PRIMARY_BASE_SHA,"seeds":list(SEEDS),"objectives":["all9","EPL"],"representations":["complex","magnitude"],**dict(PRIMARY_FIT_CONFIG),"hidden":32,"history":12,"stride_s":.5,"source_window_s":1.,"minimum_valid_rows":PRIMARY_MIN_VALID_ROWS,"bootstrap_reps":bootstrap_reps,"onsets":ONSETS,"scenarios":[*ONSETS,"DS4:NA"],"DS4":{"status":"NA","included_in_attack_go":False},"q99_primary":True,"q995_diagnostic_only":True,"matched_clean":False}; write_json(stage/"config.json",config)
        digest_paths={"thresholds":"thresholds.json","seed_metrics":"seed_metrics.csv","paired_comparisons":"paired_comparisons.csv","scenario_metrics":"scenario_metrics.csv","provenance":"provenance.json","feature_audit":"feature_audit.json","feature_schema":"feature_schema.json","fit_checkpoint_audits":"convergence_audit.json","training_history":"training_history.csv","calibration_evidence":"calibration_evidence.json"}
        source_digests={k:sha256(stage/v) for k,v in digest_paths.items()}; source_digests["per_epoch"]={str(p.relative_to(stage)):sha256(p) for p in sorted((stage/"per_epoch").glob("*.csv"))}; source_digests["checkpoints"]={str(p.relative_to(stage)):sha256(p) for p in sorted((stage/"models").glob("*.pt"))}
        decision={"primary_quantile":"q99","primary_decision":final,"amcf_wcl":"GO candidate" if final=="GO" else "AMCF WCL no-go","criteria":criteria,"source_digests":source_digests}; write_json(stage/"decision.json",decision)
        (stage/"README.md").write_text(render_readme(final,"PRIMARY COMPLETE",criteria),encoding="utf-8",newline="\n")
        write_hashes(stage); verify_hashes(stage); missing=[x for x in PRIMARY_REQUIRED_INVENTORY if not (stage/x).exists()]
        if missing: raise RuntimeError(f"missing inventory {missing}")
        # Independent verifier runs against staging.  Publication is a single atomic rename only after agreement.
        spec=importlib.util.spec_from_file_location("amcf_shape_staging_summarizer",ROOT/"scripts/summarize_amcf_shape_only.py")
        assert spec and spec.loader; summarizer=importlib.util.module_from_spec(spec); spec.loader.exec_module(summarizer)
        report=summarizer.verify_and_summarize(stage)
        if report.get("primary_decision")!=final or report.get("mode")!="primary-full" or not report.get("byte_identical"):
            raise RuntimeError("staging summarizer/independent verifier disagrees with runner")
        verify_primary_source_state(ROOT,PRIMARY_BASE_SHA)
        os.replace(stage,out); return {"status":final,"out":str(out),"source_commit":source["source_commit"],"summary":report}
    except Exception:
        shutil.rmtree(stage,ignore_errors=True); raise


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--out",type=Path,default=ROOT/FINAL_ARTIFACT); p.add_argument("--smoke",action="store_true"); p.add_argument("--fixture-seed",type=int,default=7); p.add_argument("--bootstrap-reps",type=int,default=2000); return p


def main(argv=None) -> int:
    args=parser().parse_args(argv)
    result=run_smoke(args.out,fixture_seed=args.fixture_seed) if args.smoke else run_primary(args.out,bootstrap_reps=args.bootstrap_reps)
    print(json.dumps(result,sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())

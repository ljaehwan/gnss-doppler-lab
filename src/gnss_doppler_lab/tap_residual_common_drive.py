"""Causal Tap-Residual Common-Drive V0 around a frozen PRN-local B0.

The module retains B0 standardized tap innovations as vectors and computes a
same-event, leave-one-PRN-out relation score.  It never fits on scored data.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch

INNOVATION_PREFIX = "innovation_"
EVENT_CADENCE_S = 0.5
TIMING_TOLERANCE_S = 1e-7
TIMING_CONTRACT = {
    "score_time_field": "window_start_s",
    "availability_time_field": "window_end_s",
    "availability_offset": "window_end_s - window_start_s",
    "smoothing": "strictly causal, current-and-past only, reset per run",
    "event_time_fields": "window_start_s=min start and window_end_s=max end form the same-bin availability envelope; window_mid_s=window_bin_s representative",
}
META_COLUMNS = ["run_id", "prn", "window_bin_s", "window_start_s", "window_end_s", "window_mid_s"]


def _finite_array(values, name: str) -> np.ndarray:
    try:
        out = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric and finite") from exc
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    return out


def _validate_identifiers(frame: pd.DataFrame) -> None:
    if frame[["run_id", "prn"]].isna().any().any() or any(frame[c].astype(str).str.strip().eq("").any() for c in ("run_id", "prn")):
        raise ValueError("run_id and prn must be non-null and non-empty")


def _validate_same_event_timing(frame: pd.DataFrame) -> None:
    """Require actual PRN windows to remain coherent with their 0.5 s event bin."""
    half_stride = EVENT_CADENCE_S / 2
    if np.any(np.abs(frame["window_mid_s"].to_numpy(float) - frame["window_bin_s"].to_numpy(float)) > half_stride + TIMING_TOLERANCE_S):
        raise ValueError("same-event timing must place every window midpoint within stride/2 of window_bin_s")
    for _, group in frame.groupby(["run_id", "window_bin_s"], sort=False):
        if any(np.ptp(group[column].to_numpy(float)) > EVENT_CADENCE_S + TIMING_TOLERANCE_S for column in ("window_start_s", "window_end_s", "window_mid_s")):
            raise ValueError("same-event timing span exceeds the 0.5 s event stride")


def _validate_sequence_contract(frame: pd.DataFrame) -> None:
    key = ["run_id", "prn", "window_bin_s"]
    if frame.duplicated(key).any():
        raise ValueError("duplicate PRN in (run_id, prn, window_bin_s) sequence row")
    starts, ends, mids = (frame[c].to_numpy(float) for c in ("window_start_s", "window_end_s", "window_mid_s"))
    if np.any(ends <= starts) or not np.allclose(mids, (starts + ends) / 2, atol=1e-6, rtol=0):
        raise ValueError("inconsistent window timing: require end > start and midpoint=(start+end)/2")
    for _, group in frame.groupby(["run_id", "prn"], sort=False):
        bins = np.sort(group["window_bin_s"].to_numpy(float))
        if len(bins) > 1 and not np.allclose(np.diff(bins), EVENT_CADENCE_S, atol=TIMING_TOLERANCE_S, rtol=0):
            raise ValueError("sequence gap: Tap-Residual V0 requires contiguous 0.5 s cadence")
    _validate_same_event_timing(frame)


def extract_b0_innovations(frame: pd.DataFrame, model: torch.nn.Module,
                           feature_columns: Sequence[str], mean: Sequence[float],
                           std: Sequence[float], *, seq_len: int = 12,
                           device: str | torch.device = "cpu",
                           batch_size: int = 1024) -> pd.DataFrame:
    """Extract signed standardized target-minus-B0-prediction tap vectors.

    Output is deterministically ordered by run, PRN, and target time.  B0 RMSE
    is computed exactly as sqrt(mean(innovation**2)); no relational reduction
    changes that parallel baseline score.
    """
    required = [*META_COLUMNS, *feature_columns]
    missing = [c for c in required if c not in frame]
    if missing:
        raise ValueError(f"node frame missing required columns: {missing}")
    if seq_len < 1 or batch_size < 1:
        raise ValueError("seq_len and batch_size must be positive")
    mu = _finite_array(mean, "mean")
    scale = _finite_array(std, "std")
    if mu.shape != (len(feature_columns),) or scale.shape != mu.shape or np.any(scale <= 0):
        raise ValueError("standardizer dimensions must match features and std must be positive")
    _finite_array(frame[["window_bin_s", "window_start_s", "window_end_s", "window_mid_s", *feature_columns]], "timing/features")
    _validate_identifiers(frame)
    _validate_sequence_contract(frame)
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()
    rows: list[dict] = []
    with torch.no_grad():
        for (run_id, prn), group in frame.groupby(["run_id", "prn"], sort=True):
            group = group.sort_values(["window_bin_s", "window_start_s"], kind="mergesort").reset_index(drop=True)
            raw = group[list(feature_columns)].to_numpy(np.float32)
            x = ((raw - mu) / scale).astype(np.float32)
            count = len(x) - seq_len
            if count <= 0:
                continue
            for offset in range(0, count, batch_size):
                stops = range(offset, min(offset + batch_size, count))
                seq = np.stack([x[i:i + seq_len] for i in stops])
                target_indices = np.arange(offset + seq_len, min(offset + batch_size, count) + seq_len)
                target = x[target_indices]
                pred = model(torch.from_numpy(seq).to(dev)).detach().cpu().numpy()
                if pred.shape != target.shape or not np.isfinite(pred).all():
                    raise ValueError("B0 prediction must be finite and match target vector shape")
                innovation = target - pred
                for j, target_index in enumerate(target_indices):
                    source = group.iloc[int(target_index)]
                    vector = innovation[j]
                    row = {c: source[c] for c in META_COLUMNS}
                    row["target_window_index"] = int(target_index)
                    row["availability_offset_s"] = float(source["window_end_s"] - source["window_start_s"])
                    row["b0_prn_node_rmse"] = float(np.sqrt(np.mean(np.square(vector))))
                    row.update({f"{INNOVATION_PREFIX}{k}": float(v) for k, v in enumerate(vector)})
                    rows.append(row)
    columns = [*META_COLUMNS, "target_window_index", "availability_offset_s", "b0_prn_node_rmse", *[f"{INNOVATION_PREFIX}{i}" for i in range(len(feature_columns))]]
    return pd.DataFrame(rows, columns=columns)


def _innovation_columns(frame: pd.DataFrame) -> list[str]:
    cols = [c for c in frame if c.startswith(INNOVATION_PREFIX)]
    try:
        cols.sort(key=lambda c: int(c[len(INNOVATION_PREFIX):]))
    except ValueError as exc:
        raise ValueError("innovation columns need numeric suffixes") from exc
    if not cols or cols != [f"{INNOVATION_PREFIX}{i}" for i in range(len(cols))]:
        raise ValueError("innovation vector columns must be contiguous from innovation_0")
    return cols


def score_common_drive(innovations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score nodes and events with robust leave-one-out median common vectors."""
    required = [*META_COLUMNS, "b0_prn_node_rmse"]
    missing = [c for c in required if c not in innovations]
    if missing:
        raise ValueError(f"innovation frame missing required columns: {missing}")
    _validate_identifiers(innovations)
    starts = innovations["window_start_s"].to_numpy(float)
    ends = innovations["window_end_s"].to_numpy(float)
    mids = innovations["window_mid_s"].to_numpy(float)
    if np.any(ends <= starts) or not np.allclose(mids, (starts + ends) / 2, atol=1e-6, rtol=0):
        raise ValueError("inconsistent window timing: require end > start and midpoint=(start+end)/2")
    vector_cols = _innovation_columns(innovations)
    numeric = ["window_bin_s", "window_start_s", "window_end_s", "window_mid_s", "b0_prn_node_rmse", *vector_cols]
    _finite_array(innovations[numeric], "innovation/timing inputs")
    keys = ["run_id", "window_bin_s"]
    if innovations.duplicated([*keys, "prn"]).any():
        raise ValueError("duplicate PRN within (run_id, window_bin_s) event")
    _validate_same_event_timing(innovations)
    node_parts = []
    event_rows = []
    ordered = innovations.sort_values([*keys, "prn"], kind="mergesort")
    for (run_id, window_bin), group in ordered.groupby(keys, sort=True):
        group = group.sort_values("prn", kind="mergesort").copy()
        vectors = group[vector_cols].to_numpy(float)
        count, dim = vectors.shape
        energy = np.sqrt(np.mean(np.square(vectors), axis=1))
        common = np.zeros_like(vectors)
        alignment = np.zeros(count, dtype=float)
        eligible = count >= 2
        if eligible:
            for i in range(count):
                common[i] = np.median(np.delete(vectors, i, axis=0), axis=0)
                denom = np.linalg.norm(vectors[i]) * np.linalg.norm(common[i])
                alignment[i] = max(0.0, float(np.dot(vectors[i], common[i]) / denom)) if denom > 0 else 0.0
        joint = energy * alignment
        group["relation_eligible"] = eligible
        group["event_prn_count"] = count
        group["local_support"] = energy
        group["positive_alignment"] = alignment
        group["common_drive_support"] = alignment
        group["joint_evidence"] = joint
        for k in range(dim):
            group[f"loo_common_{k}"] = common[:, k]
        node_parts.append(group)
        event_rows.append({
            "run_id": run_id, "window_bin_s": float(window_bin),
            "window_start_s": float(group.window_start_s.min()),
            "window_end_s": float(group.window_end_s.max()),
            # Event start/end are the availability envelope; midpoint is its bin representative.
            "window_mid_s": float(window_bin),
            "availability_offset_s": float(group.window_end_s.max() - group.window_start_s.min()),
            "tracked_prn_count": count, "relation_eligible": eligible,
            "b0_prn_node_rmse_max": float(group.b0_prn_node_rmse.max()),
            "b0_prn_node_rmse_mean": float(group.b0_prn_node_rmse.mean()),
            "event_local_support": float(np.max(energy)),
            "event_common_drive_support": float(np.mean(alignment)),
            "event_joint_evidence": float(np.mean(joint)),
        })
    nodes = pd.concat(node_parts, ignore_index=True) if node_parts else ordered.copy()
    events = pd.DataFrame(event_rows)
    return nodes.reset_index(drop=True), events.reset_index(drop=True)


def causal_smooth_events(events: pd.DataFrame, alpha: float = 0.35,
                         score_columns: Iterable[str] = ("event_local_support", "event_common_drive_support", "event_joint_evidence")) -> pd.DataFrame:
    """Apply an EWMA using only current/past events, independently per run."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in (0, 1]")
    out = events.sort_values(["run_id", "window_bin_s"], kind="mergesort").copy()
    for column in score_columns:
        if column not in out:
            continue
        values = _finite_array(out[column], column).astype(float)
        smoothed = np.empty(len(out), dtype=float)
        for positions in out.groupby("run_id", sort=False).indices.values():
            previous = None
            for pos in positions:
                current = float(values[pos])
                previous = current if previous is None else alpha * current + (1 - alpha) * previous
                smoothed[pos] = previous
        out[f"{column}_causal"] = smoothed
    return out.reset_index(drop=True)


def build_clean_calibration_document(events: pd.DataFrame, *, source_kind: str,
                                     source_paths: Sequence[str], source_fingerprint: str | None,
                                     checkpoint_sha256: str,
                                     quantiles: Sequence[float] = (.95, .99, .995)) -> dict:
    if source_kind != "cleanStatic":
        raise ValueError("gate calibration source must be cleanStatic")
    columns = [c for c in ("event_local_support", "event_common_drive_support", "event_joint_evidence", "event_joint_evidence_causal") if c in events]
    if events.empty or not columns:
        raise ValueError("cleanStatic calibration events/scores must be nonempty")
    _finite_array(events[columns], "cleanStatic calibration scores")
    if not quantiles or any(not 0 < float(q) < 1 for q in quantiles):
        raise ValueError("calibration quantiles must be in (0, 1)")
    thresholds = {c: {f"q{float(q):g}": float(events[c].quantile(float(q))) for q in quantiles} for c in columns}
    return {"schema": "gnss-doppler-lab.tap-residual-common-drive-v0.calibration", "thresholds": thresholds,
        "provenance": {"gate_source": "cleanStatic_only", "source_kind": source_kind,
            "source_paths": list(source_paths), "source_fingerprint": source_fingerprint,
            "source_run_ids": sorted(events.run_id.astype(str).unique().tolist()) if "run_id" in events else [],
            "checkpoint_sha256": checkpoint_sha256, "attack_labels_used": False,
            "attack_prefix_fitted": False, "cleanDynamic_role": "OOD diagnostic only; never a gate source"},
        "row_count": int(len(events)), "quantiles": [float(q) for q in quantiles]}


def _atomic_write_text(path: Path, text: str, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text); handle.flush(); os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            os.unlink(temporary)
    except BaseException:
        try: os.unlink(temporary)
        except FileNotFoundError: pass
        raise


def calibrate_clean_only(events: pd.DataFrame, output_path: str | Path, *,
                         source_kind: str, source_paths: Sequence[str],
                         checkpoint_sha256: str, source_fingerprint: str | None = None,
                         quantiles: Sequence[float] = (.95, .99, .995), overwrite: bool = False) -> dict:
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite calibration: {path}")
    document = build_clean_calibration_document(events, source_kind=source_kind, source_paths=source_paths,
        source_fingerprint=source_fingerprint, checkpoint_sha256=checkpoint_sha256, quantiles=quantiles)
    _atomic_write_text(path, json.dumps(document, indent=2, sort_keys=True) + "\n", overwrite=overwrite)
    return document

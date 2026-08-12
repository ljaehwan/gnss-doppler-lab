"""Frozen one-shot protected evaluation helpers and artifact composition."""
from __future__ import annotations

import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .gcspo_artifacts import canonical_write_json
from .gcspo_clean import AggregatedClean, EPOCH_SAMPLES, Q_FIELDS, SAMPLE_RATE_HZ, signed_q
from .gcspo_core import SharedVAR, Whitener, persistent_three_of_five, weighted_low_fpr_pauc


def phase_rows(rows, start_s: float, end_s: float):
    """Require both the full one-second window and its endpoint in a phase."""
    return [row for row in rows if float(row["window_start_s"]) >= start_s and float(row["availability_s"]) < end_s]


def reconstruct_normal_model(doc):
    model = SharedVAR(np.asarray(doc["intercept"], float), np.asarray(doc["coefficients"], float))
    whitener = Whitener(np.asarray(doc["whitener_location"], float), np.asarray(doc["whitener_covariance"], float),
                        np.asarray(doc["whitener_inverse_sqrt"], float))
    gamma = np.asarray(doc["gamma"], float)
    if model.coefficients.shape[1:] != (10, 10) or gamma.shape != (10, 10):
        raise ValueError("frozen normal model dimensionality mismatch")
    return model, whitener, gamma


def scientific_verdict(gates):
    required = {"G1_FALSE_ALARM", "G2_INCREMENTAL", "G3_GEOMETRY", "G4_PERSISTENCE", "G5_CONTROLS", "G6_SHARED"}
    observed = {row["id"] for row in gates}
    if observed != required:
        raise ValueError("mandatory scientific gate set mismatch")
    return "GO_FOR_NEURAL_STAGE1" if all(row["status"] == "PASS" for row in gates) else "NO_GO_PHYSICAL_HYPOTHESIS"


def load_receiver_tracking(paths, *, epsilons, gate, scenario):
    """Load only manifest-authenticated protected MAT capabilities."""
    paths = tuple(map(Path, paths))
    if not paths or gate is None: raise ValueError("protected tracking capabilities are required")
    chunks = {name: [] for name in ("sample", "prn", "channel", "segment", *Q_FIELDS)}
    datasets = ("PRN_start_sample_count", "PRN", *Q_FIELDS)
    for channel_id, path in enumerate(paths):
        values = {name: np.asarray(value).reshape(-1) for name, value in
                  gate.read_h5(path, datasets=datasets, scenario=scenario, phase="all_frozen_phases",
                               purpose="protected tracking scientific rows").items()}
        length = len(values["PRN"])
        if any(len(value) != length for value in values.values()) or any(not np.isfinite(value).all() for value in values.values()):
            raise ValueError("protected tracking MAT shape/finite failure")
        sample = values["PRN_start_sample_count"].astype(np.int64); prn_values = values["PRN"].astype(np.int64)
        breaks = np.r_[True, (np.diff(prn_values) != 0) | (np.diff(sample) <= 0)]
        segment = np.cumsum(breaks, dtype=np.int64) - 1
        chunks["sample"].append(sample); chunks["prn"].append(prn_values)
        chunks["channel"].append(np.full(length, channel_id, np.int64)); chunks["segment"].append(segment)
        for name in Q_FIELDS: chunks[name].append(values[name].astype(float))
    columns = {name: np.concatenate(parts) for name, parts in chunks.items()}
    known = np.isin(columns["prn"], np.asarray(sorted(epsilons), int))
    columns = {name: value[known] for name, value in columns.items()}
    epsilon = np.asarray([epsilons[int(prn)] for prn in columns["prn"]], float)
    q = signed_q(columns, epsilon=epsilon)
    identities = {(int(channel), int(sample), int(prn), row.tobytes())
                  for channel, sample, prn, row in zip(columns["channel"], columns["sample"], columns["prn"], q)}
    if len(identities) != len(q):
        raise ValueError("exact duplicate protected scientific row")
    epoch = columns["sample"] // EPOCH_SAMPLES
    order = np.lexsort((columns["sample"], columns["segment"], columns["channel"], columns["prn"], epoch))
    epoch, prn, channel, segment, sample, q = (epoch[order], columns["prn"][order], columns["channel"][order],
                                               columns["segment"][order], columns["sample"][order], q[order])
    boundaries = np.r_[0, np.flatnonzero((np.diff(epoch) != 0) | (np.diff(prn) != 0) |
                                         (np.diff(channel) != 0) | (np.diff(segment) != 0)) + 1, len(epoch)]
    out_epoch, out_prn, out_channel, out_segment, out_q, sample_min, sample_max = [], [], [], [], [], [], []
    for left, right in zip(boundaries, boundaries[1:]):
        identities = {(int(sample[index]), q[index].tobytes()) for index in range(left, right)}
        if len(identities) != right - left: raise ValueError("exact duplicate protected scientific rows")
        out_epoch.append(epoch[left]); out_prn.append(prn[left]); out_channel.append(channel[left]); out_segment.append(segment[left])
        out_q.append(np.median(q[left:right], axis=0)); sample_min.append(sample[left:right].min()); sample_max.append(sample[left:right].max())
    return AggregatedClean(np.asarray(out_epoch, np.int64), np.asarray(out_prn, np.int64),
                           np.asarray(out_channel, np.int64), np.asarray(out_segment, np.int64), np.vstack(out_q),
                           np.asarray(sample_min, np.int64), np.asarray(sample_max, np.int64), dict(epsilons),
                           tuple(str(path) for path in paths))


def score_metrics(rows, *, threshold):
    ordered = sorted(rows, key=lambda row: row["availability_s"])
    alarms = [float(row["score"]) > threshold for row in ordered]
    persistent = persistent_three_of_five(alarms)
    return {"windows": len(ordered), "alarm_ratio": float(np.mean(alarms)) if alarms else None,
            "persistent_alarm_ratio": float(np.mean(persistent)) if persistent else None,
            "first_persistent_alarm_s": next((float(row["availability_s"]) for row, flag in zip(ordered, persistent) if flag), None)}


def discrimination_metrics(negative, positive):
    neg, pos = np.asarray(negative, float), np.asarray(positive, float)
    if not len(neg) or not len(pos):
        return {"roc_auc": None, "low_fpr_pauc": None, "pr_auc": None}
    scores = np.r_[neg, pos]; labels = np.r_[np.zeros(len(neg), bool), np.ones(len(pos), bool)]
    cells = np.asarray(["negative"] * len(neg) + ["positive"] * len(pos))
    return {"roc_auc": float(roc_auc_score(labels, scores)),
            "low_fpr_pauc": weighted_low_fpr_pauc(scores, labels, cells, alpha=.05),
            "pr_auc": float(average_precision_score(labels, scores))}


def write_csv(path, rows, fieldnames):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)

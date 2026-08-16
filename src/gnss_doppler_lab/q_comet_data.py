"""Data and provenance utilities for Q-COMET Stage-0.

The module never assigns attack labels and never chooses configuration from an
attack recording.  It accepts the authenticated complex-nine-tap export format
used by the receiver pipeline and the preserved MATLAB-v7.3 receiver dumps.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


TAP_NAMES = ("E4", "E3", "E2", "E", "P", "L", "L2", "L3", "L4")
TAP_OFFSETS_CHIPS = np.arange(-0.5, 0.5001, 0.125)
REQUIRED_NPZ = {"complex_iq", "time_s", "prn", "channel", "segment_index", "sample_count"}


@dataclass(frozen=True)
class EpochData:
    """One causal observation per time-bin and PRN."""

    time_s: np.ndarray
    epoch: np.ndarray
    prn: np.ndarray
    segment: np.ndarray
    complex_taps: np.ndarray
    sample_count: np.ndarray
    cn0_db_hz: np.ndarray
    prompt_power: np.ndarray
    cadence_s: float
    recording_id: str

    def subset(self, start_s: float, end_s: float) -> "EpochData":
        keep = (self.time_s >= start_s) & (self.time_s < end_s)
        return EpochData(*(np.asarray(value)[keep] for value in (
            self.time_s, self.epoch, self.prn, self.segment, self.complex_taps,
            self.sample_count, self.cn0_db_hz, self.prompt_power)), self.cadence_s,
            self.recording_id)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _validate_rows(iq, time_s, prn, channel, segment, sample_count) -> None:
    n = len(time_s)
    if iq.shape != (n, 9, 2) or any(len(v) != n for v in (prn, channel, segment, sample_count)):
        raise ValueError("complex-nine-tap arrays have inconsistent shapes")
    if n == 0 or not np.isfinite(iq).all() or not np.isfinite(time_s).all():
        raise ValueError("complex-nine-tap source is empty or non-finite")
    if np.all(iq[..., 1] == 0) or np.std(iq[..., 1]) == 0:
        raise ValueError("Q component is absent or degenerate")
    if np.any((prn < 1) | (prn > 32)):
        raise ValueError("invalid GPS PRN")
    for ch, seg in np.unique(np.column_stack((channel, segment)), axis=0):
        use = (channel == ch) & (segment == seg)
        if np.any(np.diff(time_s[use]) <= 0) or np.any(np.diff(sample_count[use]) <= 0):
            raise ValueError("timestamp/sample count is not strictly monotone within receiver segment")


def aggregate_epochs(*, iq: np.ndarray, time_s: np.ndarray, prn: np.ndarray,
                     segment: np.ndarray, sample_count: np.ndarray,
                     cn0: np.ndarray, cadence_s: float, recording_id: str) -> EpochData:
    """Causally aggregate receiver rows into fixed bins using only rows in that bin."""
    bins = np.floor(np.asarray(time_s, float) / cadence_s + 1e-10).astype(np.int64)
    prn = np.asarray(prn, np.int64)
    segment = np.asarray(segment, np.int64)
    keys = np.rec.fromarrays((bins, prn, segment), names="epoch,prn,segment")
    order = np.argsort(keys, kind="stable")
    keys_sorted = keys[order]
    starts = np.r_[0, 1 + np.flatnonzero(keys_sorted[1:] != keys_sorted[:-1])]
    ends = np.r_[starts[1:], len(order)]
    out_iq, out_t, out_prn, out_seg, out_sample, out_cn0 = [], [], [], [], [], []
    for start, end in zip(starts, ends):
        ix = order[start:end]
        # Mean is confined to the completed bin; timestamp is the final contributing row,
        # so the declared availability time never precedes an input row.
        out_iq.append(np.mean(iq[ix, :, 0], axis=0) + 1j * np.mean(iq[ix, :, 1], axis=0))
        out_t.append(float(np.max(time_s[ix])))
        out_prn.append(int(prn[ix[0]])); out_seg.append(int(segment[ix[0]]))
        out_sample.append(int(np.max(sample_count[ix])))
        finite = np.asarray(cn0[ix], float); finite = finite[np.isfinite(finite)]
        out_cn0.append(float(np.median(finite)) if finite.size else np.nan)
    taps = np.asarray(out_iq, np.complex128)
    epochs = np.floor(np.asarray(out_t) / cadence_s + 1e-10).astype(np.int64)
    return EpochData(np.asarray(out_t), epochs, np.asarray(out_prn, np.int64),
                     np.asarray(out_seg, np.int64), taps, np.asarray(out_sample, np.uint64),
                     np.asarray(out_cn0), np.abs(taps[:, 4]) ** 2, float(cadence_s), recording_id)


def load_complex9_npz(path: str | Path, *, recording_id: str, cadence_s: float = 0.1) -> EpochData:
    path = Path(path)
    with np.load(path, allow_pickle=False) as source:
        missing = REQUIRED_NPZ.difference(source.files)
        if missing:
            raise ValueError(f"complex-nine-tap NPZ missing {sorted(missing)}")
        iq = np.asarray(source["complex_iq"])
        time_s = np.asarray(source["time_s"], float)
        prn = np.asarray(source["prn"], np.int64)
        channel = np.asarray(source["channel"], np.int64)
        segment = np.asarray(source["segment_index"], np.int64)
        sample_count = np.asarray(source["sample_count"], np.uint64)
        cn0 = np.asarray(source["cn0_db_hz"], float) if "cn0_db_hz" in source.files else np.full(len(time_s), np.nan)
    _validate_rows(iq, time_s, prn, channel, segment, sample_count)
    return aggregate_epochs(iq=iq, time_s=time_s, prn=prn, segment=segment,
                            sample_count=sample_count, cn0=cn0, cadence_s=cadence_s,
                            recording_id=recording_id)


def load_complex9_mat_directory(path: str | Path, *, recording_id: str,
                                sample_rate_hz: float, cadence_s: float = 0.1) -> EpochData:
    """Load preserved receiver MAT rows without relying on a derived NPZ."""
    path = Path(path)
    pieces = []
    tap_i = ["I_E4", "I_E3", "I_E2", "I_E", "I_P", "I_L", "I_L2", "I_L3", "I_L4"]
    tap_q = [name.replace("I_", "Q_") for name in tap_i]
    mats = sorted(path.glob("epl_tracking_ch_*.mat")) or sorted(path.glob("epl_*.mat"))
    for channel, mat in enumerate(mats):
        with h5py.File(mat, "r") as handle:
            get = lambda name: np.asarray(handle[name]).reshape(-1)
            i = np.column_stack([get(name) for name in tap_i])
            q = np.column_stack([get(name) for name in tap_q])
            sample = get("PRN_start_sample_count").astype(np.uint64)
            prn = get("PRN").astype(np.int64)
            cn0 = get("CN0_SNV_dB_Hz") if "CN0_SNV_dB_Hz" in handle else np.full(len(prn), np.nan)
        # A channel may reacquire another PRN; a new segment starts at each PRN change.
        segment = np.cumsum(np.r_[0, prn[1:] != prn[:-1]]).astype(np.int64)
        pieces.append((np.stack((i, q), axis=-1), sample / sample_rate_hz,
                       prn, np.full(len(prn), channel), segment, sample, cn0))
    if not pieces:
        raise FileNotFoundError(f"no epl_tracking_ch_*.mat in {path}")
    columns = [np.concatenate([piece[k] for piece in pieces]) for k in range(7)]
    _validate_rows(*columns[:6])
    return aggregate_epochs(iq=columns[0], time_s=columns[1], prn=columns[2],
                            segment=columns[4] + columns[3] * 1000, sample_count=columns[5],
                            cn0=columns[6], cadence_s=cadence_s, recording_id=recording_id)


def audit_split_ranges(ranges: dict[str, tuple[float, float]], *, sample_rate_hz: float,
                       bytes_per_complex_sample: int = 4) -> dict:
    rows = []
    for role, (start, end) in ranges.items():
        rows.append({"role": role, "start_s": start, "end_s": end,
                     "sample_start": int(np.floor(start * sample_rate_hz)),
                     "sample_end_exclusive": int(np.ceil(end * sample_rate_hz)),
                     "byte_start": int(np.floor(start * sample_rate_hz)) * bytes_per_complex_sample,
                     "byte_end_exclusive": int(np.ceil(end * sample_rate_hz)) * bytes_per_complex_sample})
    overlaps = []
    for a_index, a in enumerate(rows):
        for b in rows[a_index + 1:]:
            overlap = max(0, min(a["byte_end_exclusive"], b["byte_end_exclusive"]) -
                          max(a["byte_start"], b["byte_start"]))
            overlaps.append({"a": a["role"], "b": b["role"], "overlap_bytes": overlap})
    return {"ranges": rows, "pairwise_overlap": overlaps,
            "all_disjoint": all(item["overlap_bytes"] == 0 for item in overlaps)}


def desynchronize_by_prn(values: np.ndarray, prns: np.ndarray, epochs: np.ndarray,
                         *, seed: int = 20260816) -> tuple[np.ndarray, dict]:
    """Circularly shift each PRN sequence; exact vectors and norms are preserved."""
    values = np.asarray(values)
    out = values.copy(); rng = np.random.default_rng(seed); shifts = {}
    for prn in sorted(map(int, np.unique(prns))):
        ix = np.flatnonzero(prns == prn)
        if len(ix) < 3:
            shifts[str(prn)] = 0; continue
        shift = int(rng.integers(1, len(ix)))
        out[ix] = values[np.roll(ix, shift)]
        shifts[str(prn)] = shift
    if not np.array_equal(np.sort(np.linalg.norm(out, axis=1)), np.sort(np.linalg.norm(values, axis=1))):
        raise AssertionError("relation destruction did not preserve residual norms")
    return out, {"seed": seed, "shifts": shifts, "preserves_vectors_exactly": True,
                 "preserves_norms_exactly": True, "tracked_count_and_dimensions_preserved": True}

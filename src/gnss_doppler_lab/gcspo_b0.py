"""Exact-support utilities for the frozen B0 PRN-local GRU comparator."""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

SAMPLE_RATE_HZ = 25_000_000
EPSILON = 1e-12
TAP_NAMES = ("E4", "E3", "E2", "E", "P", "L", "L2", "L3", "L4")
TAP_FIELDS = tuple(f"abs_{name}" for name in TAP_NAMES)

def _pandas():
    import pandas
    return pandas



def _prn_number(value) -> int:
    text = str(value).strip().upper()
    return int(text[1:] if text.startswith("G") else text)


def role_filter(frame, start_s: float, end_s: float):
    """Keep only one-second windows wholly contained in a half-open role."""
    required = {"window_start_s", "window_end_s"}
    pd = _pandas()
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"B0 table missing role columns: {sorted(missing)}")
    starts = pd.to_numeric(frame["window_start_s"], errors="raise")
    ends = pd.to_numeric(frame["window_end_s"], errors="raise")
    if not np.isfinite(starts).all() or not np.isfinite(ends).all():
        raise ValueError("B0 role timestamps must be finite")
    return frame.loc[(starts >= start_s) & (ends < end_s)].copy().reset_index(drop=True)


def exact_common_support(b0_prn, full_windows):
    """Intersect Full/B0 windows only when their exact numeric PRN sets match."""
    required = {"window_start_s", "prn"}
    missing = required - set(b0_prn.columns)
    if missing or not {"window_start_s", "prns"}.issubset(full_windows.columns):
        raise ValueError("Full/B0 support inputs lack required columns")
    normalized = b0_prn.assign(_numeric_prn=[_prn_number(x) for x in b0_prn["prn"]])
    if normalized.duplicated(["window_start_s", "_numeric_prn"]).any():
        raise ValueError("duplicate B0 window/PRN scientific rows")
    b0_sets = {
        float(t): tuple(sorted(group["_numeric_prn"].tolist()))
        for t, group in normalized.groupby("window_start_s", sort=True)
    }
    keep = []
    for index, row in full_windows.iterrows():
        t = float(row["window_start_s"])
        if b0_sets.get(t) == tuple(sorted(_prn_number(x) for x in row["prns"])):
            keep.append(index)
    full = full_windows.loc[keep].copy().reset_index(drop=True)
    times = set(float(x) for x in full["window_start_s"])
    b0 = b0_prn[b0_prn["window_start_s"].astype(float).isin(times)].copy()
    b0 = b0.sort_values(["window_start_s", "prn"]).reset_index(drop=True)
    return b0, full


def adapt_b0_exact_support(b0_prn, full_windows, *, score_column):
    """Aggregate executable B0 PRN scores only on exact Full window support."""
    if score_column not in b0_prn.columns:
        raise ValueError("B0 score column is absent")
    b0, full = exact_common_support(b0_prn, full_windows)
    full_support = {float(row.window_start_s): list(map(_prn_number, row.prns))
                    for row in full.itertuples(index=False)}
    rows = []
    for start, group in b0.groupby("window_start_s", sort=True):
        value = group[score_column].to_numpy(dtype=float)
        if not np.all(np.isfinite(value)):
            raise ValueError("B0 exact-support score is nonfinite")
        rows.append({"window_start_s": float(start), "prns": full_support[float(start)],
                     "score": float(np.mean(value))})
    return rows


def build_scheduled_node_table(receiver_root: str | Path, roles: dict[str, tuple[float, float]]):
    """Derive the frozen nine prompt-normalized means on the global 0.5-s schedule.

    Role-specific run IDs force the frozen scorer to reset its 12-window sequence
    at every boundary. Source rows are never copied from historic score tables.
    """
    paths = sorted((Path(receiver_root) / "raw").glob("epl_tracking_ch_*.mat"))
    pd = _pandas()
    if not paths:
        raise ValueError("no B0 receiver tracking MAT files")
    rows: list[dict[str, object]] = []
    for path in paths:
        with h5py.File(path, "r") as handle:
            required = {"PRN", "PRN_start_sample_count", *TAP_FIELDS}
            missing = required - set(handle.keys())
            if missing:
                raise ValueError(f"B0 tracking MAT missing fields: {sorted(missing)}")
            prn = np.asarray(handle["PRN"]).reshape(-1).astype(int)
            time = np.asarray(handle["PRN_start_sample_count"]).reshape(-1).astype(np.float64) / SAMPLE_RATE_HZ
            taps = np.column_stack([np.asarray(handle[name]).reshape(-1).astype(float) for name in TAP_FIELDS])
        finite = np.isfinite(time) & np.isfinite(taps).all(axis=1) & (taps[:, 4] >= 0)
        prn, time, taps = prn[finite], time[finite], taps[finite]
        for role, (role_start, role_end) in roles.items():
            for sat in sorted(set(prn[(time >= role_start) & (time < role_end)])):
                sat_mask = prn == sat
                for start in np.arange(role_start, role_end - .5, .5):
                    mask = sat_mask & (time >= start) & (time < start + 1.0)
                    if int(mask.sum()) < 4:
                        continue
                    normalized = taps[mask] / (taps[mask, 4, None] + EPSILON)
                    row = {
                        "run_id": f"cleanStatic-{role}", "prn": f"G{int(sat):02d}",
                        "window_bin_s": float(start + .5), "window_start_s": float(start),
                        "window_end_s": float(start + 1.), "window_mid_s": float(start + .5),
                    }
                    row.update({f"tap_{name}_rel_prompt_mean": float(np.mean(normalized[:, index])) for index, name in enumerate(TAP_NAMES)})
                    rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("B0 scheduled node table is empty")
    if frame.duplicated(["run_id", "window_start_s", "prn"]).any():
        raise ValueError("duplicate scheduled B0 node rows")
    return frame.sort_values(["run_id", "prn", "window_start_s"]).reset_index(drop=True)


def build_protected_scheduled_node_table(paths, *, gate, scenario,
                                         roles: dict[str, tuple[float, float]]):
    """Build B0 nodes through the protected gate and retain exact epoch support."""
    pd = _pandas(); rows = []
    datasets = ("PRN", "PRN_start_sample_count", *TAP_FIELDS)
    for path in map(Path, paths):
        values = {name: np.asarray(value).reshape(-1) for name, value in
                  gate.read_h5(path, datasets=datasets, scenario=scenario,
                               phase="all_frozen_phases", purpose="protected B0 nine-tap rows").items()}
        length = len(values["PRN"])
        if any(len(value) != length for value in values.values()):
            raise ValueError("protected B0 tracking MAT shape mismatch")
        prn = values["PRN"].astype(int)
        sample = values["PRN_start_sample_count"].astype(np.int64)
        time = sample.astype(np.float64) / SAMPLE_RATE_HZ
        taps = np.column_stack([values[name].astype(float) for name in TAP_FIELDS])
        finite = np.isfinite(time) & np.isfinite(taps).all(axis=1) & (taps[:, 4] >= 0)
        prn, sample, time, taps = prn[finite], sample[finite], time[finite], taps[finite]
        for role, (role_start, role_end) in roles.items():
            for sat in sorted(set(prn[(time >= role_start) & (time < role_end)])):
                sat_mask = prn == sat
                for start in np.arange(role_start, role_end - .5, .5):
                    mask = sat_mask & (time >= start) & (time < start + 1.0)
                    if int(mask.sum()) < 4: continue
                    epoch_ids = tuple(sorted(set(map(int, sample[mask] // (SAMPLE_RATE_HZ // 50)))))
                    normalized = taps[mask] / (taps[mask, 4, None] + EPSILON)
                    row = {"run_id": f"{scenario}-{role}", "prn": f"G{int(sat):02d}",
                           "window_bin_s": float(start + .5), "window_start_s": float(start),
                           "window_end_s": float(start + 1.), "window_mid_s": float(start + .5),
                           "phase": role, "epoch_ids_json": __import__("json").dumps(epoch_ids, separators=(",", ":"))}
                    row.update({f"tap_{name}_rel_prompt_mean": float(np.mean(normalized[:, index]))
                                for index, name in enumerate(TAP_NAMES)})
                    rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty or frame.duplicated(["run_id", "window_start_s", "prn"]).any():
        raise RuntimeError("protected B0 scheduled node table is empty or duplicated")
    return frame.sort_values(["run_id", "prn", "window_start_s"]).reset_index(drop=True)

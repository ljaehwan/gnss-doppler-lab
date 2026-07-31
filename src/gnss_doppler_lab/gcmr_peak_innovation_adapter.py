"""Causal adapter from GNSS-SDR E/P/L windows to GCMR-PI ``EventRecord``.

The adapter deliberately only constructs records.  It contains no fitting,
calibration, scoring, or attack-label API.  PRN labels are used solely to join
peak windows with geometry pairs; they are never part of a numeric feature.

After copying this file into ``gnss_doppler_lab``, its default event factory is
``.gcmr_peak_innovation_pipeline.EventRecord``.  Tests and integrations may
supply an equivalent ``event_record_type`` explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np

REQUIRED_TAPS = ("E", "P", "L")


class CausalEventBuildError(ValueError):
    """The requested relation event cannot form a complete causal PI record."""


@dataclass(frozen=True)
class PeakWindowRecord:
    """One finite, half-open-window E/P/L aggregate for a single PRN.

    ``epl`` and ``cn0`` are aggregate values computed *only* from samples in
    ``[window_start_s, window_end_s)``.  Keeping them as plain numpy payloads
    makes the adapter usable without a raw-MAT path.
    """

    window_start_s: float
    window_end_s: float
    epl: np.ndarray
    cn0: float
    tap_names: tuple[str, ...] = REQUIRED_TAPS

    def validate(self) -> None:
        start, end = float(self.window_start_s), float(self.window_end_s)
        if not (math.isfinite(start) and math.isfinite(end) and start < end):
            raise ValueError("peak window must be finite and strictly ordered")
        if tuple(self.tap_names) != REQUIRED_TAPS:
            raise ValueError("only exactly ('E', 'P', 'L') real taps are accepted; VE/VL/placeholders are forbidden")
        epl = np.asarray(self.epl, dtype=np.float64)
        if epl.shape != (3,) or not np.isfinite(epl).all() or np.allclose(epl, 0.0):
            raise ValueError("E/P/L aggregate must be a finite non-placeholder vector with shape (3,)")
        if not math.isfinite(float(self.cn0)):
            raise ValueError("window CN0 aggregate must be finite")


def _validate_series(series: Any, expected_tap_names: tuple[str, ...] = REQUIRED_TAPS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate a ``TrackingPeakSeries``-compatible value and return arrays."""
    if tuple(getattr(series, "tap_names", ())) != tuple(expected_tap_names):
        raise ValueError(f"tracking series tap layout must be exactly {tuple(expected_tap_names)!r}")
    time_s = np.asarray(getattr(series, "time_s"), dtype=np.float64)
    magnitudes = np.asarray(getattr(series, "magnitudes"), dtype=np.float64)
    cn0 = np.asarray(getattr(series, "cn0_db_hz"), dtype=np.float64)
    if time_s.ndim != 1 or len(time_s) == 0 or magnitudes.shape != (len(time_s), len(expected_tap_names)) or cn0.shape != (len(time_s),):
        raise ValueError("TrackingPeakSeries must have aligned nonempty time_s, magnitudes [N,3], and cn0_db_hz")
    if not np.isfinite(time_s).all() or not np.isfinite(magnitudes).all() or not np.isfinite(cn0).all():
        raise ValueError("tracking peak inputs must be finite")
    if np.any(np.diff(time_s) < 0):
        raise ValueError("tracking peak times must be nondecreasing")
    if np.any(np.all(magnitudes == 0.0, axis=0)):
        raise ValueError("E/P/L tap columns must be real measurements, not all-zero placeholders")
    return time_s, magnitudes, cn0


def aggregate_peak_windows(series: Any, windows: Sequence[tuple[float, float]], *, min_epochs: int = 1) -> list[PeakWindowRecord]:
    """Aggregate a ``TrackingPeakSeries`` into explicit half-open E/P/L windows.

    This is pure numpy aggregation.  It does not interpolate or cross a window
    boundary, so changing a target/future sample cannot alter an earlier record.
    """
    if int(min_epochs) != min_epochs or min_epochs < 1:
        raise ValueError("min_epochs must be a positive integer")
    time_s, magnitudes, cn0 = _validate_series(series)
    output: list[PeakWindowRecord] = []
    for raw_start, raw_end in windows:
        start, end = float(raw_start), float(raw_end)
        if not (math.isfinite(start) and math.isfinite(end) and start < end):
            raise ValueError("requested windows must be finite and strictly ordered")
        mask = (time_s >= start) & (time_s < end)
        if int(mask.sum()) < min_epochs:
            continue
        output.append(PeakWindowRecord(start, end, np.mean(magnitudes[mask], axis=0), float(np.mean(cn0[mask]))))
    return output


def _prn_label(prn: int | str) -> str:
    """Canonical GPS label used only as a mapping key, never as a feature."""
    if isinstance(prn, (bool, np.bool_)):
        raise ValueError("PRN must be a GPS integer or G-prefixed integer")
    if isinstance(prn, (int, np.integer)):
        value = int(prn)
    else:
        text = str(prn).strip().upper()
        if text.startswith("G"):
            text = text[1:]
        if not text.isdigit():
            raise ValueError(f"unsupported PRN identifier: {prn!r}")
        value = int(text)
    if not 1 <= value <= 32:
        raise ValueError(f"GPS PRN must be in [1, 32]: {prn!r}")
    return f"G{value:02d}"


def _event_pairs(relation_event: Any) -> tuple[float, float, dict[tuple[str, str], np.ndarray], tuple[str, ...], dict[str, float]]:
    """Read only the first three symmetric relation conditions from an event."""
    try:
        start, end = float(relation_event.window_start_s), float(relation_event.window_end_s)
        pairs = np.asarray(relation_event.pair_prns)
        conditions = np.asarray(relation_event.conditions, dtype=np.float64)
    except AttributeError as exc:
        raise ValueError("relation_event must expose window_start_s, window_end_s, pair_prns, and conditions") from exc
    if not (math.isfinite(start) and math.isfinite(end) and start < end):
        raise ValueError("relation event window must be finite and strictly ordered")
    if pairs.ndim != 2 or pairs.shape[1:] != (2,) or conditions.ndim != 2 or conditions.shape[0] != len(pairs) or conditions.shape[1] < 3:
        raise ValueError("relation event must have pair_prns [P,2] and at least three condition columns")
    pair_conditions: dict[tuple[str, str], np.ndarray] = {}
    node_values: dict[str, list[float]] = {}
    for pair, raw_condition in zip(pairs, conditions):
        if len(pair) != 2:
            raise ValueError("relation event pair must have two PRNs")
        a, b = _prn_label(pair[0]), _prn_label(pair[1])
        if a == b:
            raise ValueError("relation event cannot contain a self pair")
        key = tuple(sorted((a, b)))
        condition = np.asarray(raw_condition[:3], dtype=np.float64).copy()
        if not np.isfinite(condition).all():
            raise ValueError("first three relation conditions must be finite [los_dot,min_elevation_sin,max_elevation_sin]")
        if condition[1] > condition[2]:
            raise ValueError("relation elevation conditions must be ordered min <= max")
        if key in pair_conditions:
            raise ValueError(f"duplicate relation pair: {key}")
        pair_conditions[key] = condition
        # Pair conditions are symmetric and do not identify either endpoint.
        # Assign each endpoint the pair midpoint, then average incident pairs.
        midpoint = float((condition[1] + condition[2]) / 2.0)
        node_values.setdefault(a, []).append(midpoint)
        node_values.setdefault(b, []).append(midpoint)
    prns = tuple(sorted(node_values))
    elevations = {prn: float(np.mean(values)) for prn, values in node_values.items()}
    return start, end, pair_conditions, prns, elevations


def _canonical_windows(records: Sequence[PeakWindowRecord], prn: str) -> list[PeakWindowRecord]:
    result = list(records)
    for record in result:
        if not isinstance(record, PeakWindowRecord):
            raise ValueError(f"peak_windows[{prn}] must contain PeakWindowRecord values")
        record.validate()
    return sorted(result, key=lambda r: (float(r.window_start_s), float(r.window_end_s)))


def _exact_target(records: Sequence[PeakWindowRecord], start: float, end: float, prn: str) -> PeakWindowRecord:
    matches = [r for r in records if float(r.window_start_s) == start and float(r.window_end_s) == end]
    if len(matches) != 1:
        raise CausalEventBuildError(f"{prn} needs exactly one E/P/L target aggregate for relation window [{start}, {end}); found {len(matches)}")
    return matches[0]


def _event_record_type() -> Callable[..., Any]:
    try:
        return importlib.import_module(".gcmr_peak_innovation_pipeline", __package__).EventRecord
    except (ImportError, AttributeError, TypeError) as exc:
        raise ImportError("EventRecord is unavailable; copy this module into gnss_doppler_lab or pass event_record_type explicitly") from exc


def build_event_record(relation_event: Any, peak_windows: Mapping[int | str, Sequence[PeakWindowRecord]], *, history_window: int, event_record_type: Callable[..., Any] | None = None) -> Any:
    """Build one variable-PRN GCMR-PI ``EventRecord`` with strictly prior history.

    Every selected PRN comes from ``relation_event.pair_prns``.  All complete
    relation pairs must have an exact target E/P/L window and ``history_window``
    valid *window aggregates whose end is no later than target start*.  Thus the
    target window and all future/overlapping windows are excluded from GRU input.
    """
    if int(history_window) != history_window or history_window < 1:
        raise ValueError("history_window must be a positive integer")
    start, end, pair_conditions, prns, elevations = _event_pairs(relation_event)
    if len(prns) < 2:
        raise CausalEventBuildError("relation event provides fewer than two PRNs")
    expected = {(a, b) for i, a in enumerate(prns) for b in prns[i + 1:]}
    missing_pairs = expected.difference(pair_conditions)
    if missing_pairs:
        raise CausalEventBuildError(f"relation event lacks complete pair conditions for PRNs {prns}: missing {sorted(missing_pairs)}")

    normalized: dict[str, list[PeakWindowRecord]] = {}
    for raw_prn, records in peak_windows.items():
        label = _prn_label(raw_prn)
        if label in normalized:
            raise ValueError(f"duplicate peak-window mapping key after PRN normalization: {label}")
        normalized[label] = _canonical_windows(records, label)

    targets: list[np.ndarray] = []
    histories: dict[str, np.ndarray] = {}
    cn0: list[float] = []
    unavailable: list[str] = []
    for prn in prns:
        records = normalized.get(prn, [])
        try:
            target = _exact_target(records, start, end, prn)
        except CausalEventBuildError as exc:
            unavailable.append(str(exc))
            continue
        # Strict causality: aggregation windows overlapping or following target
        # are impossible history candidates even if their starts precede ``end``.
        prior = [r for r in records if float(r.window_end_s) <= start]
        if len(prior) < history_window:
            unavailable.append(f"{prn} has {len(prior)} valid prior E/P/L windows before {start}; needs {history_window}")
            continue
        history = np.stack([r.epl for r in prior[-history_window:]]).astype(np.float64, copy=True)
        targets.append(np.asarray(target.epl, dtype=np.float64).copy())
        histories[prn] = history
        cn0.append(float(target.cn0))
    if unavailable:
        raise CausalEventBuildError("cannot build causal EventRecord: " + "; ".join(unavailable))

    event_type = _event_record_type() if event_record_type is None else event_record_type
    record = event_type(float(end), prns, np.asarray(targets, dtype=np.float64), histories,
                        np.asarray(cn0, dtype=np.float64),
                        np.asarray([elevations[p] for p in prns], dtype=np.float64), pair_conditions)
    # Validate at the boundary when using the target EventRecord implementation.
    validate = getattr(record, "validate", None)
    if callable(validate):
        validate(int(history_window))
    return record


@dataclass(frozen=True)
class PeakWindowIndex:
    """Indexed aggregates for one contiguous PRN segment.

    Requested windows use ``searchsorted(..., side='left')`` plus prefix sums,
    preserving exact ``[start,end)`` membership without one raw scan per event.
    """
    records: Mapping[tuple[float, float], PeakWindowRecord]

    def target(self, start: float, end: float) -> PeakWindowRecord | None:
        return self.records.get((float(start), float(end)))

    def history_before(self, start: float, count: int) -> list[PeakWindowRecord] | None:
        prior = sorted((r for r in self.records.values() if r.window_end_s <= start),
                       key=lambda r: (r.window_end_s, r.window_start_s))
        return prior[-count:] if len(prior) >= count else None


def index_peak_windows(series: Any, windows: Sequence[tuple[float, float]], *, min_epochs: int = 1, expected_tap_names: tuple[str, ...] = REQUIRED_TAPS) -> PeakWindowIndex:
    """Index only target and requested historical E/P/L windows for one segment."""
    if int(min_epochs) != min_epochs or min_epochs < 1:
        raise ValueError("min_epochs must be a positive integer")
    time_s, magnitudes, cn0 = _validate_series(series, expected_tap_names)
    requested = sorted({(float(a), float(b)) for a, b in windows})
    prefix_mag = np.vstack((np.zeros((1, magnitudes.shape[1])), np.cumsum(magnitudes, axis=0)))
    prefix_cn0 = np.r_[0.0, np.cumsum(cn0)]
    output: dict[tuple[float, float], PeakWindowRecord] = {}
    for start, end in requested:
        if not (math.isfinite(start) and math.isfinite(end) and start < end):
            raise ValueError("requested windows must be finite and strictly ordered")
        # left/left implements precisely [start,end): epochs equal to end excluded.
        lo = int(np.searchsorted(time_s, start, side="left"))
        hi = int(np.searchsorted(time_s, end, side="left"))
        n = hi - lo
        if n < min_epochs:
            continue
        output[(start, end)] = PeakWindowRecord(start, end, (prefix_mag[hi] - prefix_mag[lo]) / n,
                                                  float((prefix_cn0[hi] - prefix_cn0[lo]) / n), tuple(expected_tap_names))
    return PeakWindowIndex(output)


def build_event_record_indexed(relation_event: Any, peak_indexes: Mapping[int | str, Sequence[PeakWindowIndex]], *, history_window: int, event_record_type: Callable[..., Any] | None = None) -> Any:
    """Build a causal record from one contiguous index; never cross reacquisition segments."""
    start, end, pair_conditions, prns, elevations = _event_pairs(relation_event)
    if int(history_window) != history_window or history_window < 1:
        raise ValueError("history_window must be a positive integer")
    expected = {(a, b) for i, a in enumerate(prns) for b in prns[i + 1:]}
    if expected.difference(pair_conditions):
        raise CausalEventBuildError("relation event lacks complete pair conditions")
    normalized = {_prn_label(k): list(v) for k, v in peak_indexes.items()}
    targets=[]; histories={}; cn0=[]; unavailable=[]
    for prn in prns:
        found=[]
        for index in normalized.get(prn, []):
            target=index.target(start,end)
            if target is not None:
                found.append((target, index.history_before(start, int(history_window))))
        valid=[x for x in found if x[1] is not None]
        if len(valid) != 1:
            unavailable.append(f"{prn} needs exactly one same-segment target and {history_window} prior windows; found {len(valid)}")
            continue
        target, history = valid[0]
        targets.append(np.asarray(target.epl, dtype=np.float64).copy())
        histories[prn]=np.stack([r.epl for r in history]).astype(np.float64,copy=True)
        cn0.append(float(target.cn0))
    if unavailable:
        raise CausalEventBuildError("cannot build causal EventRecord: " + "; ".join(unavailable))
    event_type = _event_record_type() if event_record_type is None else event_record_type
    record=event_type(float(end),prns,np.asarray(targets),histories,np.asarray(cn0),np.asarray([elevations[p] for p in prns]),pair_conditions)
    validate=getattr(record,"validate",None)
    if callable(validate): validate(int(history_window))
    return record


__all__ = ["REQUIRED_TAPS", "CausalEventBuildError", "PeakWindowRecord", "PeakWindowIndex", "aggregate_peak_windows", "index_peak_windows", "build_event_record", "build_event_record_indexed"]

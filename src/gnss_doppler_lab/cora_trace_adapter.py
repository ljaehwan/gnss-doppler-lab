"""Native and authenticated legacy tracker adapters for CORA raw-IQ epochs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import h5py
import numpy as np

from .trace_native_1ms import RECORD_DTYPE, read_records


@dataclass(frozen=True)
class EpochAction:
    prn: int
    channel: int
    receiver_timestamp_s: float
    raw_start_sample: int
    raw_end_sample: int
    cn0_db_hz: float
    carrier_doppler_hz: float
    action: np.void
    source_path: str
    source_row: int
    adapter: str


def _channel(path: Path) -> int:
    match = re.search(r"_ch_(\d+)\.(?:bin|mat)$", path.name)
    if not match:
        raise ValueError(f"channel absent from {path}")
    return int(match.group(1))


def _nearest(sorted_values: np.ndarray, target: float) -> int:
    index = int(np.searchsorted(sorted_values, target))
    candidates = [i for i in (index - 1, index) if 0 <= i < len(sorted_values)]
    return min(candidates, key=lambda i: abs(float(sorted_values[i]) - target))


class NativeTraceIndex:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.by_prn: dict[int, tuple[Path, object, np.ndarray]] = {}
        for path in sorted(self.root.glob("trace_native_1ms_ch_*.bin")):
            header, records = read_records(path)
            if not len(records):
                continue
            for prn in np.unique(records["prn"]):
                if int(prn) == 0:
                    continue
                mask = records["prn"] == prn
                subset = records[mask]
                prior = self.by_prn.get(int(prn))
                if prior is None or len(subset) > len(prior[2]):
                    self.by_prn[int(prn)] = (path, header, subset)

    def select(self, prn: int, target_s: float, *, tolerance_s: float = 0.00075) -> EpochAction:
        path, header, records = self.by_prn[prn]
        index = _nearest(records["receiver_timestamp_s"], target_s)
        row = records[index]
        if abs(float(row["receiver_timestamp_s"]) - target_s) > tolerance_s:
            raise ValueError(f"no native row within tolerance for PRN {prn} at {target_s}")
        if not int(row["valid_tracking"]) or not int(row["valid_lock"]):
            raise ValueError(f"invalid native lock for PRN {prn} at {target_s}")
        start = int(row["raw_interval_start_sample"]); end = int(row["raw_interval_end_sample"])
        if end <= start or end - start != int(row["action_used_interval_length_samples"]):
            raise ValueError("native raw/action interval mismatch")
        return EpochAction(
            prn=prn, channel=int(row["channel"]), receiver_timestamp_s=float(row["receiver_timestamp_s"]),
            raw_start_sample=start, raw_end_sample=end, cn0_db_hz=float(row["cn0_db_hz"]),
            carrier_doppler_hz=float(row["action_used_carrier_doppler_hz"]), action=row,
            source_path=str(path), source_row=index, adapter="native_trace_v2",
        )


class LegacyTraceIndex:
    """Map legacy complex9 rows to the 1 ms interval ending at their dump stamp.

    The conversion is fixed by the receiver source semantics and independently
    checked against DS3: ``aux1`` is residual code phase in samples, the row's
    code/carrier NCO is current, and PRN_start_sample_count is the exclusive raw
    endpoint.  No logged complex tap enters the CORA token.
    """
    def __init__(self, root: str | Path, sample_rate_hz: float):
        self.root = Path(root); self.sample_rate_hz = float(sample_rate_hz)
        self.by_prn: dict[int, dict[str, object]] = {}
        required = ("PRN", "PRN_start_sample_count", "CN0_SNV_dB_Hz", "carrier_lock_test",
                    "carrier_doppler_hz", "code_freq_chips", "aux1", "acc_carrier_phase_rad")
        for path in sorted(self.root.glob("epl_tracking_ch_*.mat")):
            with h5py.File(path, "r") as handle:
                if any(name not in handle for name in required):
                    continue
                arrays = {name: np.asarray(handle[name]).reshape(-1) for name in required}
            if len(arrays["PRN"]) < 8:
                continue
            for prn in np.unique(arrays["PRN"]):
                if int(prn) == 0:
                    continue
                indices = np.flatnonzero(arrays["PRN"] == prn)
                prior = self.by_prn.get(int(prn))
                if prior is None or len(indices) > len(prior["indices"]):
                    self.by_prn[int(prn)] = {"path": path, "channel": _channel(path),
                                              "arrays": arrays, "indices": indices}

    def select(self, prn: int, target_s: float, *, tolerance_s: float = 0.011) -> EpochAction:
        item = self.by_prn[prn]; arrays = item["arrays"]; indices = item["indices"]
        stamps = arrays["PRN_start_sample_count"][indices].astype(np.uint64)
        target_endpoint = target_s * self.sample_rate_hz
        local = _nearest(stamps, target_endpoint); row_index = int(indices[local])
        endpoint = int(arrays["PRN_start_sample_count"][row_index])
        actual_s = endpoint / self.sample_rate_hz
        if abs(actual_s - target_s) > tolerance_s:
            raise ValueError(f"no legacy row within tolerance for PRN {prn} at {target_s}")
        if float(arrays["carrier_lock_test"][row_index]) < 0.85:
            raise ValueError(f"invalid legacy lock for PRN {prn} at {target_s}")
        code_rate = float(arrays["code_freq_chips"][row_index])
        doppler = float(arrays["carrier_doppler_hz"][row_index])
        residual_samples = float(arrays["aux1"][row_index])
        interval = int(round(1023.0 * self.sample_rate_hz / code_rate))
        action = np.zeros(1, dtype=RECORD_DTYPE)[0]
        action["action_used_code_nco_rate_chips_s"] = code_rate
        action["action_used_carrier_doppler_hz"] = doppler
        action["action_used_residual_code_phase_samples"] = residual_samples
        action["action_used_residual_code_phase_chips"] = residual_samples * code_rate / self.sample_rate_hz
        action["action_used_code_phase_step_chips_per_sample"] = code_rate / self.sample_rate_hz
        action["action_used_carrier_phase_step_rad_per_sample"] = 2.0 * np.pi * doppler / self.sample_rate_hz
        # Only a global rotation; the cross-cumulant is invariant to it.
        action["action_used_residual_carrier_phase_rad"] = float(arrays["acc_carrier_phase_rad"][row_index]) % (2.0 * np.pi)
        action["action_used_interval_length_samples"] = interval
        return EpochAction(
            prn=prn, channel=int(item["channel"]), receiver_timestamp_s=actual_s,
            raw_start_sample=endpoint - interval, raw_end_sample=endpoint,
            cn0_db_hz=float(arrays["CN0_SNV_dB_Hz"][row_index]), carrier_doppler_hz=doppler,
            action=action, source_path=str(item["path"]), source_row=row_index,
            adapter="legacy_complex9_authenticated_endpoint",
        )


def validate_epoch_sequence(epochs: list[EpochAction], raw_sample_count: int) -> dict[str, object]:
    failures = []
    for row in epochs:
        if not (0 <= row.raw_start_sample < row.raw_end_sample <= raw_sample_count):
            failures.append("RAW_IQ_BOUNDS")
        if row.action["action_used_interval_length_samples"] != row.raw_end_sample - row.raw_start_sample:
            failures.append("INTERVAL_ACTION_MISMATCH")
    by_prn: dict[int, list[EpochAction]] = {}
    for row in epochs:
        by_prn.setdefault(row.prn, []).append(row)
    for rows in by_prn.values():
        ordered = sorted(rows, key=lambda value: value.receiver_timestamp_s)
        if any(b.raw_start_sample <= a.raw_start_sample for a, b in zip(ordered, ordered[1:])):
            failures.append("NON_MONOTONE_SAMPLE_MAPPING")
        if len({row.channel for row in ordered}) != 1:
            failures.append("CHANNEL_REASSIGNMENT")
    return {"status": "PASS" if not failures else "FAIL", "failures": sorted(set(failures)),
            "epoch_count": len(epochs), "prns": sorted(by_prn)}

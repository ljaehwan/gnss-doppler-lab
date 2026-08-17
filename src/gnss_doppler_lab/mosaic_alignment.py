"""MOSAIC-GNSS Stage-0A native epoch alignment helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .trace_native_1ms import TAPS, complex_taps, read_records, validate_dump_files


@dataclass(frozen=True)
class MosaicEpochTable:
    rows: list[dict[str, object]]
    sample_rate_hz: float


def stable_epoch_rows(
    dump_dir: str | Path,
    *,
    dataset: str,
    recording: str,
    minimum_cn0_db_hz: float = 28.0,
    minimum_lock: float = 0.85,
    limit: int | None = 200,
) -> MosaicEpochTable:
    """Extract compact native 1-ms rows keyed by PRN + loop sequence + raw span."""
    out: list[dict[str, object]] = []
    sample_rates: set[float] = set()
    for path in sorted(Path(dump_dir).glob("trace_native_1ms_ch_*.bin")):
        header, records = read_records(path, mmap=True)
        sample_rates.add(header.sample_rate_hz)
        taps = complex_taps(records)
        ok = (
            (records["valid_tracking"] == 1)
            & (records["valid_lock"] == 1)
            & (records["cn0_db_hz"] >= minimum_cn0_db_hz)
            & (records["carrier_lock_test"] >= minimum_lock)
            & np.isfinite(taps.real).all(axis=1)
            & np.isfinite(taps.imag).all(axis=1)
        )
        for idx in np.flatnonzero(ok):
            rec = records[idx]
            tap_list = [[float(v.real), float(v.imag)] for v in taps[idx]]
            out.append({
                "dataset": dataset,
                "recording": recording,
                "prn": int(rec["prn"]),
                "channel": int(rec["channel"]),
                "loop_sequence": int(rec["loop_sequence"]),
                "tracking_session_id": int(rec["tracking_session_id"]),
                "raw_sample_start": int(rec["raw_interval_start_sample"]),
                "raw_sample_end": int(rec["raw_interval_end_sample"]),
                "receiver_timestamp_s": float(rec["receiver_timestamp_s"]),
                "complex_taps_order": list(TAPS),
                "complex_nine_taps_iq": tap_list,
                "prompt_iq": tap_list[4],
                "dll_code_discriminator_chips": float(rec["dll_discriminator_chips"]),
                "pll_phase_error_cycles": float(rec["pll_phase_error_cycles"]),
                "fll_frequency_error_hz": float(rec["fll_frequency_error_hz"]),
                "code_frequency_chips_s": float(rec["action_used_code_nco_rate_chips_s"]),
                "carrier_doppler_hz": float(rec["action_used_carrier_doppler_hz"]),
                "residual_code_phase_chips": float(rec["action_used_residual_code_phase_chips"]),
                "carrier_phase_accumulator_rad": float(rec["action_used_carrier_phase_accumulator_rad"]),
                "receiver_applied_code_action_chips_s": float(rec["action_used_dll_filter_output_chips_s"]),
                "receiver_applied_carrier_action_hz": float(rec["action_used_pll_fll_filter_output_hz"]),
                "cn0_db_hz": float(rec["cn0_db_hz"]),
                "lock_tracking_quality": float(rec["carrier_lock_test"]),
                "navigation_bit_wipeoff_applied": bool(rec["navigation_bit_wipeoff_applied"]),
                "source_dump": str(path),
            })
            if limit is not None and len(out) >= limit:
                if len(sample_rates) != 1:
                    raise ValueError("multiple sample rates in one dump directory")
                return MosaicEpochTable(out, sample_rates.pop())
    if len(sample_rates) != 1:
        raise ValueError("dump directory contains no rows or multiple sample rates")
    return MosaicEpochTable(out, sample_rates.pop())


def validate_causal_alignment(dump_dirs: Iterable[str | Path]) -> dict[str, object]:
    paths: list[Path] = []
    for d in dump_dirs:
        paths.extend(sorted(Path(d).glob("trace_native_1ms_ch_*.bin")))
    return validate_dump_files(paths, minimum_prns=4)


def sample_bounds_status(rows: Iterable[dict[str, object]], raw_size_bytes: int, *, bytes_per_complex_sample: int = 2) -> dict[str, object]:
    total_samples = raw_size_bytes // bytes_per_complex_sample
    checked = 0
    bad = 0
    for row in rows:
        checked += 1
        start = int(row["raw_sample_start"]); end = int(row["raw_sample_end"])
        if start < 0 or end <= start or end > total_samples:
            bad += 1
    return {"status": "PASS" if bad == 0 else "FAIL", "rows_checked": checked, "bad_rows": bad, "raw_complex_samples": int(total_samples)}


def navigation_bit_provenance(rows: Iterable[dict[str, object]]) -> dict[str, object]:
    values = [bool(r.get("navigation_bit_wipeoff_applied", False)) for r in rows]
    # Wipeoff flag says what the receiver did to correlators, not the sign
    # sequence needed to synthesize raw-IQ counterfeit samples.
    return {
        "status": "UNAVAILABLE",
        "receiver_nav_wipeoff_flag_rows": int(sum(values)),
        "rows_considered": len(values),
        "reason": "No decoded nav-bit sequence or independently validated 20-ms Prompt sign provenance was found; refusing +1 fallback.",
    }

"""R1.3 cleanStatic-only raw-IQ reconstruction primitives.

All functions are deterministic and small enough for synthetic verification.  The
production runner owns I/O; this module owns the physical equations and gates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from typing import Iterable, Mapping, Sequence

import numpy as np

from gnss_doppler_lab.acquisition_surface import gps_l1ca_code

FS = 25_000_000.0
GLOBAL_OFFSETS = (-1000, -500, 0, 500, 1000)
REQUIRED_FIELDS = (
    "PRN", "PRN_start_sample_count", "carrier_doppler_hz", "code_freq_chips",
    "aux1", "Prompt_I", "Prompt_Q", "CN0_SNV_dB_Hz", "carrier_lock_test",
    "mat_row", "channel", "mat_sha256",
)


@dataclass(frozen=True)
class Candidate:
    nco_row: str = "previous"
    aux_row: str = "previous"
    remnant_sign: int = 1
    carrier_sign: int = -1
    global_offset: int = 0

    @property
    def name(self) -> str:
        values = asdict(self)
        return "_".join(f"{key}={values[key]}" for key in values)


def _digest(value) -> str:
    if isinstance(value, np.ndarray):
        payload = np.ascontiguousarray(value).view(np.uint8)
    elif isinstance(value, bytes):
        payload = value
    else:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256(payload).hexdigest()


def clean_only_guard(scenarios: Iterable[str]) -> bool:
    """Reject every input name except the one authorized benign recording."""
    if list(scenarios) != ["cleanStatic"]:
        raise ValueError("R1.3 accepts the authorized benign recording only")
    return True


def wide_grid() -> dict[str, list[float]]:
    return {"delay_chips": [round(-1 + i * .125, 3) for i in range(17)],
            "doppler_hz": list(range(-250, 251, 50))}


def _finite(row: Mapping) -> bool:
    numeric = REQUIRED_FIELDS[:9]
    return all(key in row for key in REQUIRED_FIELDS) and all(np.isfinite(float(row[k])) for k in numeric)


def filter_stable_triples(rows: Sequence[Mapping], raw_samples: int,
                          min_cn0: float = 28.0, min_lock: float = .85,
                          min_consumed_samples: int = 24_999,
                          max_consumed_samples: int = 25_001) -> list[tuple[dict, dict, dict]]:
    """Return adjacent same-channel/PRN triples whose current epoch is stable."""
    result = []
    for i in range(1, len(rows) - 1):
        triple = tuple(dict(rows[j]) for j in (i - 1, i, i + 1))
        if not all(_finite(r) for r in triple):
            continue
        prns = {int(r["PRN"]) for r in triple}
        channels = {str(r["channel"]) for r in triple}
        raw_stamps = [r["PRN_start_sample_count"] for r in triple]
        if any(float(x) != int(float(x)) for x in raw_stamps):
            continue
        samples = [int(x) for x in raw_stamps]
        cur = triple[1]
        if (len(prns) != 1 or next(iter(prns)) not in range(1, 33) or len(channels) != 1
                or not (0 <= samples[0] < samples[1] < samples[2] <= int(raw_samples))
                or [int(r["mat_row"]) for r in triple] != list(range(int(triple[0]["mat_row"]), int(triple[0]["mat_row"])+3))
                or min(np.diff(samples)) < int(min_consumed_samples)
                or max(np.diff(samples)) > int(max_consumed_samples)
                or any(float(r["CN0_SNV_dB_Hz"]) < min_cn0 for r in triple)
                or any(float(r["carrier_lock_test"]) < min_lock for r in triple)):
            continue
        result.append(triple)
    return result


def _row(triple: Sequence[Mapping], name: str) -> Mapping:
    try:
        return triple[{"previous": 0, "current": 1, "next": 2}[name]]
    except KeyError as exc:
        raise ValueError(f"invalid candidate row {name!r}") from exc


def interval_rows(triple: Sequence[Mapping], interval: str, global_offset: int = 0):
    if interval == "prev_to_cur":
        start, end, prompt = triple[0]["PRN_start_sample_count"], triple[1]["PRN_start_sample_count"], "current"
    elif interval == "cur_to_next":
        start, end, prompt = triple[1]["PRN_start_sample_count"], triple[2]["PRN_start_sample_count"], "next"
    else:
        raise ValueError("only exact adjacent intervals are physical")
    return int(start) + int(global_offset), int(end) + int(global_offset), prompt


def source_support(triple: Sequence[Mapping], vector_length: int) -> dict:
    """Map current Prompt to its fixed support and variable consume boundary.

    Row ``k-1`` stores the updated boundary/state used by the call producing
    row ``k``.  Its stamp is therefore that call's ``nitems_read``.
    """
    if len(triple) != 3 or int(vector_length) <= 0:
        raise ValueError("a stable triple and positive authenticated vector length are required")
    start = int(triple[0]["PRN_start_sample_count"])
    consumed_end = int(triple[1]["PRN_start_sample_count"])
    return {"authenticated": True, "length_samples": int(vector_length),
            "start_sample": start, "end_sample": start + int(vector_length),
            "consumed_start_sample": start, "consumed_end_sample": consumed_end,
            "consumed_length_samples": consumed_end - start}


def validate_candidate_rows(triple: Sequence[Mapping], candidate: Candidate) -> None:
    names = (candidate.nco_row, candidate.aux_row)
    referenced = [_row(triple, name) for name in names]
    if len({int(row["PRN"]) for row in referenced}) != 1:
        raise ValueError("candidate referenced rows with different PRNs")
    if candidate.remnant_sign not in (-1, 1) or candidate.carrier_sign not in (-1, 1):
        raise ValueError("candidate signs must be +/-1")


def code_replica(prn, n: int, fs: float, code_freq_chips: float,
                 aux1_samples: float, remnant_sign: int, delay_offset_chips: float,
                 *, base_phase: float = 0.0, replica_direction: int = 1):
    """Generate a delay-specific replica directly from the physical phase equation."""
    remnant_chips = float(aux1_samples) * float(code_freq_chips) / float(fs)
    phase = (float(base_phase) + replica_direction * np.arange(n, dtype=np.float64)
             * float(code_freq_chips) / float(fs) + remnant_sign * remnant_chips
             + float(delay_offset_chips))
    chip_index = np.floor(phase).astype(np.int64) % 1023
    return gps_l1ca_code(prn)[chip_index], _digest(chip_index)


def carrier_wipeoff(n: int, fs: float, tracker_doppler_hz: float,
                    doppler_offset_hz: float, carrier_sign: int):
    phase = carrier_sign * 2 * np.pi * (float(tracker_doppler_hz) + float(doppler_offset_hz)) * np.arange(n) / float(fs)
    value = np.exp(1j * phase)
    return value, _digest(value.astype(np.complex64))


def caf(iq: np.ndarray, prn, fs: float, code_freq_chips: float, aux1_samples: float,
        tracker_doppler_hz: float, candidate: Candidate, grid: Mapping[str, Sequence[float]] | None = None):
    grid = grid or wide_grid()
    values = np.empty((len(grid["doppler_hz"]), len(grid["delay_chips"])), dtype=float)
    chip_hashes, wipe_hashes = {}, {}
    for di, doppler in enumerate(grid["doppler_hz"]):
        wipe, wipe_hash = carrier_wipeoff(len(iq), fs, tracker_doppler_hz, doppler, candidate.carrier_sign)
        wipe_hashes[str(doppler)] = wipe_hash
        wiped = np.asarray(iq) * wipe
        for ci, delay in enumerate(grid["delay_chips"]):
            replica, chip_hash = code_replica(prn, len(iq), fs, code_freq_chips, aux1_samples,
                                               candidate.remnant_sign, delay,
                                               replica_direction=1)
            chip_hashes[str(delay)] = chip_hash
            values[di, ci] = abs(np.vdot(replica, wiped))
    flat = int(values.argmax()); di, ci = np.unravel_index(flat, values.shape)
    center_di = list(grid["doppler_hz"]).index(0); center_ci = list(grid["delay_chips"]).index(0)
    return {"peak_delay_offset_chips": grid["delay_chips"][ci],
            "peak_doppler_offset_hz": grid["doppler_hz"][di],
            "peak_magnitude": float(values[di, ci]),
            "center_magnitude": float(values[center_di, center_ci]),
            "exact_center": bool(di == center_di and ci == center_ci),
            "within_tolerance": bool(abs(grid["delay_chips"][ci]) <= .125 and abs(grid["doppler_hz"][di]) <= 50),
            "grid_boundary": bool(di in (0, values.shape[0]-1) or ci in (0, values.shape[1]-1)),
            "replica_chip_index_hash": chip_hashes, "carrier_wipeoff_hash": wipe_hashes,
            "result_field_hash": _digest(values)}


def candidate_application(iq: np.ndarray, triple: Sequence[Mapping], candidate: Candidate,
                          start: int, end: int, fs: float, result_field=None) -> dict:
    """Return separately auditable hashes of inputs actually applied."""
    validate_candidate_rows(triple, candidate)
    aux = _row(triple, candidate.aux_row); nco = _row(triple, candidate.nco_row)
    n = len(iq)
    replica, replica_hash = code_replica(aux["PRN"], n, fs, nco["code_freq_chips"], aux["aux1"],
                                         candidate.remnant_sign, 0, replica_direction=1)
    wipe, wipe_hash = carrier_wipeoff(n, fs, nco["carrier_doppler_hz"], 0, candidate.carrier_sign)
    prompt = triple[1]
    physical_result = result_field if result_field is not None else replica * wipe * np.asarray(iq)
    return {"raw_interval_content_sha256": _digest(np.asarray(iq)),
            "raw_interval_range_sha256": _digest([int(start), int(end)]),
            "replica_chip_indices_sha256": replica_hash,
            "carrier_wipeoff_sha256": wipe_hash,
            "aux_indices_sha256": _digest({"mat_row": int(aux["mat_row"]), "aux1": float(aux["aux1"])}),
            "nco_indices_sha256": _digest({"mat_row": int(nco["mat_row"]),
                                            "code_freq_chips": float(nco["code_freq_chips"]),
                                            "carrier_doppler_hz": float(nco["carrier_doppler_hz"])}),
            "prompt_indices_sha256": _digest({"mat_row": int(prompt["mat_row"]),
                                               "Prompt_I": float(prompt["Prompt_I"]),
                                               "Prompt_Q": float(prompt["Prompt_Q"])}),
            "aux_row_index": int(aux["mat_row"]), "aux1_value": float(aux["aux1"]),
            "nco_row_index": int(nco["mat_row"]),
            "code_freq_chips_value": float(nco["code_freq_chips"]),
            "carrier_doppler_hz_value": float(nco["carrier_doppler_hz"]),
            "prompt_row_index": int(prompt["mat_row"]),
            "prompt_i_value": float(prompt["Prompt_I"]), "prompt_q_value": float(prompt["Prompt_Q"]),
            "result_field_sha256": (physical_result if isinstance(physical_result, str)
                                      and len(physical_result) == 64 else _digest(physical_result))}


def roles_nonoverlap(intervals: Sequence[Mapping]) -> bool:
    roles = sorted({row["role"] for row in intervals})
    for i, left in enumerate(roles):
        for right in roles[i + 1:]:
            a = [x for x in intervals if x["role"] == left]
            b = [x for x in intervals if x["role"] == right]
            if any(x["start"] < y["end"] and y["start"] < x["end"] for x in a for y in b):
                return False
    return True


def gate_verdict(a1: bool, a2: bool, a3: bool, selected=None) -> dict:
    if not a1: verdict = "SOURCE_BINDING_INVALID"
    elif not a2: verdict = "RECONSTRUCTION_IMPLEMENTATION_INVALID"
    elif not a3: verdict = "TRACKER_RAW_ALIGNMENT_UNRESOLVED"
    else: verdict = "PHYSICAL_CENTER_VALID"
    passed = a1 and a2 and a3
    return {"A1_SOURCE_BINDING": "PASS" if a1 else "FAIL",
            "A2_IMPLEMENTATION_AND_INTERVAL_VALIDITY": "PASS" if a2 else "FAIL",
            "A3_MULTI_PRN_CENTER_RECOVERY": "PASS" if a3 else "FAIL",
            "verdict": verdict, "selected_alignment": selected if passed else None,
            "physics_no_go_claim": False}

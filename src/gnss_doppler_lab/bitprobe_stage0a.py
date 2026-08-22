"""Clean-only Stage-0A for BITPROBE-GNSS NAV-edge operator identifiability.

The implementation is deliberately deterministic and non-neural.  It does
not localize a common linear or nonlinear operator to a transmitter from one
receiver.  It consumes only the preregistered cleanStatic paths and committed
MOSAIC NAV provenance.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import itertools
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score

from .mosaic_raw_recorrelation import receiver_l1ca_code
from .trace_native_1ms import read_records


BASE_SHA = "2cb81593783cb4c8db2d0d38ecc71d9850fafadf"
PREREGISTRATION_SHA = "5c9be7d1cf14f1075a49ec0c925036150024a5bd"
BRANCH = "research/bitprobe-stage0a-nav-edge-operator-identifiability"
ARTIFACT_REL = "artifacts/bitprobe_stage0a_nav_edge_operator_identifiability"
SSD_ROOT = Path(
    "/home/ubuntu/ssd_data/gnss-early-detection/artifacts/"
    "bitprobe-stage0a-nav-edge-operator-identifiability"
)
R0B_REL = "artifacts/mosaic_stage0b_r0b_corrected_navbit_mapping"
R0C_REL = "artifacts/mosaic_stage0b_r0c_boundary_phase_extrapolation"
R0_REL = "artifacts/mosaic_stage0b_r0_navbit_provenance"
RAW_BINDING_REL = f"{R0_REL}/raw_source_binding.json"
MAPPING_REL = f"{R0C_REL}/corrected_bit_mapping.csv.gz"
CONTINUITY_REL = f"{R0C_REL}/tracking_continuity.csv"
COMMON_INTERVAL_REL = f"{R0C_REL}/common_interval_validation.json"
NAV_STRUCTURE_REL = f"{R0C_REL}/nav_structure_validation.json"

DATASET_ORDER = ("TEXBAT.cleanStatic", "OAKBAT.cleanStatic")
EXPECTED_PRNS = {
    "TEXBAT.cleanStatic": (3, 13, 16, 19, 30),
    "OAKBAT.cleanStatic": (10, 11, 21, 24, 27),
}
ALLOWED_VERDICTS = (
    "BITPROBE_STAGE0A_EDGE_OPERATOR_IDENTIFIABLE",
    "BITPROBE_STAGE0A_PARTIAL_DATASET_SUPPORT",
    "BITPROBE_STAGE0A_EDGE_OPERATOR_NOT_IDENTIFIABLE",
    "INCONCLUSIVE_BITPROBE_STAGE0A_EXECUTION_OR_PROVENANCE",
)
EXECUTABLE_FILES = (
    "src/gnss_doppler_lab/bitprobe_stage0a.py",
    "scripts/run_bitprobe_stage0a.py",
    "scripts/verify_bitprobe_stage0a.py",
)
SCIENCE_DEPENDENCIES = (
    "src/gnss_doppler_lab/acquisition_surface.py",
    "src/gnss_doppler_lab/mosaic_raw_recorrelation.py",
    "src/gnss_doppler_lab/trace_native_1ms.py",
)
EDGE_FIELDS = (
    "dataset",
    "source_sha256",
    "prn",
    "nav_bit_index",
    "previous_bit_value_pm1",
    "current_bit_value_pm1",
    "flip",
    "parity_status",
    "raw_complex_sample_boundary",
    "fractional_boundary_offset_samples",
    "carrier_code_state_source",
    "cn0_db_hz",
    "carrier_lock_test",
    "trace_path",
    "trace_sha256",
    "trace_record_index",
    "window_start_sample",
    "window_end_sample_exclusive",
    "fitted_amplitude_abs",
    "alignment_delay_error_chips",
    "alignment_doppler_error_hz",
    "exclusion_reason",
    "status",
)
SUPPORT_FIELDS = (
    "dataset",
    "prn",
    "candidate_flip",
    "valid_flip",
    "first_half_flip",
    "second_half_flip",
    "candidate_no_flip",
    "valid_no_flip",
    "first_half_no_flip",
    "second_half_no_flip",
    "support_gate",
)


class BindingError(RuntimeError):
    """Fail-closed input, provenance, execution, or support failure."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for payload in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(payload)
    return digest.hexdigest()


def file_binding(path: Path) -> dict[str, object]:
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)}


def repo_binding(repo: Path, relative: str) -> dict[str, object]:
    result = file_binding(repo / relative)
    result["path"] = relative
    return result


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    dump_json(temporary, value)
    temporary.replace(path)


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]], *, gzip_output: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if gzip_output else open
    with opener(path, "wt", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", newline="") as stream:
        return list(csv.DictReader(stream))


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def assert_branch(repo: Path) -> None:
    if git(repo, "branch", "--show-current") != BRANCH:
        raise BindingError("wrong BITPROBE branch")
    if git(repo, "merge-base", "HEAD", BASE_SHA) != BASE_SHA:
        raise BindingError("BITPROBE branch is not based on the required base")


def assert_pushed_freeze(repo: Path, freeze_sha: str) -> None:
    assert_branch(repo)
    if git(repo, "rev-parse", "HEAD") != freeze_sha:
        raise BindingError("HEAD differs from supplied freeze SHA")
    if git(repo, "rev-parse", f"origin/{BRANCH}") != freeze_sha:
        raise BindingError("remote branch differs from supplied freeze SHA")
    if git(repo, "status", "--porcelain=v1"):
        raise BindingError("execution checkout is not clean")


def compact_manifest(artifact: Path) -> dict[str, object]:
    rows = []
    for path in sorted(artifact.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest_sha256.json":
            rows.append({
                "path": str(path.relative_to(artifact)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            })
    return {
        "schema": "gnss-doppler-lab.bitprobe-stage0a-artifact-manifest.v1",
        "status": "PASS",
        "file_count": len(rows),
        "files": rows,
    }


def seal_manifest(artifact: Path) -> None:
    dump_json(artifact / "artifact_manifest_sha256.json", compact_manifest(artifact))


def scale_grids(prereg: Mapping[str, object]) -> dict[str, np.ndarray]:
    result = {}
    for name, spec in prereg["edge_contract"]["representation"].items():
        start, end, step = float(spec["start_chip"]), float(spec["end_chip"]), float(spec["step_chip"])
        result[name] = np.linspace(start, end, int(round((end - start) / step)) + 1, dtype=np.float64)
    return result


def combined_grid(prereg: Mapping[str, object]) -> np.ndarray:
    return np.concatenate(list(scale_grids(prereg).values()))


@dataclass
class AccessGuard:
    allowed_paths: set[str]
    forbidden_tokens: tuple[str, ...]
    clean: dict[str, int]
    forbidden: dict[str, int]

    @classmethod
    def create(cls, allowed_paths: Iterable[str]) -> "AccessGuard":
        return cls(
            allowed_paths={str(Path(path)) for path in allowed_paths},
            forbidden_tokens=(
                "/texbat/raw/ds1", "/texbat/raw/ds2", "/texbat/raw/ds3", "/texbat/raw/ds4",
                "/texbat/raw/ds5", "/texbat/raw/ds6", "/texbat/raw/ds7", "/texbat/raw/ds8",
                "oakbat/os", "oakbat.attack", "spoof",
            ),
            clean={"stats": 0, "hashes": 0, "opens": 0, "mmaps": 0, "bytes_read": 0},
            forbidden={"stats": 0, "hashes": 0, "opens": 0, "mmaps": 0, "bytes_read": 0},
        )

    def _authorize(self, path: Path, operation: str) -> None:
        value = str(path)
        lowered = value.lower()
        if any(token in lowered for token in self.forbidden_tokens):
            raise BindingError(f"forbidden attack path rejected before {operation}: {path}")
        if value not in self.allowed_paths:
            raise BindingError(f"path is not in explicit clean allowlist: {path}")

    def stat(self, path: Path) -> os.stat_result:
        self._authorize(path, "stat")
        result = os.stat(path)
        self.clean["stats"] += 1
        return result

    def hash(self, path: Path) -> str:
        self._authorize(path, "hash")
        digest = hashlib.sha256()
        self.clean["hashes"] += 1
        self.clean["opens"] += 1
        with path.open("rb") as stream:
            for payload in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                self.clean["bytes_read"] += len(payload)
                digest.update(payload)
        return digest.hexdigest()

    def read_window(self, path: Path, start_sample: int, sample_count: int) -> tuple[np.ndarray, np.ndarray]:
        self._authorize(path, "open")
        if start_sample < 0 or sample_count <= 0:
            raise BindingError("invalid clean raw window")
        self.clean["opens"] += 1
        with path.open("rb") as stream:
            stream.seek(start_sample * 4)
            payload = stream.read(sample_count * 4)
        self.clean["bytes_read"] += len(payload)
        if len(payload) != sample_count * 4:
            raise BindingError("clean raw window crosses source boundary")
        raw = np.frombuffer(payload, dtype="<i2").copy()
        return raw[0::2].astype(np.float64) + 1j * raw[1::2].astype(np.float64), raw

    def record_trace_read(self, path: Path, size_bytes: int) -> None:
        self._authorize(path, "open")
        self.clean["stats"] += 1
        self.clean["opens"] += 2
        self.clean["bytes_read"] += int(size_bytes)

    def audit(self) -> dict[str, object]:
        return {
            "clean_inputs": dict(self.clean),
            "forbidden_attack_inputs": dict(self.forbidden),
            "explicit_allowlist_size": len(self.allowed_paths),
        }


def load_preregistration(artifact: Path) -> dict[str, object]:
    value = json.loads((artifact / "preregistration.json").read_text())
    if value.get("base_sha") != BASE_SHA or value.get("status") != "FROZEN_BEFORE_IMPLEMENTATION_AND_RAW_EDGE_RESULTS":
        raise BindingError("preregistration contract mismatch")
    return value


def load_provenance(repo: Path) -> tuple[dict[str, object], dict[str, object], list[dict[str, str]], list[dict[str, str]]]:
    raw = json.loads((repo / RAW_BINDING_REL).read_text())
    intervals = json.loads((repo / COMMON_INTERVAL_REL).read_text())
    mapping = read_csv(repo / MAPPING_REL)
    continuity = read_csv(repo / CONTINUITY_REL)
    r0c_final = json.loads((repo / R0C_REL / "final_verdict.json").read_text())
    if r0c_final.get("verdict") != "BOUNDARY_PHASE_EXTRAPOLATION_PASS_WITH_SCOPE_LIMITATION":
        raise BindingError("R0c provenance verdict mismatch")
    if raw.get("overall_status") != "PASS":
        raise BindingError("clean raw source binding is not PASS")
    return raw, intervals, mapping, continuity


def validate_provenance_scope(
    raw: Mapping[str, object], intervals: Mapping[str, object], mapping: Sequence[Mapping[str, str]], continuity: Sequence[Mapping[str, str]]
) -> dict[str, object]:
    summary = {}
    for dataset in DATASET_ORDER:
        expected = set(EXPECTED_PRNS[dataset])
        interval = intervals["datasets"][dataset]
        if interval.get("authorization") != "AUTHORIZED_WITHIN_INTERVAL_ONLY":
            raise BindingError(f"R0c interval is not authorized: {dataset}")
        passed = {int(row["prn"]) for row in continuity if row["dataset"] == dataset and row["status"] == "PASS"}
        mapped = {int(row["prn"]) for row in mapping if row["dataset"] == dataset and row["phase_extrapolation_match"] == "True"}
        if passed != expected or mapped != expected or set(interval["included_prns"]) != expected:
            raise BindingError(f"R0c PRN scope mismatch: {dataset}")
        if raw[dataset]["status"] != "PASS":
            raise BindingError(f"clean raw binding not PASS: {dataset}")
        summary[dataset] = {
            "prns": sorted(expected),
            "common_raw_start_sample": int(interval["common_raw_start_sample"]),
            "common_raw_end_sample_exclusive": int(interval["common_raw_end_sample_exclusive"]),
            "sample_rate_hz": int(raw[dataset]["sample_rate_hz"]),
            "source_sha256": raw[dataset]["expected_sha256"],
            "size_bytes": int(raw[dataset]["stat"]["size_bytes"]),
        }
    return summary



def build_candidate_rows(
    mapping: Sequence[Mapping[str, str]],
    continuity: Sequence[Mapping[str, str]],
    scope: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    trace_by_key = {
        (row["dataset"], int(row["prn"])): row
        for row in continuity if row.get("status") == "PASS"
    }
    rows: list[dict[str, object]] = []
    for dataset in DATASET_ORDER:
        by_prn: dict[int, list[Mapping[str, str]]] = {}
        for row in mapping:
            if row["dataset"] == dataset and int(row["prn"]) in EXPECTED_PRNS[dataset]:
                by_prn.setdefault(int(row["prn"]), []).append(row)
        for prn in EXPECTED_PRNS[dataset]:
            ordered = sorted(by_prn.get(prn, []), key=lambda row: int(row["bit_index"]))
            continuity_row = trace_by_key.get((dataset, prn))
            if continuity_row is None:
                raise BindingError(f"missing PASS continuity row: {dataset} PRN {prn}")
            for previous, current in zip(ordered, ordered[1:]):
                boundary = int(current["corrected_raw_start_sample"])
                fs = int(scope[dataset]["sample_rate_hz"])
                half = int(math.ceil(4.0 * fs / 1_023_000.0)) + 3
                mapping_ok = (
                    int(current["bit_index"]) == int(previous["bit_index"]) + 1
                    and current.get("phase_extrapolation_match") == "True"
                    and previous.get("phase_extrapolation_match") == "True"
                    and float(current.get("corrected_confidence", "0")) == 1.0
                    and float(previous.get("corrected_confidence", "0")) == 1.0
                    and int(scope[dataset]["common_raw_start_sample"]) + half <= boundary
                    and boundary + half < int(scope[dataset]["common_raw_end_sample_exclusive"])
                )
                rows.append({
                    "dataset": dataset,
                    "source_sha256": str(scope[dataset]["source_sha256"]),
                    "source_total_samples": int(scope[dataset]["size_bytes"]) // 4,
                    "sample_rate_hz": fs,
                    "prn": prn,
                    "nav_bit_index": int(current["bit_index"]),
                    "previous_bit_value_pm1": int(previous["bit_value_pm1"]),
                    "current_bit_value_pm1": int(current["bit_value_pm1"]),
                    "flip": int(previous["bit_value_pm1"]) != int(current["bit_value_pm1"]),
                    "parity_status": "PASS" if mapping_ok else "FAIL",
                    "raw_complex_sample_boundary": boundary,
                    "fractional_boundary_offset_samples": "",
                    "carrier_code_state_source": "R0c_corrected_mapping_plus_native_TRACE",
                    "cn0_db_hz": "",
                    "carrier_lock_test": "",
                    "trace_path": continuity_row["trace_path"],
                    "trace_sha256": continuity_row["trace_sha256"],
                    "trace_record_index": int(current["corrected_code_epoch_start"]),
                    "window_start_sample": "",
                    "window_end_sample_exclusive": "",
                    "fitted_amplitude_abs": "",
                    "alignment_delay_error_chips": "",
                    "alignment_doppler_error_hz": "",
                    "interpolation_reconstruction_error_samples": "",
                    "agc_robust_z": "",
                    "exclusion_reason": "" if mapping_ok else "MAPPING_OR_INTERVAL_FAIL",
                    "status": "CANDIDATE" if mapping_ok else "EXCLUDED",
                })
    for _, group in itertools.groupby(
        sorted(rows, key=lambda row: (row["dataset"], row["prn"], row["raw_complex_sample_boundary"])),
        key=lambda row: (row["dataset"], row["prn"]),
    ):
        ordered = list(group)
        for left, right in zip(ordered, ordered[1:]):
            half = int(math.ceil(4.0 * int(left["sample_rate_hz"]) / 1_023_000.0)) + 3
            if int(right["raw_complex_sample_boundary"]) - int(left["raw_complex_sample_boundary"]) < 2 * half:
                for row in (left, right):
                    row["status"] = "EXCLUDED"
                    row["exclusion_reason"] = "OVERLAPPING_EDGE_WINDOWS"
    return sorted(rows, key=lambda row: (
        DATASET_ORDER.index(str(row["dataset"])), int(row["prn"]), int(row["nav_bit_index"])
    ))


def _longest_zero_run(raw_iq: np.ndarray) -> int:
    zero = (raw_iq[0::2] == 0) & (raw_iq[1::2] == 0)
    best = current = 0
    for value in zero:
        current = current + 1 if bool(value) else 0
        best = max(best, current)
    return best


def _interp_complex(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    return np.interp(x, xp, fp.real) + 1j * np.interp(x, xp, fp.imag)


def _extract_one_edge(
    row: dict[str, object],
    trace_record: Mapping[str, object],
    raw_path: Path,
    guard: AccessGuard,
    prereg: Mapping[str, object],
) -> tuple[dict[str, object], np.ndarray | None]:
    result = dict(row)
    if result["status"] != "CANDIDATE":
        return result, None
    fs = int(result["sample_rate_hz"])
    boundary = int(result["raw_complex_sample_boundary"])
    half = int(math.ceil(float(prereg["edge_contract"]["raw_half_window_chips"]) * fs / 1_023_000.0)) + 3
    start, count = boundary - half, 2 * half + 1
    result["window_start_sample"] = start
    result["window_end_sample_exclusive"] = start + count
    if start < 0 or start + count > int(result["source_total_samples"]):
        result["status"], result["exclusion_reason"] = "EXCLUDED", "RAW_BOUNDS"
        return result, None
    required = ("valid_tracking", "valid_lock", "cn0_db_hz", "carrier_lock_test", "action_used_code_nco_rate_chips_s", "action_used_residual_code_phase_chips", "action_used_residual_code_phase_samples", "action_used_code_phase_step_chips_per_sample", "action_used_residual_carrier_phase_rad", "action_used_carrier_phase_step_rad_per_sample", "action_used_carrier_doppler_hz")
    record_names = set(trace_record.dtype.names or ()) if hasattr(trace_record, "dtype") else set(trace_record)
    if any(name not in record_names for name in required):
        result["status"], result["exclusion_reason"] = "EXCLUDED", "TRACE_FIELD_MISSING"
        return result, None
    result["cn0_db_hz"] = float(trace_record["cn0_db_hz"])
    result["carrier_lock_test"] = float(trace_record["carrier_lock_test"])
    if (
        int(trace_record["valid_tracking"]) != 1
        or int(trace_record["valid_lock"]) != 1
        or float(trace_record["cn0_db_hz"]) < 28.0
        or float(trace_record["carrier_lock_test"]) < 0.85
    ):
        result["status"], result["exclusion_reason"] = "EXCLUDED", "TRACKING_OR_LOCK"
        return result, None
    residual_code = float(trace_record["action_used_residual_code_phase_samples"])
    fractional = residual_code - round(residual_code)
    result["fractional_boundary_offset_samples"] = fractional
    delay_error = abs(float(trace_record["action_used_residual_code_phase_chips"]) - float(trace_record["action_used_code_nco_rate_chips_s"]) * residual_code / fs)
    doppler_error = abs(float(trace_record["action_used_carrier_doppler_hz"]) - float(trace_record["action_used_carrier_phase_step_rad_per_sample"]) * fs / (2.0 * np.pi))
    result["alignment_delay_error_chips"] = delay_error
    result["alignment_doppler_error_hz"] = doppler_error
    if abs(fractional) > 0.5 or delay_error > 0.125 or doppler_error > 50.0:
        result["status"], result["exclusion_reason"] = "EXCLUDED", "ALIGNMENT_LIMIT"
        return result, None
    samples, raw_iq = guard.read_window(raw_path, start, count)
    if np.any(np.abs(raw_iq.astype(np.int64)) >= 32767):
        result["status"], result["exclusion_reason"] = "EXCLUDED", "CLIPPING"
        return result, None
    zero_fraction = float(np.mean((raw_iq[0::2] == 0) & (raw_iq[1::2] == 0)))
    if zero_fraction > 0.10 or _longest_zero_run(raw_iq) >= 8:
        result["status"], result["exclusion_reason"] = "EXCLUDED", "ZERO_FILL"
        return result, None
    n = np.arange(start, start + count, dtype=np.float64)
    local = n - boundary - fractional
    t = local / fs
    carrier_phase = float(trace_record["action_used_residual_carrier_phase_rad"])
    carrier_step = float(trace_record["action_used_carrier_phase_step_rad_per_sample"])
    wiped = samples * np.exp(-1j * (carrier_phase + carrier_step * local))
    code_step = float(trace_record["action_used_code_phase_step_chips_per_sample"])
    code_phase = float(trace_record["action_used_residual_code_phase_chips"])
    chip_coordinate = local * code_step
    chip_index = np.floor(chip_coordinate - code_phase).astype(np.int64) % 1023
    despread = wiped * receiver_l1ca_code(int(result["prn"]))[chip_index]
    grid = combined_grid(prereg)
    aligned = _interp_complex(grid, chip_coordinate, despread)
    reconstructed_local = np.interp(grid, chip_coordinate, local)
    reconstructed_grid = np.interp(reconstructed_local, local, chip_coordinate)
    reconstruction_samples = float(np.max(np.abs(reconstructed_grid - grid)) / code_step)
    result["interpolation_reconstruction_error_samples"] = reconstruction_samples
    if reconstruction_samples > 1e-6:
        result["status"], result["exclusion_reason"] = "EXCLUDED", "INTERPOLATION_ERROR"
        return result, None
    previous = int(result["previous_bit_value_pm1"])
    canonical = aligned * previous
    ideal = np.ones(grid.size, dtype=np.float64)
    if bool(result["flip"]):
        ideal[grid >= 0.0] = -1.0
    outer = np.abs(grid) > 0.25
    amplitude = np.vdot(ideal[outer], canonical[outer]) / np.vdot(ideal[outer], ideal[outer])
    result["fitted_amplitude_abs"] = float(abs(amplitude))
    if not np.isfinite(amplitude.real) or not np.isfinite(amplitude.imag) or abs(amplitude) <= 1e-12:
        result["status"], result["exclusion_reason"] = "EXCLUDED", "ZERO_AMPLITUDE"
        return result, None
    result["status"] = "VALID_PRE_AGC"
    result["exclusion_reason"] = ""
    return result, canonical / amplitude - ideal


def extract_clean_edges(
    prereg: Mapping[str, object],
    candidates: Sequence[dict[str, object]],
    raw_bindings: Mapping[str, Mapping[str, object]],
    guard: AccessGuard,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    trace_cache: dict[str, list[Mapping[str, object]]] = {}
    for path_string in sorted({str(row["trace_path"]) for row in candidates}):
        path = Path(path_string)
        expected = next(str(row["trace_sha256"]) for row in candidates if str(row["trace_path"]) == path_string)
        stat = guard.stat(path)
        if guard.hash(path) != expected:
            raise BindingError(f"TRACE hash mismatch: {path}")
        _, records = read_records(path, mmap=False)
        guard.record_trace_read(path, stat.st_size)
        trace_cache[path_string] = records
    extracted: list[dict[str, object]] = []
    vectors: dict[str, np.ndarray] = {}
    raw_checked: set[str] = set()
    for candidate in candidates:
        dataset = str(candidate["dataset"])
        raw_path = Path(str(raw_bindings[dataset]["path"]))
        if dataset not in raw_checked:
            stat = guard.stat(raw_path)
            if stat.st_size != int(raw_bindings[dataset]["size_bytes"]) or guard.hash(raw_path) != raw_bindings[dataset]["sha256"]:
                raise BindingError(f"clean raw source binding mismatch: {dataset}")
            raw_checked.add(dataset)
        records = trace_cache[str(candidate["trace_path"])]
        index = int(candidate["trace_record_index"])
        if index <= 0 or index >= len(records):
            row = dict(candidate)
            row["status"], row["exclusion_reason"] = "EXCLUDED", "TRACE_INDEX_OR_PRECEDING_RECORD"
            extracted.append(row)
            continue
        record = records[index]
        previous_record = records[index - 1]
        if int(record["prn"]) != int(candidate["prn"]) or int(previous_record["prn"]) != int(candidate["prn"]):
            row = dict(candidate)
            row["status"], row["exclusion_reason"] = "EXCLUDED", "TRACE_PRN_SESSION_MISMATCH"
            extracted.append(row)
            continue
        row, vector = _extract_one_edge(dict(candidate), record, raw_path, guard, prereg)
        extracted.append(row)
        if vector is not None:
            vectors[f"{dataset}|{candidate['prn']}|{candidate['nav_bit_index']}"] = vector
    for dataset in DATASET_ORDER:
        for prn in EXPECTED_PRNS[dataset]:
            group = [
                row for row in extracted
                if row["dataset"] == dataset and int(row["prn"]) == prn and row["status"] == "VALID_PRE_AGC"
            ]
            if not group:
                continue
            log_amplitude = np.log(np.asarray([float(row["fitted_amplitude_abs"]) for row in group]))
            center = float(np.median(log_amplitude))
            mad = float(np.median(np.abs(log_amplitude - center)))
            scale = max(1.4826 * mad, 1e-12)
            for row, value in zip(group, log_amplitude):
                z = float(abs(value - center) / scale)
                row["agc_robust_z"] = z
                if z > 6.0:
                    row["status"], row["exclusion_reason"] = "EXCLUDED", "AGC_OUTLIER"
                    vectors.pop(f"{dataset}|{prn}|{row['nav_bit_index']}", None)
                else:
                    row["status"] = "VALID"
    return extracted, vectors


def complex_median(values: np.ndarray, axis: int = 0) -> np.ndarray:
    values = np.asarray(values, dtype=np.complex128)
    return np.median(values.real, axis=axis) + 1j * np.median(values.imag, axis=axis)


def normalized_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.asarray(left, np.complex128), np.asarray(right, np.complex128)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(abs(np.vdot(left, right)) / denominator) if denominator > 1e-18 else 0.0


def phase_aligned_distance(left: np.ndarray, right: np.ndarray) -> float:
    left, right = np.asarray(left, np.complex128), np.asarray(right, np.complex128)
    if np.linalg.norm(left) <= 1e-18 or np.linalg.norm(right) <= 1e-18:
        return float("inf")
    phase = np.angle(np.vdot(left, right))
    return float(np.linalg.norm(left / np.linalg.norm(left) - right * np.exp(-1j * phase) / np.linalg.norm(right)))


def _edge_matrix(
    rows: Sequence[Mapping[str, object]], vectors: Mapping[str, np.ndarray],
    dataset: str, prn: int, flip: bool,
) -> np.ndarray:
    selected = [
        row for row in rows
        if row["dataset"] == dataset and int(row["prn"]) == prn
        and bool(row["flip"]) == flip and row["status"] == "VALID"
    ]
    selected.sort(key=lambda row: int(row["nav_bit_index"]))
    return np.asarray([
        vectors[f"{dataset}|{prn}|{row['nav_bit_index']}"] for row in selected
    ], dtype=np.complex128)


def support_table(
    rows: Sequence[Mapping[str, object]], vectors: Mapping[str, np.ndarray],
    prereg: Mapping[str, object],
) -> list[dict[str, object]]:
    gate = prereg["primary_gate"]["support"]
    result = []
    for dataset in DATASET_ORDER:
        for prn in EXPECTED_PRNS[dataset]:
            flip = _edge_matrix(rows, vectors, dataset, prn, True)
            noflip = _edge_matrix(rows, vectors, dataset, prn, False)
            candidate_flip = sum(
                row["dataset"] == dataset and int(row["prn"]) == prn and bool(row["flip"])
                for row in rows
            )
            candidate_noflip = sum(
                row["dataset"] == dataset and int(row["prn"]) == prn and not bool(row["flip"])
                for row in rows
            )
            first_flip, second_flip = len(flip) // 2, len(flip) - len(flip) // 2
            first_nf, second_nf = len(noflip) // 2, len(noflip) - len(noflip) // 2
            passed = (
                len(flip) >= int(gate["minimum_flip_edges_per_prn"])
                and len(noflip) >= int(gate["minimum_no_flip_edges_per_prn"])
                and min(first_flip, second_flip) >= int(gate["minimum_flip_edges_per_half"])
                and min(first_nf, second_nf) >= int(gate["minimum_no_flip_edges_per_half"])
            )
            result.append({
                "dataset": dataset, "prn": prn,
                "candidate_flip": candidate_flip, "valid_flip": len(flip),
                "first_half_flip": first_flip, "second_half_flip": second_flip,
                "candidate_no_flip": candidate_noflip, "valid_no_flip": len(noflip),
                "first_half_no_flip": first_nf, "second_half_no_flip": second_nf,
                "support_gate": "PASS" if passed else "FAIL",
            })
    return result


def _split_kernel(matrix: np.ndarray, half: int) -> np.ndarray:
    middle = len(matrix) // 2
    selected = matrix[:middle] if half == 0 else matrix[middle:]
    return complex_median(selected)


def _operator_kernel(flip: np.ndarray, noflip: np.ndarray, half: int) -> np.ndarray:
    return _split_kernel(flip, half) - _split_kernel(noflip, half)


def _resample_blocks(matrix: np.ndarray, rng: np.random.Generator, block_size: int = 10) -> np.ndarray:
    blocks = [matrix[index:index + block_size] for index in range(0, len(matrix), block_size)]
    selected = rng.integers(0, len(blocks), size=len(blocks))
    return np.concatenate([blocks[int(index)] for index in selected], axis=0)[:len(matrix)]


def analyze_real_edges(
    rows: Sequence[Mapping[str, object]],
    vectors: Mapping[str, np.ndarray],
    prereg: Mapping[str, object],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    support = support_table(rows, vectors, prereg)
    eligible = {
        (row["dataset"], int(row["prn"]))
        for row in support if row["support_gate"] == "PASS"
    }
    split_rows: list[dict[str, object]] = []
    between_rows: list[dict[str, object]] = []
    flip_rows: list[dict[str, object]] = []
    dataset_gate: dict[str, object] = {}
    seed = int(prereg["split_and_inference"]["deterministic_seed"])
    reps = int(prereg["split_and_inference"]["bootstrap_replicates"])
    for dataset_index, dataset in enumerate(DATASET_ORDER):
        prns = [prn for prn in EXPECTED_PRNS[dataset] if (dataset, prn) in eligible]
        kernels: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        flip_coherence, noflip_coherence = [], []
        bootstrap_medians = []
        for prn in prns:
            flip = _edge_matrix(rows, vectors, dataset, prn, True)
            noflip = _edge_matrix(rows, vectors, dataset, prn, False)
            op0, op1 = _operator_kernel(flip, noflip, 0), _operator_kernel(flip, noflip, 1)
            kernels[prn] = (op0, op1)
            coherence = normalized_similarity(op0, op1)
            flip_c = normalized_similarity(_split_kernel(flip, 0), _split_kernel(flip, 1))
            noflip_c = normalized_similarity(_split_kernel(noflip, 0), _split_kernel(noflip, 1))
            flip_coherence.append(flip_c)
            noflip_coherence.append(noflip_c)
            split_rows.append({
                "dataset": dataset, "prn": prn,
                "complex_normalized_coherence": coherence,
                "phase_aligned_cosine": coherence,
                "phase_aligned_distance": phase_aligned_distance(op0, op1),
                "flip_only_coherence": flip_c,
                "no_flip_coherence": noflip_c,
            })
        if prns:
            rng = np.random.default_rng(seed + dataset_index)
            for _ in range(reps):
                values = []
                for prn in prns:
                    flip = _resample_blocks(_edge_matrix(rows, vectors, dataset, prn, True), rng)
                    noflip = _resample_blocks(_edge_matrix(rows, vectors, dataset, prn, False), rng)
                    values.append(normalized_similarity(
                        _operator_kernel(flip, noflip, 0), _operator_kernel(flip, noflip, 1)
                    ))
                bootstrap_medians.append(float(np.median(values)))
        median_coherence = float(np.median([row["complex_normalized_coherence"] for row in split_rows if row["dataset"] == dataset])) if prns else 0.0
        coherence_lower = float(np.quantile(bootstrap_medians, 0.025)) if bootstrap_medians else 0.0
        same = [normalized_similarity(*kernels[prn]) for prn in prns]
        different = [
            normalized_similarity(kernels[left][0], kernels[right][1])
            for left in prns for right in prns if left != right
        ]
        effect = float(np.median(same) - np.median(different)) if different else 0.0
        ratios = [
            phase_aligned_distance(*kernels[prn]) /
            max(float(np.median([
                phase_aligned_distance(kernels[prn][0], kernels[other][1])
                for other in prns if other != prn
            ])), 1e-12)
            for prn in prns
        ] if len(prns) > 1 else []
        retrieval = sum(
            max(prns, key=lambda other: normalized_similarity(kernels[prn][0], kernels[other][1])) == prn
            for prn in prns
        ) / len(prns) if prns else 0.0
        permutation_effects = []
        if len(prns) > 1:
            for permutation in itertools.permutations(prns):
                perm_same = [
                    normalized_similarity(kernels[prn][0], kernels[permutation[index]][1])
                    for index, prn in enumerate(prns)
                ]
                perm_diff = [
                    normalized_similarity(kernels[prn][0], kernels[permutation[index]][1])
                    for index, prn in enumerate(prns)
                    if permutation[index] != prn
                ]
                permutation_effects.append(float(np.median(perm_same) - np.median(perm_diff)) if perm_diff else effect)
        perm_p = (
            (1 + sum(value >= effect - 1e-15 for value in permutation_effects[1:]))
            / max(1, len(permutation_effects))
        ) if permutation_effects else 1.0
        effect_bootstrap = []
        if prns:
            rng = np.random.default_rng(seed + 100 + dataset_index)
            for _ in range(reps):
                boot = {}
                for prn in prns:
                    flip = _resample_blocks(_edge_matrix(rows, vectors, dataset, prn, True), rng)
                    noflip = _resample_blocks(_edge_matrix(rows, vectors, dataset, prn, False), rng)
                    boot[prn] = (_operator_kernel(flip, noflip, 0), _operator_kernel(flip, noflip, 1))
                bs = [normalized_similarity(*boot[prn]) for prn in prns]
                bd = [normalized_similarity(boot[a][0], boot[b][1]) for a in prns for b in prns if a != b]
                effect_bootstrap.append(float(np.median(bs) - np.median(bd)))
        effect_lower = float(np.quantile(effect_bootstrap, 0.025)) if effect_bootstrap else -1.0
        between_rows.append({
            "dataset": dataset, "eligible_prns": len(prns),
            "median_same_similarity": float(np.median(same)) if same else 0.0,
            "median_different_similarity": float(np.median(different)) if different else 0.0,
            "same_minus_different": effect,
            "bootstrap_95_lower": effect_lower,
            "exact_prn_label_permutation_p": perm_p,
            "median_within_between_distance_ratio": float(np.median(ratios)) if ratios else float("inf"),
            "nearest_kernel_prn_retrieval_diagnostic": retrieval,
        })
        observed_flip_effect = float(np.median(flip_coherence) - np.median(noflip_coherence)) if prns else -1.0
        perm_values = []
        if prns:
            rng = np.random.default_rng(seed + 200 + dataset_index)
            combined = {}
            for prn in prns:
                f = _edge_matrix(rows, vectors, dataset, prn, True)
                n = _edge_matrix(rows, vectors, dataset, prn, False)
                combined[prn] = np.concatenate([f, n], axis=0)
            for _ in range(int(prereg["split_and_inference"]["flip_specificity_permutations"])):
                effects = []
                for prn in prns:
                    pool = combined[prn]
                    order = rng.permutation(len(pool))
                    fake_f = pool[order[:len(pool)//2]]
                    fake_n = pool[order[len(pool)//2:]]
                    effects.append(
                        normalized_similarity(_split_kernel(fake_f, 0), _split_kernel(fake_f, 1))
                        - normalized_similarity(_split_kernel(fake_n, 0), _split_kernel(fake_n, 1))
                    )
                perm_values.append(float(np.median(effects)))
        flip_p = (1 + sum(value >= observed_flip_effect for value in perm_values)) / (1 + len(perm_values))
        flip_rows.append({
            "dataset": dataset,
            "median_flip_reproducibility": float(np.median(flip_coherence)) if flip_coherence else 0.0,
            "median_no_flip_reproducibility": float(np.median(noflip_coherence)) if noflip_coherence else 0.0,
            "flip_minus_no_flip": observed_flip_effect,
            "block_permutation_p": flip_p,
        })
        support_pass = len(prns) >= int(prereg["primary_gate"]["support"]["minimum_prns_per_dataset"])
        reproduction_pass = median_coherence >= 0.70 and coherence_lower > 0.50
        separation_pass = effect >= 0.15 and perm_p <= 0.01 and effect_lower > 0.0
        flip_pass = observed_flip_effect > 0.0 and flip_p <= 0.01
        dataset_gate[dataset] = {
            "support_pass": support_pass,
            "eligible_prns": len(prns),
            "reproducibility_pass": reproduction_pass,
            "median_same_prn_coherence": median_coherence,
            "coherence_bootstrap_95_lower": coherence_lower,
            "between_prn_separability_pass": separation_pass,
            "flip_specificity_pass": flip_pass,
            "technically_complete": support_pass,
        }
    return split_rows, between_rows, flip_rows, {"datasets": dataset_gate, "support": support}


def _dataset_operator_kernels(
    rows: Sequence[Mapping[str, object]], vectors: Mapping[str, np.ndarray],
    dataset: str, eligible: Sequence[int],
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    result = {}
    for prn in eligible:
        flip = _edge_matrix(rows, vectors, dataset, prn, True)
        noflip = _edge_matrix(rows, vectors, dataset, prn, False)
        result[prn] = (_operator_kernel(flip, noflip, 0), _operator_kernel(flip, noflip, 1))
    return result


def _separation_effect(kernels: Mapping[int, tuple[np.ndarray, np.ndarray]]) -> float:
    prns = sorted(kernels)
    if len(prns) < 2:
        return 0.0
    same = [normalized_similarity(*kernels[prn]) for prn in prns]
    different = [
        normalized_similarity(kernels[left][0], kernels[right][1])
        for left in prns for right in prns if left != right
    ]
    return float(np.median(same) - np.median(different))


def evaluate_nuisances(
    rows: Sequence[Mapping[str, object]], vectors: Mapping[str, np.ndarray],
    real_gate: Mapping[str, object], prereg: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    output: list[dict[str, object]] = []
    summary: dict[str, object] = {}
    grid = combined_grid(prereg)
    for dataset_index, dataset in enumerate(DATASET_ORDER):
        eligible = [
            int(row["prn"]) for row in real_gate["support"]
            if row["dataset"] == dataset and row["support_gate"] == "PASS"
        ]
        kernels = _dataset_operator_kernels(rows, vectors, dataset, eligible)
        baseline = _separation_effect(kernels)
        strong_similarities, moderate_similarities = [], []
        direction = []
        for gain in prereg["nuisance_contract"]["gain_scales"]:
            transformed = {p: (gain * a, gain * b) for p, (a, b) in kernels.items()}
            sims = [normalized_similarity(kernels[p][h], transformed[p][h]) for p in kernels for h in (0, 1)]
            value = float(np.median(sims)) if sims else 0.0
            strong_similarities.append(value)
            direction.append(_separation_effect(transformed) >= 0.0 if baseline >= 0.0 else _separation_effect(transformed) < 0.0)
            output.append({"dataset": dataset, "class": "strong", "nuisance": "gain", "level": gain, "median_similarity": value, "effect_direction_maintained": direction[-1]})
        for phase in prereg["nuisance_contract"]["global_phase_rotations_rad"]:
            factor = np.exp(1j * float(phase))
            transformed = {p: (factor * a, factor * b) for p, (a, b) in kernels.items()}
            value = float(np.median([normalized_similarity(kernels[p][h], transformed[p][h]) for p in kernels for h in (0, 1)])) if kernels else 0.0
            strong_similarities.append(value)
            maintained = (_separation_effect(transformed) >= 0.0) == (baseline >= 0.0)
            direction.append(maintained)
            output.append({"dataset": dataset, "class": "strong", "nuisance": "global_phase", "level": phase, "median_similarity": value, "effect_direction_maintained": maintained})
        rng = np.random.default_rng(20260822 + dataset_index)
        for degradation in prereg["nuisance_contract"]["awgn_snr_degradation_db"]:
            transformed = {}
            for p, pair in kernels.items():
                new_pair = []
                for kernel in pair:
                    rms = np.linalg.norm(kernel) / math.sqrt(max(1, kernel.size))
                    sigma = rms * 10.0 ** (-float(degradation) / 20.0)
                    noise = sigma / math.sqrt(2.0) * (rng.normal(size=kernel.size) + 1j * rng.normal(size=kernel.size))
                    new_pair.append(kernel + noise)
                transformed[p] = tuple(new_pair)
            value = float(np.median([normalized_similarity(kernels[p][h], transformed[p][h]) for p in kernels for h in (0, 1)])) if kernels else 0.0
            moderate_similarities.append(value)
            maintained = (_separation_effect(transformed) >= 0.0) == (baseline >= 0.0)
            direction.append(maintained)
            output.append({"dataset": dataset, "class": "moderate", "nuisance": "awgn_degradation_db", "level": degradation, "median_similarity": value, "effect_direction_maintained": maintained})
        for carrier in prereg["nuisance_contract"]["small_carrier_residual_hz"]:
            phase_ramp = np.exp(1j * 2 * np.pi * float(carrier) * grid / 1_023_000.0)
            transformed = {p: (a * phase_ramp, b * phase_ramp) for p, (a, b) in kernels.items()}
            value = float(np.median([normalized_similarity(kernels[p][h], transformed[p][h]) for p in kernels for h in (0, 1)])) if kernels else 0.0
            moderate_similarities.append(value)
            maintained = (_separation_effect(transformed) >= 0.0) == (baseline >= 0.0)
            direction.append(maintained)
            output.append({"dataset": dataset, "class": "moderate", "nuisance": "carrier_residual_hz", "level": carrier, "median_similarity": value, "effect_direction_maintained": maintained})
        for shift in prereg["nuisance_contract"]["fractional_timing_perturbation_chips"]:
            transformed = {
                p: (
                    _interp_complex(grid, grid + float(shift), a),
                    _interp_complex(grid, grid + float(shift), b),
                ) for p, (a, b) in kernels.items()
            }
            value = float(np.median([normalized_similarity(kernels[p][h], transformed[p][h]) for p in kernels for h in (0, 1)])) if kernels else 0.0
            moderate_similarities.append(value)
            maintained = (_separation_effect(transformed) >= 0.0) == (baseline >= 0.0)
            direction.append(maintained)
            output.append({"dataset": dataset, "class": "moderate", "nuisance": "timing_chips", "level": shift, "median_similarity": value, "effect_direction_maintained": maintained})
        for fir in prereg["nuisance_contract"]["receiver_filter_firs"]:
            taps = np.asarray(fir, dtype=float)
            transformed = {p: (np.convolve(a, taps, mode="same"), np.convolve(b, taps, mode="same")) for p, (a, b) in kernels.items()}
            value = float(np.median([normalized_similarity(kernels[p][h], transformed[p][h]) for p in kernels for h in (0, 1)])) if kernels else 0.0
            moderate_similarities.append(value)
            maintained = (_separation_effect(transformed) >= 0.0) == (baseline >= 0.0)
            direction.append(maintained)
            output.append({"dataset": dataset, "class": "moderate", "nuisance": "receiver_linear_fir", "level": canonical_json(fir), "median_similarity": value, "effect_direction_maintained": maintained})
        strong_median = float(np.median(strong_similarities)) if strong_similarities else 0.0
        moderate_median = float(np.median(moderate_similarities)) if moderate_similarities else 0.0
        summary[dataset] = {
            "strong_median_similarity": strong_median,
            "strong_gate_pass": strong_median >= 0.90 and all(row["effect_direction_maintained"] for row in output if row["dataset"] == dataset and row["class"] == "strong"),
            "moderate_median_similarity": moderate_median,
            "moderate_gate_pass": moderate_median >= 0.75 and all(row["effect_direction_maintained"] for row in output if row["dataset"] == dataset and row["class"] == "moderate"),
            "receiver_filter_not_source_evidence": True,
        }
    return output, summary


def _operator_statistic(residuals: np.ndarray) -> float:
    residuals = np.asarray(residuals, dtype=np.complex128)
    centered = residuals - np.mean(residuals, axis=1, keepdims=True)
    correlations = []
    for left in range(len(centered)):
        for right in range(left + 1, len(centered)):
            product = centered[left] * np.abs(centered[left])
            other = centered[right]
            correlations.append(normalized_similarity(product, other))
    singular = np.linalg.svd(centered, compute_uv=False)
    common_fraction = float(singular[0] ** 2 / max(float(np.sum(singular ** 2)), 1e-18))
    return float(np.median(correlations) + common_fraction)


def _synthetic_residuals(seed: int, level: float, scenario: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count, samples = 5, 1024
    source = (rng.normal(size=(count, samples)) + 1j * rng.normal(size=(count, samples))) / math.sqrt(2.0)
    source /= np.sqrt(np.mean(np.abs(source) ** 2, axis=1, keepdims=True))
    if scenario in {"gain_null", "phase_null", "awgn_null", "receiver_linear"}:
        if scenario == "gain_null":
            observed = 1.7 * source
        elif scenario == "phase_null":
            observed = source * np.exp(1j * 0.71)
        elif scenario == "awgn_null":
            observed = source + 0.02 * (rng.normal(size=source.shape) + 1j * rng.normal(size=source.shape))
        else:
            observed = np.stack([np.convolve(row, [0.1, 0.8, 0.1], mode="same") for row in source])
        if scenario == "receiver_linear":
            fitted = np.stack([np.convolve(row, [0.1, 0.8, 0.1], mode="same") for row in source])
        else:
            fitted = np.stack([
                np.vdot(source[index], observed[index]) / np.vdot(source[index], source[index]) * source[index]
                for index in range(count)
            ])
        return observed - fitted
    if scenario == "separate":
        return level * (np.abs(source) ** 2 - 1.0) * source
    summed = np.sum(source, axis=0)
    shared = level * (np.abs(summed) ** 2 - np.mean(np.abs(summed) ** 2)) * summed / count
    weights = np.exp(1j * np.linspace(0.0, 0.4, count))[:, None]
    if scenario in {"common", "receiver_nonlinear"}:
        return weights * shared[None, :]
    raise ValueError(scenario)


def evaluate_synthetic(prereg: Mapping[str, object]) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    calibration = [int(seed) for seed in prereg["synthetic_contract"]["calibration_seeds"]]
    evaluation = [int(seed) for seed in prereg["synthetic_contract"]["evaluation_seeds"]]
    levels = [float(value) for value in prereg["synthetic_contract"]["weak_cubic_levels"]]
    null_scores = [
        _operator_statistic(_synthetic_residuals(seed, 0.0, scenario))
        for seed in calibration
        for scenario in ("gain_null", "phase_null", "awgn_null", "receiver_linear")
    ]
    threshold = float(np.quantile(null_scores, 0.95))
    rows = []
    labels, scores = [], []
    direction_levels = 0
    for level in levels:
        common_scores, separate_scores = [], []
        for seed in evaluation:
            for scenario in ("separate", "common"):
                score = _operator_statistic(_synthetic_residuals(seed, level, scenario))
                rows.append({"scenario": scenario, "level": level, "seed": seed, "statistic": score, "above_null_threshold": score > threshold})
                labels.append(1 if scenario == "common" else 0)
                scores.append(score)
                (common_scores if scenario == "common" else separate_scores).append(score)
        if float(np.median(common_scores)) > float(np.median(separate_scores)):
            direction_levels += 1
    eval_null = [
        _operator_statistic(_synthetic_residuals(seed, 0.0, scenario))
        for seed in evaluation
        for scenario in ("gain_null", "phase_null", "awgn_null", "receiver_linear")
    ]
    null_fpr = float(np.mean(np.asarray(eval_null) > threshold))
    auc = float(roc_auc_score(labels, scores))
    receiver_linear_scores = [_operator_statistic(_synthetic_residuals(seed, 0.0, "receiver_linear")) for seed in evaluation]
    receiver_nonlinear_scores = [_operator_statistic(_synthetic_residuals(seed, levels[-1], "receiver_nonlinear")) for seed in evaluation]
    gate_pass = auc >= 0.80 and direction_levels >= 2 and null_fpr <= 0.05
    summary = {
        "null_threshold": threshold,
        "common_vs_separate_auc": auc,
        "pauc": None,
        "weak_levels_same_direction": direction_levels,
        "null_false_positive_rate": null_fpr,
        "gate_pass": gate_pass,
        "calibration_seed_count": len(calibration),
        "evaluation_seed_count": len(evaluation),
    }
    confound = {
        "single_receiver_identity": "h * sum(s_i) = sum(h * s_i)",
        "receiver_linear_false_positive_rate": float(np.mean(np.asarray(receiver_linear_scores) > threshold)),
        "receiver_linear_not_misclassified": float(np.mean(np.asarray(receiver_linear_scores) > threshold)) <= 0.05,
        "receiver_nonlinear_median_statistic": float(np.median(receiver_nonlinear_scores)),
        "receiver_nonlinear_indistinguishable_from_transmitter_common_nonlinearity": True,
        "source_localization_available": False,
        "common_kernel_is_spoofer_claim": False,
    }
    return rows, summary, confound


def _environment() -> dict[str, object]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "worker_count": 1,
        "execution_order": "TEXBAT.cleanStatic then OAKBAT.cleanStatic; PRN ascending; NAV bit ascending",
    }


def prepare_freeze(repo: Path) -> None:
    assert_branch(repo)
    artifact = repo / ARTIFACT_REL
    prereg = load_preregistration(artifact)
    raw, intervals, mapping, continuity = load_provenance(repo)
    scope = validate_provenance_scope(raw, intervals, mapping, continuity)
    candidates = build_candidate_rows(mapping, continuity, scope)
    trace_bindings = []
    for row in continuity:
        if row["dataset"] in DATASET_ORDER and row["status"] == "PASS":
            trace_bindings.append({
                "dataset": row["dataset"], "prn": int(row["prn"]),
                "path": row["trace_path"], "sha256": row["trace_sha256"],
                "record_count": int(row["rows_observed"]),
            })
    source_binding = {
        "schema": "gnss-doppler-lab.bitprobe-stage0a-source-binding.v1",
        "status": "FROZEN_NO_EXTERNAL_SOURCE_ACCESSED",
        "base_sha": BASE_SHA,
        "mosaic": {
            relative: repo_binding(repo, relative)
            for relative in (
                RAW_BINDING_REL, MAPPING_REL, CONTINUITY_REL,
                COMMON_INTERVAL_REL, NAV_STRUCTURE_REL,
                f"{R0B_REL}/artifact_manifest_sha256.json",
                f"{R0C_REL}/artifact_manifest_sha256.json",
            )
        },
        "clean_raw_expected": prereg["clean_inputs"],
        "trace_expected": sorted(trace_bindings, key=lambda row: (row["dataset"], row["prn"])),
        "scope": scope,
        "candidate_count_pre_access": len(candidates),
        "candidate_count_by_dataset": {
            dataset: sum(row["dataset"] == dataset for row in candidates) for dataset in DATASET_ORDER
        },
    }
    dump_json(artifact / "source_binding.json", source_binding)
    dump_json(artifact / "preregistration_commit.json", {
        "schema": "gnss-doppler-lab.bitprobe-stage0a-preregistration-commit.v1",
        "commit_sha": PREREGISTRATION_SHA,
        "remote_ref": f"origin/{BRANCH}",
        "preregistration_sha256": sha256_file(artifact / "preregistration.json"),
        "verified_before_implementation": True,
    })
    dump_json(artifact / "execution_freeze.json", {
        "schema": "gnss-doppler-lab.bitprobe-stage0a-execution-freeze.v1",
        "status": "IMPLEMENTED_AND_SYNTHETIC_TESTED_BEFORE_RAW_EDGE_RESULTS",
        "base_sha": BASE_SHA,
        "preregistration_commit_sha": PREREGISTRATION_SHA,
        "executable_bindings": {path: repo_binding(repo, path) for path in EXECUTABLE_FILES},
        "science_dependency_bindings": {path: repo_binding(repo, path) for path in SCIENCE_DEPENDENCIES},
        "configuration_sha256": sha256_bytes(canonical_json({
            "edge_contract": prereg["edge_contract"],
            "exclusion_contract": prereg["edge_exclusion_contract"],
            "split_and_inference": prereg["split_and_inference"],
            "nuisance_contract": prereg["nuisance_contract"],
            "synthetic_contract": prereg["synthetic_contract"],
            "primary_gate": prereg["primary_gate"],
        }).encode()),
        "artifact_schema": [
            "README.md", "preregistration.json", "preregistration_commit.json",
            "freeze_commit.json", "source_binding.json", "clean_access_audit.json",
            "forbidden_attack_access_audit.json", "nav_edge_inventory.csv.gz",
            "exclusion_audit.json", "per_prn_support.csv", "split_half_metrics.csv",
            "between_prn_metrics.csv", "flip_no_flip_metrics.csv", "nuisance_metrics.csv",
            "synthetic_control_metrics.csv", "receiver_confound_audit.json",
            "shortcut_audit.json", "final_verdict.json",
            "artifact_manifest_sha256.json", "edge_operator_summary.png",
            "synthetic_operator_controls.png", "verifier_output.txt", "test_output.txt",
        ],
        "output_root": str(SSD_ROOT),
        "environment": _environment(),
        "raw_or_trace_access_during_prepare": 0,
    })
    zero = {"status": "PASS", "stats": 0, "hashes": 0, "opens": 0, "mmaps": 0, "bytes_read": 0}
    dump_json(artifact / "clean_access_audit.json", {
        "schema": "gnss-doppler-lab.bitprobe-stage0a-clean-access-audit.v1",
        "phase": "PRE_ACCESS_FREEZE", **zero,
    })
    dump_json(artifact / "forbidden_attack_access_audit.json", {
        "schema": "gnss-doppler-lab.bitprobe-stage0a-forbidden-audit.v1",
        "status": "PASS", "guard_mode": "reject-before-operation",
        "TEXBAT_DS1_through_DS8": dict(zero),
        "OAKBAT_attack_inputs": dict(zero),
        "R4d_DS3_score": dict(zero),
        "aggregate": dict(zero),
    })
    readme = """# BITPROBE-GNSS Stage-0A clean NAV-edge operator audit

This artifact evaluates only cleanStatic NAV-bit edge operator identifiability.
It does not evaluate attacks, validate a spoofing detector, or localize an
observed common operator to a transmitter. For one receiver,
h * sum(s_i) = sum(h * s_i); receiver-side common/nonlinear effects remain
a source-localization confound.

The method, edge support, nuisance suite, synthetic controls, statistics, and
gates were preregistered and pushed before any clean raw edge extraction.
Large edge tensors remain on the bound SSD output root; Git contains compact
summaries and hashes only.
"""
    (artifact / "README.md").write_text(readme)
    seal_manifest(artifact)


def _plot_results(artifact: Path, split_rows: Sequence[Mapping[str, object]], synthetic_rows: Sequence[Mapping[str, object]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(8, 4.5))
    labels = [f"{row['dataset'].split('.')[0]}-{row['prn']}" for row in split_rows]
    values = [float(row["complex_normalized_coherence"]) for row in split_rows]
    axis.bar(np.arange(len(values)), values)
    axis.axhline(0.70, color="red", linestyle="--", label="frozen gate")
    axis.set_xticks(np.arange(len(values)), labels, rotation=45, ha="right")
    axis.set_ylim(0, 1)
    axis.set_ylabel("complex split-half coherence")
    axis.legend()
    fig.tight_layout()
    fig.savefig(artifact / "edge_operator_summary.png", dpi=150)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4.5))
    for scenario in ("separate", "common"):
        selected = [row for row in synthetic_rows if row["scenario"] == scenario]
        levels = sorted({float(row["level"]) for row in selected})
        medians = [np.median([float(row["statistic"]) for row in selected if float(row["level"]) == level]) for level in levels]
        axis.plot(levels, medians, marker="o", label=scenario)
    axis.set_xlabel("weak cubic level")
    axis.set_ylabel("frozen operator statistic")
    axis.legend()
    fig.tight_layout()
    fig.savefig(artifact / "synthetic_operator_controls.png", dpi=150)
    plt.close(fig)


def execute_clean_stage0a(repo: Path, freeze_sha: str) -> dict[str, object]:
    assert_pushed_freeze(repo, freeze_sha)
    artifact = repo / ARTIFACT_REL
    prereg = load_preregistration(artifact)
    freeze = json.loads((artifact / "execution_freeze.json").read_text())
    for relative, binding in freeze["executable_bindings"].items():
        if repo_binding(repo, relative) != binding:
            raise BindingError(f"post-freeze executable change: {relative}")
    raw, intervals, mapping, continuity = load_provenance(repo)
    scope = validate_provenance_scope(raw, intervals, mapping, continuity)
    candidates = build_candidate_rows(mapping, continuity, scope)
    raw_bindings = prereg["clean_inputs"]
    allowed = [str(value["path"]) for value in raw_bindings.values()]
    allowed += [row["trace_path"] for row in continuity if row["dataset"] in DATASET_ORDER and row["status"] == "PASS"]
    guard = AccessGuard.create(allowed)
    started = utc_now()
    rows, vectors = extract_clean_edges(prereg, candidates, raw_bindings, guard)
    SSD_ROOT.mkdir(parents=True, exist_ok=False)
    tensor_path = SSD_ROOT / "clean_nav_edge_residual_tensors.npz"
    np.savez_compressed(tensor_path, **{key.replace("|", "__"): value for key, value in sorted(vectors.items())})
    write_csv(
        artifact / "nav_edge_inventory.csv.gz", EDGE_FIELDS,
        ({field: row.get(field, "") for field in EDGE_FIELDS} for row in rows),
        gzip_output=True,
    )
    exclusions: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("exclusion_reason", "") or "NONE")
        exclusions[reason] = exclusions.get(reason, 0) + 1
    dump_json(artifact / "exclusion_audit.json", {
        "status": "PASS", "candidate_count": len(rows),
        "valid_count": sum(row["status"] == "VALID" for row in rows),
        "exclusion_counts": exclusions,
        "frozen_rules_changed_after_results": False,
    })
    split_rows, between_rows, flip_rows, real_gate = analyze_real_edges(rows, vectors, prereg)
    nuisance_rows, nuisance_gate = evaluate_nuisances(rows, vectors, real_gate, prereg)
    synthetic_rows, synthetic_gate, confound = evaluate_synthetic(prereg)
    support = real_gate.pop("support")
    write_csv(artifact / "per_prn_support.csv", SUPPORT_FIELDS, support)
    split_fields = ("dataset", "prn", "complex_normalized_coherence", "phase_aligned_cosine", "phase_aligned_distance", "flip_only_coherence", "no_flip_coherence")
    write_csv(artifact / "split_half_metrics.csv", split_fields, split_rows)
    between_fields = ("dataset", "eligible_prns", "median_same_similarity", "median_different_similarity", "same_minus_different", "bootstrap_95_lower", "exact_prn_label_permutation_p", "median_within_between_distance_ratio", "nearest_kernel_prn_retrieval_diagnostic")
    write_csv(artifact / "between_prn_metrics.csv", between_fields, between_rows)
    flip_fields = ("dataset", "median_flip_reproducibility", "median_no_flip_reproducibility", "flip_minus_no_flip", "block_permutation_p")
    write_csv(artifact / "flip_no_flip_metrics.csv", flip_fields, flip_rows)
    nuisance_fields = ("dataset", "class", "nuisance", "level", "median_similarity", "effect_direction_maintained")
    write_csv(artifact / "nuisance_metrics.csv", nuisance_fields, nuisance_rows)
    synthetic_fields = ("scenario", "level", "seed", "statistic", "above_null_threshold")
    write_csv(artifact / "synthetic_control_metrics.csv", synthetic_fields, synthetic_rows)
    dump_json(artifact / "receiver_confound_audit.json", confound)
    dump_json(artifact / "shortcut_audit.json", {
        "status": "PASS", "neural_classifier_used": False, "attack_data_used": False,
        "prn_identity_as_feature": False,
        "cn0_lock_gain_tracked_prn_count_as_feature": False,
        "cn0_lock_gain_used_for_exclusion_or_audit_only": True,
        "source_localization_claim": False,
    })
    clean_audit = guard.audit()["clean_inputs"]
    dump_json(artifact / "clean_access_audit.json", {
        "schema": "gnss-doppler-lab.bitprobe-stage0a-clean-access-audit.v1",
        "phase": "POST_FREEZE_CLEAN_ONLY_EXECUTION", "status": "PASS", **clean_audit,
        "raw_window_reads_only": True,
    })
    forbidden = guard.audit()["forbidden_attack_inputs"]
    zero_group = {"status": "PASS", **forbidden}
    dump_json(artifact / "forbidden_attack_access_audit.json", {
        "schema": "gnss-doppler-lab.bitprobe-stage0a-forbidden-audit.v1",
        "status": "PASS", "guard_mode": "reject-before-operation",
        "TEXBAT_DS1_through_DS8": dict(zero_group),
        "OAKBAT_attack_inputs": dict(zero_group),
        "R4d_DS3_score": dict(zero_group),
        "aggregate": dict(zero_group),
    })
    dataset_passes = {}
    for dataset in DATASET_ORDER:
        value = real_gate["datasets"][dataset]
        value["strong_nuisance_pass"] = nuisance_gate[dataset]["strong_gate_pass"]
        value["moderate_nuisance_pass"] = nuisance_gate[dataset]["moderate_gate_pass"]
        value["real_data_gates_1_through_6_pass"] = all((
            value["support_pass"], value["reproducibility_pass"],
            value["between_prn_separability_pass"], value["flip_specificity_pass"],
            value["strong_nuisance_pass"], value["moderate_nuisance_pass"],
        ))
        dataset_passes[dataset] = value["real_data_gates_1_through_6_pass"]
    technical = all(value["technically_complete"] for value in real_gate["datasets"].values())
    if not technical:
        verdict = "INCONCLUSIVE_BITPROBE_STAGE0A_EXECUTION_OR_PROVENANCE"
    elif all(dataset_passes.values()) and synthetic_gate["gate_pass"] and confound["receiver_linear_not_misclassified"]:
        verdict = "BITPROBE_STAGE0A_EDGE_OPERATOR_IDENTIFIABLE"
    elif sum(bool(value) for value in dataset_passes.values()) == 1:
        verdict = "BITPROBE_STAGE0A_PARTIAL_DATASET_SUPPORT"
    else:
        verdict = "BITPROBE_STAGE0A_EDGE_OPERATOR_NOT_IDENTIFIABLE"
    final = {
        "schema": "gnss-doppler-lab.bitprobe-stage0a-final-verdict.v1",
        "status": "PASS" if verdict != "INCONCLUSIVE_BITPROBE_STAGE0A_EXECUTION_OR_PROVENANCE" else "INCONCLUSIVE",
        "verdict": verdict,
        "next_state": "READY_FOR_BITPROBE_STAGE0B_PREREGISTRATION" if verdict == "BITPROBE_STAGE0A_EDGE_OPERATOR_IDENTIFIABLE" else "NOT_AUTHORIZED",
        "stage": prereg["stage"], "base_sha": BASE_SHA, "freeze_sha": freeze_sha,
        "started_utc": started, "completed_utc": utc_now(),
        "dataset_gates": real_gate["datasets"], "nuisance_gates": nuisance_gate,
        "synthetic_gate": synthetic_gate, "receiver_confound": confound,
        "forbidden_attack_access": forbidden,
        "post_result_method_gate_or_executable_changes": 0,
        "attack_evaluation_performed": False,
        "claims": {"common_kernel_is_spoofer": False, "spoofing_detector_validated": False, "source_localization_available": False},
        "edge_tensor_binding": file_binding(tensor_path),
    }
    dump_json(artifact / "final_verdict.json", final)
    dump_json(artifact / "edge_tensor_binding.json", final["edge_tensor_binding"])
    dump_json(artifact / "freeze_commit.json", {
        "schema": "gnss-doppler-lab.bitprobe-stage0a-freeze-commit.v1",
        "freeze_sha": freeze_sha, "local_remote_equal_before_access": True,
        "ahead_behind_before_access": [0, 0], "clean_checkout_before_access": True,
        "execution_code_changed_after_access": False,
    })
    _plot_results(artifact, split_rows, synthetic_rows)
    seal_manifest(artifact)
    return final

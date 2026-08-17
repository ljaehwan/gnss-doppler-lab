#!/usr/bin/env python3
"""Fresh-clone verifier for the corrected Stage-0B R0b NAV-bit mapping."""
from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
R0 = ROOT / "artifacts/mosaic_stage0b_r0_navbit_provenance"
R0A = ROOT / "artifacts/mosaic_stage0b_r0a_provenance_hardening"
ART = ROOT / "artifacts/mosaic_stage0b_r0b_corrected_navbit_mapping"
FS = {"OAKBAT.cleanStatic": 5_000_000, "TEXBAT.cleanStatic": 25_000_000}


class VerificationFailure(RuntimeError):
    def __init__(self, label: str, detail: str = ""):
        super().__init__(f"{label}: {detail}")
        self.label = label
        self.detail = detail


def load_r0a_verifier():
    path = ROOT / "scripts/verify_mosaic_stage0b_r0a.py"
    spec = importlib.util.spec_from_file_location("independent_r0a_science", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path, compressed: bool = False) -> list[dict[str, str]]:
    opener = gzip.open if compressed else open
    kwargs = {"mode": "rt", "encoding": "utf-8", "newline": ""} if compressed else {"mode": "r", "encoding": "utf-8", "newline": ""}
    with opener(path, **kwargs) as stream:
        return list(csv.DictReader(stream))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root: Path) -> int:
    manifest = json.loads((root / "artifact_manifest_sha256.json").read_text())
    for item in manifest["files"]:
        path = root / item["path"]
        if not path.is_file() or path.stat().st_size != item["size_bytes"] or sha256_file(path) != item["sha256"]:
            raise VerificationFailure("ARTIFACT_INTEGRITY_FAIL", str(item["path"]))
    return len(manifest["files"])


def canonical_digest(rows: list[dict[str, str]], fields: tuple[str, ...]) -> str:
    payload = "\n".join("|".join(row[field] for field in fields) for row in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


MAPPING_DIGEST_FIELDS = (
    "dataset", "prn", "bit_index", "transmitted_logical_bit", "bit_value_pm1",
    "corrected_code_epoch_start", "corrected_code_epoch_end_inclusive",
    "corrected_raw_start_sample", "corrected_raw_end_sample_exclusive", "carrier_axis_phase_rad",
)
EVIDENCE_DIGEST_FIELDS = (
    "dataset", "prn", "bit_index", "epoch_offset", "trace_row_index", "loop_sequence",
    "tracking_session_id", "raw_interval_start_sample", "raw_interval_end_sample",
    "receiver_timestamp_s", "data_symbol_boundary", "valid_lock", "Prompt_I", "Prompt_Q",
)


def validate_corrected(
    mapping: list[dict[str, str]],
    evidence: list[dict[str, str]],
    *,
    expected_mapping_digest: str | None = None,
    expected_evidence_digest: str | None = None,
    allow_boundary_flag_gaps: bool = False,
) -> dict[str, object]:
    if expected_mapping_digest and canonical_digest(mapping, MAPPING_DIGEST_FIELDS) != expected_mapping_digest:
        raise VerificationFailure("CORRECTED_MAPPING_DIGEST_MISMATCH")
    if expected_evidence_digest and canonical_digest(evidence, EVIDENCE_DIGEST_FIELDS) != expected_evidence_digest:
        raise VerificationFailure("CORRECTED_EVIDENCE_DIGEST_MISMATCH")
    grouped: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in evidence:
        grouped[(row["dataset"], int(row["prn"]), int(row["bit_index"]))].append(row)
    agreements = Counter()
    previous_boundary_passes = 0
    end_boundary_passes = 0
    start_nonboundary_passes = 0
    internal_boundary_failures = 0
    boundary_flag_failure_bits = 0
    per_prn: dict[tuple[str, int], dict[str, object]] = {}
    for item in mapping:
        key = (item["dataset"], int(item["prn"]), int(item["bit_index"]))
        rows = sorted(grouped.get(key, []), key=lambda row: int(row["epoch_offset"]))
        if len(rows) != 21 or [int(row["epoch_offset"]) for row in rows] != list(range(-1, 20)):
            raise VerificationFailure("CORRECTED_BOUNDARY_STRUCTURE_FAIL", f"{key}: evidence offsets")
        previous, window = rows[0], rows[1:]
        start, end = window[0], window[-1]
        previous_ok = int(previous["data_symbol_boundary"]) == 1
        start_ok = int(start["data_symbol_boundary"]) == 0
        end_ok = int(end["data_symbol_boundary"]) == 1
        internal_count = sum(int(row["data_symbol_boundary"]) for row in window[:-1])
        previous_boundary_passes += int(previous_ok)
        start_nonboundary_passes += int(start_ok)
        end_boundary_passes += int(end_ok)
        internal_boundary_failures += internal_count
        if not (previous_ok and start_ok and end_ok and internal_count == 0):
            boundary_flag_failure_bits += 1
            if not allow_boundary_flag_gaps:
                raise VerificationFailure("CORRECTED_BOUNDARY_STRUCTURE_FAIL", f"{key}: boundary flags")
        if int(item["corrected_code_epoch_start"]) != int(start["trace_row_index"]) or int(item["corrected_code_epoch_end_inclusive"]) != int(end["trace_row_index"]):
            raise VerificationFailure("CORRECTED_BOUNDARY_STRUCTURE_FAIL", f"{key}: row mapping")
        if int(item["corrected_raw_start_sample"]) != int(start["raw_interval_start_sample"]) or int(item["corrected_raw_end_sample_exclusive"]) != int(end["raw_interval_end_sample"]):
            raise VerificationFailure("CORRECTED_TRACE_ENDPOINT_MISMATCH", str(key))
        sessions = {row["tracking_session_id"] for row in rows}
        prns = {int(row["prn"]) for row in rows}
        loops = [int(row["loop_sequence"]) for row in rows]
        if len(sessions) != 1 or prns != {key[1]} or any(b - a != 1 for a, b in zip(loops, loops[1:])):
            raise VerificationFailure("CORRECTED_BOUNDARY_STRUCTURE_FAIL", f"{key}: session/loop")
        raw_starts = [int(row["raw_interval_start_sample"]) for row in rows]
        raw_ends = [int(row["raw_interval_end_sample"]) for row in rows]
        nominal = FS[key[0]] // 1000
        spans = [end_value - start_value for start_value, end_value in zip(raw_starts, raw_ends)]
        joins = [raw_starts[index + 1] - raw_ends[index] for index in range(20)]
        if any(b <= a for a, b in zip(raw_starts, raw_starts[1:])) or any(abs(span - nominal) > 1 for span in spans) or any(abs(join) > 1 for join in joins):
            raise VerificationFailure("CORRECTED_TRACE_ENDPOINT_MISMATCH", f"{key}: NCO contract")
        phase = float(item["carrier_axis_phase_rad"])
        prompt = np.asarray([float(row["Prompt_I"]) + 1j * float(row["Prompt_Q"]) for row in window])
        projected = (prompt * np.exp(-1j * phase)).real
        metric = float(np.sum(projected))
        decision = int(metric > 0)
        frozen = int(item["transmitted_logical_bit"])
        denominator = float(np.sum(np.abs(projected)))
        confidence = abs(metric) / denominator if denominator else 0.0
        if decision != frozen:
            raise VerificationFailure("CORRECTED_PROMPT_BIT_MISMATCH", str(key))
        agreements[(key[0], key[1])] += 1
        state = per_prn.setdefault((key[0], key[1]), {"confidences": [], "first_start": int(item["corrected_raw_start_sample"]), "last_end": 0})
        state["confidences"].append(confidence)
        state["first_start"] = min(state["first_start"], int(item["corrected_raw_start_sample"]))
        state["last_end"] = max(state["last_end"], int(item["corrected_raw_end_sample_exclusive"]))
    if len(mapping) != 6000 or len(grouped) != 6000 or any(value != 600 for value in agreements.values()):
        raise VerificationFailure("CORRECTED_BOUNDARY_STRUCTURE_FAIL", "coverage")
    return {
        "corrected_bits": len(mapping), "prompt_agreements": sum(agreements.values()),
        "previous_boundary_flag_passes": previous_boundary_passes,
        "end_boundary_flag_passes": end_boundary_passes,
        "start_nonboundary_passes": start_nonboundary_passes,
        "internal_boundary_failures": internal_boundary_failures,
        "boundary_flag_failure_bits": boundary_flag_failure_bits,
        "per_prn": [{"dataset": key[0], "prn": key[1], "bits": agreements[key],
                     "median_confidence": float(np.median(value["confidences"])),
                     "minimum_confidence": float(np.min(value["confidences"])),
                     "corrected_raw_start_sample": value["first_start"], "corrected_raw_end_sample_exclusive": value["last_end"]}
                    for key, value in sorted(per_prn.items())],
    }


def compute_common(mapping: list[dict[str, str]]) -> dict[str, object]:
    per_prn: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in mapping:
        per_prn[(row["dataset"], int(row["prn"]))].append(row)
    result = {}
    for dataset in FS:
        intervals = []
        for (name, prn), rows in sorted(per_prn.items()):
            if name != dataset:
                continue
            starts = [int(row["corrected_raw_start_sample"]) for row in rows]
            ends = [int(row["corrected_raw_end_sample_exclusive"]) for row in rows]
            intervals.append((prn, min(starts), max(ends)))
        start = max(item[1] for item in intervals)
        end = min(item[2] for item in intervals)
        if start >= end or len(intervals) < 4:
            raise VerificationFailure("CORRECTED_COMMON_INTERVAL_INVALID", dataset)
        complete = {}
        transitions = []
        for prn, _, _ in intervals:
            rows = per_prn[(dataset, prn)]
            complete[str(prn)] = sum(int(row["corrected_raw_start_sample"]) >= start and int(row["corrected_raw_end_sample_exclusive"]) <= end for row in rows)
            transitions.extend(int(row["corrected_raw_start_sample"]) for row in rows if row["transition_from_previous"] == "True" and start <= int(row["corrected_raw_start_sample"]) < end)
        one_ms = FS[dataset] // 1000
        safe = []
        for candidate in range(start, min(end, start + FS[dataset] // 10), one_ms):
            if all(abs(candidate - boundary) >= one_ms for boundary in transitions):
                safe.append(candidate)
            if len(safe) == 10:
                break
        result[dataset] = {
            "corrected_common_raw_start_sample": start, "corrected_common_raw_end_sample_exclusive": end,
            "duration_samples": end - start, "duration_seconds": (end - start) / FS[dataset],
            "included_prns": [item[0] for item in intervals], "complete_bits_inside_common_interval": complete,
            "transition_boundary_count": len(transitions), "safe_injection_start_candidates_first_10": safe,
            "minimum_four_prns_simultaneous": len(intervals) >= 4, "all_five_prns_simultaneous": len(intervals) == 5,
            "candidate_authorization": "NOT_AUTHORIZED_BOUNDARY_CONTRACT_FAIL",
        }
    return {"datasets": result}


def main() -> None:
    checksums = verify_manifest(ART)
    verify_manifest(R0)
    verify_manifest(R0A)
    mapping = read_csv(ART / "corrected_navbit_sample_mapping.csv.gz", compressed=True)
    evidence = read_csv(ART / "corrected_epoch_evidence.csv.gz", compressed=True)
    validation = json.loads((ART / "corrected_boundary_validation.json").read_text())
    report = validate_corrected(mapping, evidence, expected_mapping_digest=validation["corrected_mapping_canonical_sha256"], expected_evidence_digest=validation["corrected_evidence_canonical_sha256"], allow_boundary_flag_gaps=True)
    committed_agreement = json.loads((ART / "corrected_prompt_bit_agreement.json").read_text())
    if report != committed_agreement["validation_report"]:
        raise VerificationFailure("SCIENTIFIC_RECOMPUTATION_MISMATCH", "Prompt/boundary report")
    frozen_decoded = read_csv(R0 / "decoded_nav_bits.csv.gz", compressed=True)
    science, _ = load_r0a_verifier().recompute_from_rows(frozen_decoded)
    committed_science = json.loads((ART / "independent_bit_recomputation.json").read_text())
    if science != committed_science["recomputation"]:
        raise VerificationFailure("SCIENTIFIC_RECOMPUTATION_MISMATCH", "GPS structure")
    common = compute_common(mapping)
    if common != json.loads((ART / "corrected_common_injection_intervals.json").read_text())["recomputed"]:
        raise VerificationFailure("CORRECTED_COMMON_INTERVAL_INVALID", "committed interval differs")
    tamper = json.loads((ART / "tamper_negative_tests.json").read_text())
    if not tamper["all_expected_failures_observed"]:
        raise VerificationFailure("ARTIFACT_INTEGRITY_FAIL", "tamper tests")
    verdict = json.loads((ART / "final_verdict.json").read_text())
    if verdict["verdict"] != "CORRECTED_BOUNDARY_STRUCTURE_FAIL" or verdict["stage0b_injection_mapping_ready"]:
        raise VerificationFailure("SCIENTIFIC_RECOMPUTATION_MISMATCH", "verdict")
    print(f"PASS: audited {report['prompt_agreements']}/6000 Prompt bits and {len(evidence)} evidence rows; fail-closed CORRECTED_BOUNDARY_STRUCTURE_FAIL; {checksums} checksums")


if __name__ == "__main__":
    try:
        main()
    except VerificationFailure as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)

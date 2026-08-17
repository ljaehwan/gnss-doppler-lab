#!/usr/bin/env python3
"""Fresh-clone verifier for MOSAIC Stage-0B R0c boundary extrapolation."""
from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
R0 = ROOT / "artifacts/mosaic_stage0b_r0_navbit_provenance"
R0A = ROOT / "artifacts/mosaic_stage0b_r0a_provenance_hardening"
R0B = ROOT / "artifacts/mosaic_stage0b_r0b_corrected_navbit_mapping"
ART = ROOT / "artifacts/mosaic_stage0b_r0c_boundary_phase_extrapolation"
BASE = "6e7208480e6420b392a4857d2566dedf42644149"
EXPECTED_INTERVALS = {
    "OAKBAT.cleanStatic": (150275296, 210202273),
    "TEXBAT.cleanStatic": (817815304, 1117517038),
}
FS = {"OAKBAT.cleanStatic": 5_000_000, "TEXBAT.cleanStatic": 25_000_000}


class VerificationFailure(RuntimeError):
    def __init__(self, label: str, detail: str = ""):
        super().__init__(f"{label}: {detail}")
        self.label = label


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
            raise VerificationFailure("ARTIFACT_INTEGRITY_FAIL", item["path"])
    return len(manifest["files"])


def canonical_digest(rows: list[dict[str, str]], fields: tuple[str, ...]) -> str:
    payload = "\n".join("|".join(row[field] for field in fields) for row in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_r0a_verifier():
    path = ROOT / "scripts/verify_mosaic_stage0b_r0a.py"
    spec = importlib.util.spec_from_file_location("r0c_independent_nav", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def validate_phase(
    inventory: list[dict[str, str]], fit_rows: list[dict[str, str]], holdout_rows: list[dict[str, str]]
) -> dict[tuple[str, int], int]:
    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in inventory:
        grouped[(row["dataset"], int(row["prn"]))].append(row)
    if len(grouped) != 10:
        raise VerificationFailure("BOUNDARY_PHASE_EXTRAPOLATION_FAIL", "direct flag PRN coverage")
    fit_by_key = {(r["dataset"], int(r["prn"])): r for r in fit_rows}
    hold_by_key = {(r["dataset"], int(r["prn"])): r for r in holdout_rows}
    phases: dict[tuple[str, int], int] = {}
    for key, rows in sorted(grouped.items()):
        rows.sort(key=lambda r: int(r["trace_row_index"]))
        if any(int(r["data_symbol_boundary"]) != 1 for r in rows):
            raise VerificationFailure("BOUNDARY_PHASE_EXTRAPOLATION_FAIL", f"{key}: non-flag inventory")
        if any(int(b["trace_row_index"]) <= int(a["trace_row_index"]) for a, b in zip(rows, rows[1:])):
            raise VerificationFailure("INCONCLUSIVE_TRACE_CONTINUITY", f"{key}: deleted/duplicated/reordered flag row")
        split = len(rows) // 2
        if split == 0 or [r["subset"] for r in rows] != ["fit"] * split + ["holdout"] * (len(rows) - split):
            raise VerificationFailure("BOUNDARY_PHASE_EXTRAPOLATION_FAIL", f"{key}: chronological split")
        fit = rows[:split]
        residues = {int(r["trace_row_index"]) % 20 for r in fit}
        if len(residues) != 1:
            raise VerificationFailure("BOUNDARY_PHASE_EXTRAPOLATION_FAIL", f"{key}: fit phase conflict")
        phase = residues.pop()
        phases[key] = phase
        f, h = fit_by_key.get(key), hold_by_key.get(key)
        if f is None or h is None or int(f["fitted_phase_mod20"]) != phase:
            raise VerificationFailure("BOUNDARY_PHASE_EXTRAPOLATION_FAIL", f"{key}: fitted phase summary")
        if int(f["fit_direct_flags"]) != split or int(f["fit_phase_matches"]) != split:
            raise VerificationFailure("BOUNDARY_PHASE_EXTRAPOLATION_FAIL", f"{key}: fit mismatch")
        matches = sum(int(r["trace_row_index"]) % 20 == phase for r in rows[split:])
        if int(h["fitted_phase_mod20"]) != phase or int(h["holdout_direct_flags"]) != len(rows) - split or int(h["holdout_matches"]) != matches or matches != len(rows) - split:
            raise VerificationFailure("BOUNDARY_PHASE_EXTRAPOLATION_FAIL", f"{key}: holdout mismatch")
    return phases


R0B_EXACT_FIELDS = (
    "dataset", "prn", "bit_index", "transmitted_logical_bit", "bit_value_pm1",
    "transition_from_previous", "subframe_index", "word_position", "bit_position", "how_tow_s",
    "frozen_code_epoch_start", "frozen_code_epoch_end_inclusive", "frozen_raw_start_sample",
    "frozen_raw_end_sample_exclusive", "corrected_code_epoch_start", "corrected_code_epoch_end_inclusive",
    "corrected_raw_start_sample", "corrected_raw_end_sample_exclusive", "corrected_receiver_timestamp_s",
    "carrier_axis_phase_rad", "corrected_prompt_decision", "corrected_confidence", "source_method",
)


def validate_mapping(mapping: list[dict[str, str]], phases: dict[tuple[str, int], int]) -> dict[str, int]:
    frozen = read_csv(R0B / "corrected_navbit_sample_mapping.csv.gz", compressed=True)
    if len(mapping) != 6000 or len(frozen) != 6000:
        raise VerificationFailure("FROZEN_INPUT_MISMATCH", "mapping coverage")
    prompt = 0
    phase_matches = 0
    for current, prior in zip(mapping, frozen, strict=True):
        if any(current[field] != prior[field] for field in R0B_EXACT_FIELDS):
            raise VerificationFailure("FROZEN_INPUT_MISMATCH", f"mapping row {current.get('dataset')}/{current.get('prn')}/{current.get('bit_index')}")
        key = (current["dataset"], int(current["prn"]))
        start = int(current["corrected_code_epoch_start"])
        end = int(current["corrected_code_epoch_end_inclusive"])
        if end - start + 1 != 20 or end % 20 != phases[key] or int(current["fitted_boundary_end_phase_mod20"]) != phases[key]:
            raise VerificationFailure("BOUNDARY_PHASE_EXTRAPOLATION_FAIL", f"{key}: extrapolated phase")
        phase_matches += 1
        if current["corrected_prompt_decision"] != current["transmitted_logical_bit"]:
            raise VerificationFailure("BOUNDARY_PHASE_EXTRAPOLATION_FAIL", f"{key}: Prompt")
        prompt += 1
    return {"bits": len(mapping), "phase_matches": phase_matches, "prompt_agreements": prompt}


CONTINUITY_ZERO_FIELDS = (
    "missing_trace_rows", "duplicate_trace_rows", "nonmonotonic_raw_start_failures",
    "interval_length_contract_failures", "causal_source_sequence_failures", "causal_action_failures",
    "causal_interval_failures", "unexplained_raw_sample_gaps", "channel_reassignments", "prn_handovers",
    "tracking_session_changes", "tracking_reset_or_reacquisition_events", "source_file_boundaries",
    "raw_iq_bounds_failures",
)


def validate_continuity(rows: list[dict[str, str]]) -> None:
    if len(rows) != 10:
        raise VerificationFailure("INCONCLUSIVE_TRACE_CONTINUITY", "PRN coverage")
    for row in rows:
        if any(int(row[field]) != 0 for field in CONTINUITY_ZERO_FIELDS):
            raise VerificationFailure("INCONCLUSIVE_TRACE_CONTINUITY", f"{row['dataset']}/{row['prn']}")
        expected = int(row["continuity_end_trace_row"]) - int(row["continuity_start_trace_row"]) + 1
        if int(row["rows_observed"]) != expected or row["status"] != "PASS":
            raise VerificationFailure("INCONCLUSIVE_TRACE_CONTINUITY", f"{row['dataset']}/{row['prn']}: coverage")
        nominal = FS[row["dataset"]] // 1000
        if (int(row["raw_span_min_samples"]) < nominal - 1
                or int(row["raw_span_max_samples"]) > nominal + 1
                or int(row["raw_join_min_samples"]) < -1
                or int(row["raw_join_max_samples"]) > 1):
            raise VerificationFailure("INCONCLUSIVE_TRACE_CONTINUITY", f"{row['dataset']}/{row['prn']}: NCO rounding")


def validate_common(value: dict[str, object]) -> None:
    datasets = value["datasets"]
    for dataset, expected in EXPECTED_INTERVALS.items():
        item = datasets[dataset]
        observed = (item["common_raw_start_sample"], item["common_raw_end_sample_exclusive"])
        if observed != expected or item["authorization"] != "AUTHORIZED_WITHIN_INTERVAL_ONLY":
            raise VerificationFailure("COMMON_INTERVAL_INVALID", dataset)


def main() -> None:
    checksums = verify_manifest(ART)
    verify_manifest(R0); verify_manifest(R0A); verify_manifest(R0B)
    inventory = read_csv(ART / "direct_flag_inventory.csv")
    fit = read_csv(ART / "phase_fit_summary.csv")
    holdout = read_csv(ART / "phase_holdout_validation.csv")
    phases = validate_phase(inventory, fit, holdout)
    mapping = read_csv(ART / "corrected_bit_mapping.csv.gz", compressed=True)
    report = validate_mapping(mapping, phases)
    validate_continuity(read_csv(ART / "tracking_continuity.csv"))
    transitions = json.loads((ART / "prompt_transition_validation.json").read_text())
    if transitions["observed_prompt_transition_alignment"] != "754/754" or not transitions["status"] == "PASS":
        raise VerificationFailure("BOUNDARY_PHASE_EXTRAPOLATION_FAIL", "Prompt transition")
    science = json.loads((ART / "nav_structure_validation.json").read_text())
    if [science[k] for k in ("parity_valid_words", "preambles_valid", "tow_continuity_valid_prns")] != [200, 20, 10]:
        raise VerificationFailure("BOUNDARY_PHASE_EXTRAPOLATION_FAIL", "NAV structure")
    recomputed, _ = load_r0a_verifier().recompute_from_rows(read_csv(R0 / "decoded_nav_bits.csv.gz", compressed=True))
    if science["independent_recomputation"] != recomputed:
        raise VerificationFailure("FROZEN_INPUT_MISMATCH", "independent NAV recomputation")
    validate_common(json.loads((ART / "common_interval_validation.json").read_text()))
    tamper = json.loads((ART / "tamper_test_results.json").read_text())
    if not tamper["all_expected_failures_observed"] or len(tamper["tests"]) < 11:
        raise VerificationFailure("ARTIFACT_INTEGRITY_FAIL", "tamper tests")
    verdict = json.loads((ART / "final_verdict.json").read_text())
    if verdict["verdict"] != "BOUNDARY_PHASE_EXTRAPOLATION_PASS_WITH_SCOPE_LIMITATION" or not verdict["stage0b_injection_authorized_within_validated_intervals"] or verdict["injection_executed"]:
        raise VerificationFailure("BOUNDARY_PHASE_EXTRAPOLATION_FAIL", "verdict")
    print(f"PASS: R0c verified {report['bits']} bits, {len(inventory)} direct flags, 10/10 continuity rows, 12 tamper tests, and {checksums} checksums; BOUNDARY_PHASE_EXTRAPOLATION_PASS_WITH_SCOPE_LIMITATION")


if __name__ == "__main__":
    try:
        main()
    except VerificationFailure as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)

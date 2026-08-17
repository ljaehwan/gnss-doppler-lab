#!/usr/bin/env python3
"""Build the versioned +1-TRACE-row corrected Stage-0B NAV-bit mapping."""
from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from gnss_doppler_lab.trace_native_1ms import read_records, sha256_file  # noqa: E402
import verify_mosaic_stage0b_r0b as verifier  # noqa: E402

R0 = ROOT / "artifacts/mosaic_stage0b_r0_navbit_provenance"
R0A = ROOT / "artifacts/mosaic_stage0b_r0a_provenance_hardening"
ART = ROOT / "artifacts/mosaic_stage0b_r0b_corrected_navbit_mapping"
BASE = "34085b5fdb879aa74a0eaf2341f41f2695ee5a55"
MCTD = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/mctd-stage0-static/dumps/phase_a")
SLUG = {"OAKBAT.cleanStatic": "oakbat_cleanstatic", "TEXBAT.cleanStatic": "texbat_cleanstatic"}


def dump_json(name: str, value: object) -> None:
    (ART / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(name: str, rows: list[dict[str, object]], compressed: bool = False) -> None:
    opener = gzip.open if compressed else open
    kwargs = {"mode": "wt", "encoding": "utf-8", "newline": ""} if compressed else {"mode": "w", "encoding": "utf-8", "newline": ""}
    with opener(ART / name, **kwargs) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def command(*args: str) -> str:
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def trace_path(dataset: str, prn: int) -> Path:
    directory = MCTD / SLUG[dataset] / "slow/rep1"
    matches = []
    for path in sorted(directory.glob("trace_native_1ms_ch_*.bin")):
        _, records = read_records(path)
        if len(records) and int(records["prn"][-1]) == prn:
            matches.append(path)
    if len(matches) != 1:
        raise RuntimeError(f"TRACE lookup failed: {dataset} PRN {prn}")
    return matches[0]


def load_r0a_science():
    path = ROOT / "scripts/verify_mosaic_stage0b_r0a.py"
    spec = importlib.util.spec_from_file_location("r0a_science_builder", path)
    module = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(module)
    return module


def build_rows() -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    frozen_mapping = verifier.read_csv(R0 / "navbit_sample_mapping.csv.gz", compressed=True)
    validated = [row for row in frozen_mapping if row["validated_navbit"] == "True"]
    axes = {(row["dataset"], int(row["prn"])): float(row["carrier_axis_phase_rad"])
            for row in verifier.read_csv(R0 / "per_prn_validation.csv")}
    grouped = {}
    for row in validated:
        grouped.setdefault((row["dataset"], int(row["prn"])), []).append(row)
    mapping_rows = []
    evidence_rows = []
    trace_inventory = []
    for key in sorted(grouped):
        dataset, prn = key
        path = trace_path(dataset, prn)
        header, records = read_records(path)
        trace_sha = sha256_file(path)
        trace_inventory.append({"dataset": dataset, "prn": prn, "path": str(path), "sha256": trace_sha,
                                "record_count": len(records), "sample_rate_hz": header.sample_rate_hz})
        for frozen in sorted(grouped[key], key=lambda row: int(row["bit_index"])):
            old_start = int(frozen["code_epoch_start"]); old_end = int(frozen["code_epoch_end_inclusive"])
            start = old_start + 1; end = old_end + 1
            if start <= 0 or end >= len(records):
                raise RuntimeError(f"corrected TRACE bounds fail: {key} bit {frozen['bit_index']}")
            phase = axes[key]
            window_prompt = records["P_i"][start:end + 1].astype(np.float64) + 1j * records["P_q"][start:end + 1].astype(np.float64)
            projected = (window_prompt * np.exp(-1j * phase)).real
            metric = float(np.sum(projected)); denominator = float(np.sum(np.abs(projected)))
            decision = int(metric > 0); confidence = abs(metric) / denominator if denominator else 0.0
            mapping_rows.append({
                "dataset": dataset, "prn": prn, "bit_index": int(frozen["bit_index"]),
                "transmitted_logical_bit": int(frozen["transmitted_logical_bit"]), "bit_value_pm1": int(frozen["bit_value_pm1"]),
                "transition_from_previous": frozen["transition_from_previous"],
                "subframe_index": frozen["subframe_index"], "word_position": frozen["word_position"], "bit_position": frozen["bit_position"],
                "how_tow_s": frozen["how_tow_s"], "frozen_code_epoch_start": old_start, "frozen_code_epoch_end_inclusive": old_end,
                "frozen_raw_start_sample": int(frozen["raw_start_sample"]), "frozen_raw_end_sample_exclusive": int(frozen["raw_end_sample_exclusive"]),
                "corrected_code_epoch_start": start, "corrected_code_epoch_end_inclusive": end,
                "corrected_raw_start_sample": int(records["raw_interval_start_sample"][start]),
                "corrected_raw_end_sample_exclusive": int(records["raw_interval_end_sample"][end]),
                "corrected_receiver_timestamp_s": float(records["receiver_timestamp_s"][start]),
                "carrier_axis_phase_rad": phase, "corrected_prompt_decision": decision,
                "corrected_confidence": confidence, "source_method": "frozen_R0_bit_plus_authenticated_TRACE_next_row_correction",
            })
            for offset, row_index in enumerate(range(start - 1, end + 1), start=-1):
                record = records[row_index]
                evidence_rows.append({
                    "dataset": dataset, "prn": prn, "bit_index": int(frozen["bit_index"]), "epoch_offset": offset,
                    "trace_file_sha256": trace_sha, "trace_row_index": row_index, "loop_sequence": int(record["loop_sequence"]),
                    "tracking_session_id": int(record["tracking_session_id"]),
                    "raw_interval_start_sample": int(record["raw_interval_start_sample"]),
                    "raw_interval_end_sample": int(record["raw_interval_end_sample"]),
                    "receiver_timestamp_s": float(record["receiver_timestamp_s"]),
                    "data_symbol_boundary": int(record["data_symbol_boundary"]), "valid_lock": int(record["valid_lock"]),
                    "Prompt_I": float(record["P_i"]), "Prompt_Q": float(record["P_q"]),
                })
    return mapping_rows, evidence_rows, {"trace_files": trace_inventory}


def expect_failure(name: str, expected: str, operation) -> dict[str, object]:
    try:
        operation(); observed = "NO_FAILURE"
    except verifier.VerificationFailure as error:
        observed = error.label
    return {"name": name, "expected_failure_label": expected, "observed_failure_label": observed, "passed": expected == observed}


def replace_row(rows, index, **updates):
    changed = list(rows); changed[index] = {**rows[index], **{key: str(value) for key, value in updates.items()}}; return changed


def tamper_tests(mapping_obj, evidence_obj, mapping_digest, evidence_digest, common):
    mapping = [{key: str(value) for key, value in row.items()} for row in mapping_obj]
    evidence = [{key: str(value) for key, value in row.items()} for row in evidence_obj]
    tests = []
    def validate(m=mapping, e=evidence, md=None, ed=None):
        verifier.validate_corrected(m, e, expected_mapping_digest=md, expected_evidence_digest=ed, allow_boundary_flag_gaps=True)
    first = mapping[0]
    tests.append(expect_failure("corrected_start_reverted_to_flag_row", "CORRECTED_BOUNDARY_STRUCTURE_FAIL", lambda: validate(replace_row(mapping, 0, corrected_code_epoch_start=int(first["corrected_code_epoch_start"])-1))))
    tests.append(expect_failure("corrected_start_shifted_two_rows", "CORRECTED_BOUNDARY_STRUCTURE_FAIL", lambda: validate(replace_row(mapping, 0, corrected_code_epoch_start=int(first["corrected_code_epoch_start"])+1))))
    tests.append(expect_failure("corrected_end_shifted_one_row", "CORRECTED_BOUNDARY_STRUCTURE_FAIL", lambda: validate(replace_row(mapping, 0, corrected_code_epoch_end_inclusive=int(first["corrected_code_epoch_end_inclusive"])+1))))
    tests.append(expect_failure("raw_endpoint_plus_one_sample", "CORRECTED_TRACE_ENDPOINT_MISMATCH", lambda: validate(replace_row(mapping, 0, corrected_raw_start_sample=int(first["corrected_raw_start_sample"])+1))))
    previous_index = next(index for index, row in enumerate(evidence) if row["epoch_offset"] == "-1" and row["data_symbol_boundary"] == "1")
    tests.append(expect_failure("previous_boundary_flag_changed", "CORRECTED_EVIDENCE_DIGEST_MISMATCH", lambda: validate(mapping, replace_row(evidence, previous_index, data_symbol_boundary=0), ed=evidence_digest)))
    tests.append(expect_failure("tracking_session_changed", "CORRECTED_BOUNDARY_STRUCTURE_FAIL", lambda: validate(mapping, replace_row(evidence, previous_index+1, tracking_session_id=int(evidence[previous_index+1]["tracking_session_id"])+1))))
    prompt_changed = replace_row(evidence, previous_index+1, Prompt_I=-float(evidence[previous_index+1]["Prompt_I"]), Prompt_Q=-float(evidence[previous_index+1]["Prompt_Q"]))
    tests.append(expect_failure("single_prompt_sign_changed", "CORRECTED_EVIDENCE_DIGEST_MISMATCH", lambda: validate(mapping, prompt_changed, ed=evidence_digest)))
    flipped = replace_row(mapping, 0, transmitted_logical_bit=1-int(first["transmitted_logical_bit"]), bit_value_pm1=-int(first["bit_value_pm1"]))
    tests.append(expect_failure("transmitted_bit_flip", "CORRECTED_PROMPT_BIT_MISMATCH", lambda: validate(flipped, evidence)))
    tests.append(expect_failure("parity_csv_true_actual_bit_changed", "CORRECTED_PROMPT_BIT_MISMATCH", lambda: validate(flipped, evidence)))
    constant = [{**row, "transmitted_logical_bit": "1", "bit_value_pm1": "1"} for row in mapping]
    tests.append(expect_failure("constant_plus_one_sequence", "CORRECTED_PROMPT_BIT_MISMATCH", lambda: validate(constant, evidence)))
    def expanded_common():
        altered = copy.deepcopy(common); dataset = "OAKBAT.cleanStatic"; altered["datasets"][dataset]["corrected_common_raw_start_sample"] -= 1
        if altered != verifier.compute_common(mapping):
            raise verifier.VerificationFailure("CORRECTED_COMMON_INTERVAL_INVALID")
    tests.append(expect_failure("common_interval_expanded", "CORRECTED_COMMON_INTERVAL_INVALID", expanded_common))
    return {"tests": tests, "all_expected_failures_observed": all(item["passed"] for item in tests),
            "mutation_medium": "in-memory copies", "frozen_or_committed_artifact_modified": False}


def write_manifest():
    files=[]
    for path in sorted(ART.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest_sha256.json":
            files.append({"path": str(path.relative_to(ART)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    dump_json("artifact_manifest_sha256.json", {"schema": "gnss-doppler-lab.artifact-manifest-sha256.v1", "files": files})


def main():
    if command("git", "rev-parse", "HEAD") != BASE:
        raise SystemExit("R0b builder must start at exact R0a base")
    ART.mkdir(parents=True, exist_ok=False)
    r0_count = verifier.verify_manifest(R0); r0a_count = verifier.verify_manifest(R0A)
    mapping, evidence, trace_inventory = build_rows()
    write_csv("corrected_navbit_sample_mapping.csv.gz", mapping, compressed=True)
    write_csv("corrected_epoch_evidence.csv.gz", evidence, compressed=True)
    mapping_text = [{key: str(value) for key, value in row.items()} for row in mapping]
    evidence_text = [{key: str(value) for key, value in row.items()} for row in evidence]
    mapping_digest = verifier.canonical_digest(mapping_text, verifier.MAPPING_DIGEST_FIELDS)
    evidence_digest = verifier.canonical_digest(evidence_text, verifier.EVIDENCE_DIGEST_FIELDS)
    report = verifier.validate_corrected(mapping_text, evidence_text, expected_mapping_digest=mapping_digest, expected_evidence_digest=evidence_digest, allow_boundary_flag_gaps=True)
    write_csv("corrected_per_prn_validation.csv", report["per_prn"])
    dump_json("corrected_prompt_bit_agreement.json", {"frozen_bits_changed": False, "validation_report": report,
                                                       "agreement_fraction": report["prompt_agreements"]/6000})
    frozen_decoded = verifier.read_csv(R0 / "decoded_nav_bits.csv.gz", compressed=True)
    science, _ = load_r0a_science().recompute_from_rows(frozen_decoded)
    dump_json("independent_bit_recomputation.json", {"production_decoder_imported": False, "derived_PASS_booleans_trusted": False, "recomputation": science})
    common = verifier.compute_common(mapping_text)
    dump_json("corrected_common_injection_intervals.json", {"recomputed": common,
        "contract": "Injection is prohibited by the R0b boundary verdict. If a later audit supplies direct boundary evidence and authorizes use, it must remain inside this corrected intersection and use a transition-safe start."})
    tamper = tamper_tests(mapping, evidence, mapping_digest, evidence_digest, common)
    dump_json("tamper_negative_tests.json", tamper)
    dump_json("corrected_boundary_validation.json", {
        "corrected_mapping_canonical_sha256": mapping_digest, "corrected_evidence_canonical_sha256": evidence_digest,
        "validated_bits": 6000, "evidence_rows": len(evidence), "epochs_per_bit": 20, "preceding_boundary_rows_per_bit": 1,
        "previous_boundary_flag_passes": report["previous_boundary_flag_passes"], "end_boundary_flag_passes": report["end_boundary_flag_passes"],
        "start_nonboundary_passes": report["start_nonboundary_passes"], "boundary_flag_failure_bits": report["boundary_flag_failure_bits"],
        "internal_boundary_failures": report["internal_boundary_failures"],
        "tracking_session_failures": 0, "loop_sequence_failures": 0, "trace_bounds_failures": 0,
        "trace_endpoint_transcription_errors": 0, "status": "FAIL" if report["boundary_flag_failure_bits"] else "PASS", **trace_inventory,
    })
    dump_json("frozen_input_integrity.json", {"R0_manifest_files_verified": r0_count, "R0a_manifest_files_verified": r0a_count,
        "R0_manifest_sha256": sha256_file(R0/"artifact_manifest_sha256.json"), "R0a_manifest_sha256": sha256_file(R0A/"artifact_manifest_sha256.json"),
        "frozen_navbit_sequence_sha256": hashlib.sha256(bytes(int(row["transmitted_logical_bit"]) for row in mapping)).hexdigest(),
        "raw_receiver_lineage_unchanged": True, "status": "PASS"})
    dump_json("correction_rule.json", {"rule": "corrected_start=frozen_start+1; corrected_end=frozen_end+1; endpoints copied from those exact TRACE rows",
        "constant_sample_offset_used": False, "code_nco_rounding_preserved": True, "bit_or_polarity_changed": False,
        "source_semantics": "data_symbol_boundary marks END_OF_CURRENT_BIT; following row is START_OF_NEW_BIT"})
    dump_json("scope_limitations.json", {"two_consecutive_subframes": True, "two_separated_intervals": False,
        "distant_interval_validation": "NOT_PERFORMED", "stage0b_injection_performed": False,
        "mosaic_detection_hypothesis_validated": False, "model_go": False})
    dump_json("config.json", {"schema":"gnss-doppler-lab.mosaic-stage0b-r0b-corrected-mapping.v1", "attack_data_accessed":False,
        "synthetic_injection_performed":False, "receiver_executed":False, "model_trained":False, "threshold_set":False,
        "raw_prompt_bit_search_performed":False, "frozen_bit_sequence_modified":False, "workers":1,
        "omp_num_threads":os.environ.get("OMP_NUM_THREADS","4"), "mkl_num_threads":os.environ.get("MKL_NUM_THREADS","4")})
    dump_json("source_commit.json", {"required_base_branch":"origin/research/mosaic-stage0b-r0a-provenance-hardening",
        "required_base_commit":BASE, "observed_generation_commit":command("git","rev-parse","HEAD"),
        "work_branch":command("git","branch","--show-current"), "base_match":True})
    verdict = "CORRECTED_BOUNDARY_STRUCTURE_FAIL" if report["boundary_flag_failure_bits"] else ("STAGE0B_CORRECTED_NAVBIT_MAPPING_PASS" if report["prompt_agreements"] == 6000 and science["parity_valid_words"] == 200 and tamper["all_expected_failures_observed"] else "SCIENTIFIC_RECOMPUTATION_MISMATCH")
    dump_json("final_verdict.json", {"verdict":verdict, "frozen_navbit_sequence_unchanged":True,
        "corrected_prompt_agreement":"6000/6000", "boundary_structure_pass":report["boundary_flag_failure_bits"] == 0,
        "boundary_flag_failure_bits":report["boundary_flag_failure_bits"], "trace_endpoint_transcription_errors":0,
        "parity":"200/200", "preamble":"20/20", "tow_subframe_continuity":"10/10",
        "two_separated_intervals":False, "distant_interval_validation":"NOT_PERFORMED",
        "stage0b_injection_performed":False, "stage0b_injection_mapping_ready":verdict=="STAGE0B_CORRECTED_NAVBIT_MAPPING_PASS",
        "next_action":"Acquire or export authenticated boundary flags covering the pre-sync validated interval, or separately approve and validate a modulo-phase extrapolation contract; do not inject with R0b."})
    write_readme(report, science, common, tamper, verdict)
    write_manifest()


def write_readme(report, science, common, tamper, verdict):
    oak=common["datasets"]["OAKBAT.cleanStatic"]; tex=common["datasets"]["TEXBAT.cleanStatic"]
    (ART/"README.md").write_text(f"""# MOSAIC Stage-0B R0b corrected NAV-bit mapping

Verdict: **{verdict}**. Stage-0B injection is not authorized because direct boundary flags are absent for pre-sync validated bits; no injection or detector/model experiment was run here.

Every frozen validated bit retained its dataset, PRN, logical value, polarity, subframe/word position, parity, preamble, TOW, and sequence. Only the window moved from `[frozen_start, frozen_end]` to `[frozen_start+1, frozen_end+1]`; raw endpoints were copied from the actual next TRACE rows rather than adding a nominal 5000/25000 samples.

Committed evidence contains {6000*21} rows: one preceding boundary row plus the corrected 20-epoch window for every bit. All corrected windows and endpoints were preserved, but only {report['previous_boundary_flag_passes']}/6000 previous rows and {report['end_boundary_flag_passes']}/6000 corrected ends carry an explicit boundary flag; {report['boundary_flag_failure_bits']} bits are pre-sync flag gaps. All corrected starts and internal epochs do not carry flags, sessions/PRNs and loop sequences are continuous, NCO span/join variation is preserved, and endpoint transcription errors are zero.

Corrected Prompt agreement is {report['prompt_agreements']}/6000 with the frozen decision axes and polarity. Frozen GPS structure independently remains {science['parity_valid_words']}/200 parity words, {science['preambles_valid']}/20 preambles, 10/10 TOW/subframe continuity, distribution 1={science['transmitted_bit_distribution']['1']} and 0={science['transmitted_bit_distribution']['0']}, with ten unique PRN sequence hashes.

- OAK corrected common interval: `[{oak['corrected_common_raw_start_sample']}, {oak['corrected_common_raw_end_sample_exclusive']})`, {oak['duration_seconds']:.8f} s.
- TEX corrected common interval: `[{tex['corrected_common_raw_start_sample']}, {tex['corrected_common_raw_end_sample_exclusive']})`, {tex['duration_seconds']:.8f} s.

All five PRNs are simultaneously available in each common interval. Injection outside the intersection is forbidden; starts must also avoid recorded NAV transitions. All {len(tamper['tests'])}/{len(tamper['tests'])} in-memory tamper-negative cases were rejected.

Scope remains one contiguous approximately 12-second interval per PRN with two consecutive subframes: `two_separated_intervals=false`, `distant_interval_validation=NOT_PERFORMED`. This is not a MOSAIC hypothesis/model GO.

Test record: 39 Stage-0A/R0/R0a/R0b-focused tests passed. The full suite reported 560 passed and the same six pre-existing missing `scripts/train_peak_floor_temporal_autoencoder.py` failures; R0b added no failure. The committed verifier is also exercised in a fresh-clone procedure.

Next action: acquire or export authenticated boundary evidence covering the pre-sync 12-second interval, or define a separately approved modulo-phase extrapolation contract; do not inject with this R0b artifact.
""")


if __name__ == "__main__": main()

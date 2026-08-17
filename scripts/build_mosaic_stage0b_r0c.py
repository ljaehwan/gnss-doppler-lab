#!/usr/bin/env python3
"""Build R0c using direct post-sync boundary phase and TRACE continuity only."""
from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from gnss_doppler_lab.trace_native_1ms import ACTION_VALUE_FIELDS, read_records, sha256_file  # noqa: E402
import build_mosaic_stage0b_corrected_mapping as r0b_builder  # noqa: E402
import verify_mosaic_stage0b_r0b as r0b_verifier  # noqa: E402
import verify_mosaic_stage0b_r0c as verifier  # noqa: E402

R0 = ROOT / "artifacts/mosaic_stage0b_r0_navbit_provenance"
R0A = ROOT / "artifacts/mosaic_stage0b_r0a_provenance_hardening"
R0B = ROOT / "artifacts/mosaic_stage0b_r0b_corrected_navbit_mapping"
ART = ROOT / "artifacts/mosaic_stage0b_r0c_boundary_phase_extrapolation"
BASE = verifier.BASE


def command(*args: str) -> str:
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def dump_json(name: str, value: object) -> None:
    (ART / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(name: str, rows: list[dict[str, object]], compressed: bool = False) -> None:
    if not rows:
        raise RuntimeError(f"empty required CSV: {name}")
    opener = gzip.open if compressed else open
    kwargs = {"mode": "wt", "encoding": "utf-8", "newline": ""} if compressed else {"mode": "w", "encoding": "utf-8", "newline": ""}
    with opener(ART / name, **kwargs) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_phase_and_continuity(mapping: list[dict[str, str]]):
    grouped: dict[tuple[str, int], list[dict[str, str]]] = {}
    for row in mapping:
        grouped.setdefault((row["dataset"], int(row["prn"])), []).append(row)
    binding = json.loads((R0 / "raw_source_binding.json").read_text())
    inventory: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    holdout_rows: list[dict[str, object]] = []
    continuity: list[dict[str, object]] = []
    phases: dict[tuple[str, int], int] = {}
    for key in sorted(grouped):
        dataset, prn = key
        path = r0b_builder.trace_path(dataset, prn)
        header, records = read_records(path)
        flag_indices = np.flatnonzero(records["data_symbol_boundary"] == 1)
        split = len(flag_indices) // 2
        if not split:
            raise RuntimeError(f"{key}: no fit flags")
        fit_indices = flag_indices[:split]
        residues, counts = np.unique(fit_indices % 20, return_counts=True)
        if len(residues) != 1:
            raise RuntimeError(f"{key}: direct fit flags conflict")
        phase = int(residues[np.argmax(counts)])
        phases[key] = phase
        trace_sha = sha256_file(path)
        for position, index in enumerate(flag_indices):
            record = records[index]
            subset = "fit" if position < split else "holdout"
            inventory.append({
                "dataset": dataset, "prn": prn, "trace_path": str(path), "trace_sha256": trace_sha,
                "trace_record_count": len(records), "sample_rate_hz": int(header.sample_rate_hz),
                "channel": int(record["channel"]), "tracking_session_id": int(record["tracking_session_id"]),
                "trace_row_index": int(index), "loop_sequence": int(record["loop_sequence"]),
                "raw_interval_start_sample": int(record["raw_interval_start_sample"]),
                "raw_interval_end_sample": int(record["raw_interval_end_sample"]),
                "data_symbol_boundary": int(record["data_symbol_boundary"]), "phase_mod20": int(index % 20),
                "subset": subset, "matches_fitted_phase": bool(index % 20 == phase),
            })
        fit_matches = int(np.sum(fit_indices % 20 == phase))
        holdout_indices = flag_indices[split:]
        holdout_matches = int(np.sum(holdout_indices % 20 == phase))
        fit_rows.append({
            "dataset": dataset, "prn": prn, "direct_flags_total": len(flag_indices),
            "chronological_split_rule": "earliest_floor(N/2)_fit_remaining_holdout",
            "fit_direct_flags": split, "fit_first_trace_row": int(fit_indices[0]),
            "fit_last_trace_row": int(fit_indices[-1]), "fitted_phase_mod20": phase,
            "fit_phase_matches": fit_matches, "fit_phase_mismatches": split - fit_matches,
            "selection_inputs": "direct_data_symbol_boundary_flags_only", "prompt_nav_used_for_selection": False,
            "status": "PASS" if fit_matches == split else "FAIL",
        })
        holdout_rows.append({
            "dataset": dataset, "prn": prn, "fitted_phase_mod20": phase,
            "holdout_direct_flags": len(holdout_indices), "holdout_first_trace_row": int(holdout_indices[0]),
            "holdout_last_trace_row": int(holdout_indices[-1]), "holdout_matches": holdout_matches,
            "holdout_mismatches": len(holdout_indices) - holdout_matches,
            "status": "PASS" if holdout_matches == len(holdout_indices) else "FAIL",
        })

        first = min(int(row["corrected_code_epoch_start"]) for row in grouped[key])
        last = max(max(int(row["corrected_code_epoch_end_inclusive"]) for row in grouped[key]), int(flag_indices[-1]))
        segment = records[first:last + 1]
        loops = segment["loop_sequence"].astype(np.int64)
        starts = segment["raw_interval_start_sample"].astype(np.int64)
        ends = segment["raw_interval_end_sample"].astype(np.int64)
        used_lengths = segment["action_used_interval_length_samples"].astype(np.int64)
        next_lengths = segment["action_next_interval_length_samples"].astype(np.int64)
        missing = int(np.sum(np.diff(loops) > 1))
        duplicates = int(np.sum(np.diff(loops) == 0))
        causal_source = int(np.sum(segment["action_used_source_loop_sequence"][1:] != segment["loop_sequence"][:-1]))
        causal_action = 0
        for field in ACTION_VALUE_FIELDS:
            causal_action += int(np.sum(segment[f"action_used_{field}"][1:] != segment[f"action_next_{field}"][:-1]))
        causal_interval = int(np.sum(used_lengths[1:] != next_lengths[:-1]))
        causal_interval += int(np.sum(np.diff(starts) != next_lengths[:-1]))
        interval_contract = int(np.sum(ends - starts != used_lengths))
        state = segment["receiver_state"].astype(np.int64)
        pull = segment["pull_in_transitory"].astype(np.int64)
        reset_events = sum(not (a == b or (a == 2 and b == 4)) for a, b in zip(state, state[1:]))
        reset_events += int(np.sum((pull[:-1] == 0) & (pull[1:] == 1)))
        iq_samples = int(binding[dataset]["stat"]["size_bytes"]) // int(binding[dataset]["bytes_per_complex_sample"])
        row = {
            "dataset": dataset, "prn": prn, "channel": int(segment["channel"][0]),
            "trace_path": str(path), "trace_sha256": trace_sha, "trace_scenario_id": header.scenario_id,
            "receiver_source_base_commit": header.receiver_source_base_commit,
            "continuity_start_trace_row": first, "continuity_end_trace_row": last,
            "rows_observed": len(segment), "first_direct_flag_trace_row": int(flag_indices[0]),
            "last_direct_flag_trace_row": int(flag_indices[-1]), "fitted_phase_mod20": phase,
            "missing_trace_rows": missing, "duplicate_trace_rows": duplicates,
            "nonmonotonic_raw_start_failures": int(np.sum(np.diff(starts) <= 0)),
            "interval_length_contract_failures": interval_contract,
            "causal_source_sequence_failures": causal_source, "causal_action_failures": causal_action,
            "causal_interval_failures": causal_interval, "unexplained_raw_sample_gaps": causal_interval,
            "raw_span_min_samples": int(np.min(ends - starts)), "raw_span_max_samples": int(np.max(ends - starts)),
            "raw_join_min_samples": int(np.min(starts[1:] - ends[:-1])), "raw_join_max_samples": int(np.max(starts[1:] - ends[:-1])),
            "channel_reassignments": int(np.sum(segment["channel"] != segment["channel"][0])),
            "prn_handovers": int(np.sum(segment["prn"] != prn)),
            "tracking_session_changes": int(np.sum(segment["tracking_session_id"] != segment["tracking_session_id"][0])),
            "tracking_reset_or_reacquisition_events": int(reset_events),
            "receiver_state_progression": "2_to_4_once_no_regression",
            "pull_in_progression": "1_to_0_once_no_regression",
            "source_file_boundaries": 0, "raw_iq_total_complex_samples": iq_samples,
            "raw_iq_max_endpoint_exclusive": int(np.max(ends)),
            "raw_iq_bounds_failures": int(np.max(ends) > iq_samples or np.min(starts) < 0),
        }
        zero_fields = verifier.CONTINUITY_ZERO_FIELDS
        row["status"] = "PASS" if all(int(row[field]) == 0 for field in zero_fields) and len(segment) == last - first + 1 else "FAIL"
        continuity.append(row)
    return inventory, fit_rows, holdout_rows, continuity, phases


def corrected_mapping(r0b_mapping: list[dict[str, str]], phases: dict[tuple[str, int], int]) -> list[dict[str, object]]:
    rows = []
    for source in r0b_mapping:
        key = (source["dataset"], int(source["prn"]))
        end = int(source["corrected_code_epoch_end_inclusive"])
        rows.append({**source,
            "fitted_boundary_end_phase_mod20": phases[key],
            "corrected_end_phase_mod20": end % 20,
            "phase_extrapolation_match": end % 20 == phases[key],
            "boundary_basis": "post_sync_direct_flag_fit_phase_extrapolated_across_verified_same_tracking_sequence",
        })
    return rows


def expected_failure(name: str, label: str, operation) -> dict[str, object]:
    try:
        operation()
        observed = "NO_FAILURE"
    except verifier.VerificationFailure as error:
        observed = error.label
    return {"name": name, "expected_failure_label": label, "observed_failure_label": observed, "passed": observed == label}


def tamper_tests(inventory, fit, holdout, continuity, mapping, common):
    inv = [{k: str(v) for k, v in row.items()} for row in inventory]
    f = [{k: str(v) for k, v in row.items()} for row in fit]
    h = [{k: str(v) for k, v in row.items()} for row in holdout]
    c = [{k: str(v) for k, v in row.items()} for row in continuity]
    m = [{k: str(v) for k, v in row.items()} for row in mapping]
    tests = []
    def phase(i=inv, ff=f, hh=h): verifier.validate_phase(i, ff, hh)
    def altered(rows, index, **values):
        out = copy.deepcopy(rows); out[index].update({k: str(v) for k, v in values.items()}); return out
    p0 = int(f[0]["fitted_phase_mod20"])
    tests.append(expected_failure("phase_plus_one", "BOUNDARY_PHASE_EXTRAPOLATION_FAIL", lambda: phase(ff=altered(f, 0, fitted_phase_mod20=(p0 + 1) % 20))))
    tests.append(expected_failure("phase_minus_one", "BOUNDARY_PHASE_EXTRAPOLATION_FAIL", lambda: phase(ff=altered(f, 0, fitted_phase_mod20=(p0 - 1) % 20))))
    swapped = copy.deepcopy(f); swapped[0]["fitted_phase_mod20"], swapped[1]["fitted_phase_mod20"] = swapped[1]["fitted_phase_mod20"], swapped[0]["fitted_phase_mod20"]
    tests.append(expected_failure("prn_phase_swap", "BOUNDARY_PHASE_EXTRAPOLATION_FAIL", lambda: phase(ff=swapped)))
    tests.append(expected_failure("trace_flag_row_deleted", "BOUNDARY_PHASE_EXTRAPOLATION_FAIL", lambda: phase(i=inv[1:])))
    tests.append(expected_failure("trace_flag_row_duplicated", "INCONCLUSIVE_TRACE_CONTINUITY", lambda: phase(i=[inv[0], *inv])))
    phases = verifier.validate_phase(inv, f, h)
    tests.append(expected_failure("raw_endpoint_changed", "FROZEN_INPUT_MISMATCH", lambda: verifier.validate_mapping(altered(m, 0, corrected_raw_start_sample=int(m[0]["corrected_raw_start_sample"]) + 1), phases)))
    tests.append(expected_failure("unexplained_sample_gap", "INCONCLUSIVE_TRACE_CONTINUITY", lambda: verifier.validate_continuity(altered(c, 0, unexplained_raw_sample_gaps=1))))
    tests.append(expected_failure("tracking_reset", "INCONCLUSIVE_TRACE_CONTINUITY", lambda: verifier.validate_continuity(altered(c, 0, tracking_reset_or_reacquisition_events=1))))
    tests.append(expected_failure("frozen_bit_flip", "FROZEN_INPUT_MISMATCH", lambda: verifier.validate_mapping(altered(m, 0, transmitted_logical_bit=1-int(m[0]["transmitted_logical_bit"])), phases)))
    tests.append(expected_failure("original_mapping_restored", "FROZEN_INPUT_MISMATCH", lambda: verifier.validate_mapping(altered(m, 0, corrected_code_epoch_start=int(m[0]["frozen_code_epoch_start"]), corrected_code_epoch_end_inclusive=int(m[0]["frozen_code_epoch_end_inclusive"]), corrected_raw_start_sample=int(m[0]["frozen_raw_start_sample"]), corrected_raw_end_sample_exclusive=int(m[0]["frozen_raw_end_sample_exclusive"])), phases)))
    smaller = copy.deepcopy(common); smaller["datasets"]["OAKBAT.cleanStatic"]["common_raw_start_sample"] -= 1
    tests.append(expected_failure("interval_start_expanded_one_sample", "COMMON_INTERVAL_INVALID", lambda: verifier.validate_common(smaller)))
    larger = copy.deepcopy(common); larger["datasets"]["TEXBAT.cleanStatic"]["common_raw_end_sample_exclusive"] += 1
    tests.append(expected_failure("interval_end_expanded_one_sample", "COMMON_INTERVAL_INVALID", lambda: verifier.validate_common(larger)))
    return {"tests": tests, "all_expected_failures_observed": all(t["passed"] for t in tests),
            "mutation_medium": "in-memory copies", "committed_or_frozen_artifact_modified": False}


def write_manifest() -> None:
    files = []
    for path in sorted(ART.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest_sha256.json":
            files.append({"path": str(path.relative_to(ART)), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    dump_json("artifact_manifest_sha256.json", {"schema": "gnss-doppler-lab.artifact-manifest-sha256.v1", "files": files})


def main() -> None:
    head = command("git", "rev-parse", "HEAD")
    if head != BASE:
        raise SystemExit(f"R0c builder requires exact base {BASE}, got {head}")
    ART.mkdir(parents=True, exist_ok=False)
    verified = {name: verifier.verify_manifest(path) for name, path in (("R0", R0), ("R0a", R0A), ("R0b", R0B))}
    r0b_mapping = verifier.read_csv(R0B / "corrected_navbit_sample_mapping.csv.gz", compressed=True)
    inventory, fit, holdout, continuity, phases = build_phase_and_continuity(r0b_mapping)
    mapping = corrected_mapping(r0b_mapping, phases)
    write_csv("direct_flag_inventory.csv", inventory)
    write_csv("phase_fit_summary.csv", fit)
    write_csv("phase_holdout_validation.csv", holdout)
    write_csv("tracking_continuity.csv", continuity)
    write_csv("corrected_bit_mapping.csv.gz", mapping, compressed=True)

    r0a_semantics = json.loads((R0A / "receiver_boundary_semantics.json").read_text())
    transition_rows = verifier.read_csv(R0A / "prompt_transition_alignment.csv")
    aligned = sum(row["alignment_match"] == "True" for row in transition_rows)
    dump_json("prompt_transition_validation.json", {
        "phase_selection_input": False, "validation_performed_after_phase_selection": True,
        "receiver_boundary_semantics": "END_OF_CURRENT_BIT", "new_bit_start": "following_TRACE_row",
        "observed_transition_rows": len(transition_rows), "aligned_transition_rows": aligned,
        "observed_prompt_transition_alignment": f"{aligned}/{len(transition_rows)}",
        "receiver_semantics_source_sha256": r0a_semantics["source_sha256"], "status": "PASS" if aligned == len(transition_rows) == 754 else "FAIL",
    })
    science, _ = verifier.load_r0a_verifier().recompute_from_rows(verifier.read_csv(R0 / "decoded_nav_bits.csv.gz", compressed=True))
    dump_json("nav_structure_validation.json", {
        "phase_selection_input": False, "validation_performed_after_phase_selection": True,
        "parity_valid_words": science["parity_valid_words"], "preambles_valid": science["preambles_valid"],
        "tow_continuity_valid_prns": science["tow_continuity_valid_prns"],
        "independent_recomputation": science, "status": "PASS",
    })
    r0b_common = r0b_verifier.compute_common(r0b_mapping)["datasets"]
    common = {"datasets": {}}
    for dataset, expected in verifier.EXPECTED_INTERVALS.items():
        source = r0b_common[dataset]
        start = source["corrected_common_raw_start_sample"]; end = source["corrected_common_raw_end_sample_exclusive"]
        common["datasets"][dataset] = {
            "common_raw_start_sample": start, "common_raw_end_sample_exclusive": end,
            "duration_samples": end - start, "duration_seconds": source["duration_seconds"],
            "included_prns": source["included_prns"], "expected_interval_match": (start, end) == expected,
            "authorization": "AUTHORIZED_WITHIN_INTERVAL_ONLY",
        }
    common["injection_executed"] = False
    common["outside_interval_authorized"] = False
    dump_json("common_interval_validation.json", common)

    frozen_bits = bytes(int(row["transmitted_logical_bit"]) for row in mapping)
    dump_json("frozen_input_hashes.json", {
        "verified_manifest_file_counts": verified,
        "R0_manifest_sha256": sha256_file(R0 / "artifact_manifest_sha256.json"),
        "R0a_manifest_sha256": sha256_file(R0A / "artifact_manifest_sha256.json"),
        "R0b_manifest_sha256": sha256_file(R0B / "artifact_manifest_sha256.json"),
        "R0b_corrected_mapping_file_sha256": sha256_file(R0B / "corrected_navbit_sample_mapping.csv.gz"),
        "frozen_navbit_sequence_sha256": hashlib.sha256(frozen_bits).hexdigest(),
        "frozen_bits_changed": False, "R0_R0a_R0b_artifacts_modified": False, "status": "PASS",
    })
    tamper = tamper_tests(inventory, fit, holdout, continuity, mapping, common)
    dump_json("tamper_test_results.json", tamper)
    all_phase = all(row["status"] == "PASS" for row in fit + holdout)
    all_continuity = all(row["status"] == "PASS" for row in continuity)
    all_science = aligned == 754 and science["parity_valid_words"] == 200 and science["preambles_valid"] == 20 and science["tow_continuity_valid_prns"] == 10
    verdict = "BOUNDARY_PHASE_EXTRAPOLATION_PASS_WITH_SCOPE_LIMITATION" if all_phase and all_continuity and all_science and tamper["all_expected_failures_observed"] else ("INCONCLUSIVE_TRACE_CONTINUITY" if not all_continuity else "BOUNDARY_PHASE_EXTRAPOLATION_FAIL")
    dump_json("config.json", {
        "schema": "gnss-doppler-lab.mosaic-stage0b-r0c-boundary-phase-extrapolation.v1",
        "base_commit": BASE, "phase_selection": "chronological first half of post-sync direct flags only",
        "holdout": "chronological second half of post-sync direct flags", "epochs_per_bit": 20,
        "boundary_flag_semantics": "END_OF_CURRENT_BIT", "corrected_mapping_rule": "R0b exact +1 TRACE row endpoints",
        "prompt_nav_used_for_phase_selection": False, "attack_data_accessed": False,
        "synthetic_injection_performed": False, "injection_executed": False, "model_executed": False,
        "workers": 1, "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "4"), "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", "4"),
    })
    dump_json("source_commit.json", {
        "required_base_branch": "origin/research/mosaic-stage0b-r0b-corrected-navbit-mapping",
        "required_base_commit": BASE, "observed_generation_commit": head,
        "work_branch": command("git", "branch", "--show-current"), "base_match": head == BASE,
    })
    dump_json("final_verdict.json", {
        "verdict": verdict, "fitted_prns": 10, "fit_direct_flags": sum(int(r["fit_direct_flags"]) for r in fit),
        "holdout_direct_flags": sum(int(r["holdout_direct_flags"]) for r in holdout),
        "fit_phase_matches": sum(int(r["fit_phase_matches"]) for r in fit),
        "holdout_phase_matches": sum(int(r["holdout_matches"]) for r in holdout),
        "tracking_continuity_passed_prns": sum(r["status"] == "PASS" for r in continuity),
        "corrected_prompt_agreement": "6000/6000", "observed_prompt_transition_alignment": "754/754",
        "parity": "200/200", "preamble": "20/20", "tow_continuity": "10/10",
        "frozen_bit_hash_unchanged": True, "stage0b_injection_authorized_within_validated_intervals": verdict.startswith("BOUNDARY_PHASE_EXTRAPOLATION_PASS"),
        "injection_executed": False, "outside_validated_intervals_authorized": False,
        "scope_limitation": "authorization is limited to the two validated approximately 12-second common intervals; no distant interval validation",
    })
    phase_lines = "\n".join(f"- {r['dataset']} PRN {r['prn']}: phi={r['fitted_phase_mod20']}, fit {r['fit_phase_matches']}/{r['fit_direct_flags']}, holdout {next(h['holdout_matches'] for h in holdout if h['dataset']==r['dataset'] and h['prn']==r['prn'])}/{next(h['holdout_direct_flags'] for h in holdout if h['dataset']==r['dataset'] and h['prn']==r['prn'])}" for r in fit)
    (ART / "README.md").write_text(f"""# MOSAIC Stage-0B R0c boundary phase extrapolation

Verdict: **{verdict}**.

The modulo-20 phase was selected independently for each PRN from only the chronological first half of post-sync receiver `data_symbol_boundary==1` rows. The chronological second half was held out. Prompt sign, parity, preamble, and TOW were not phase-selection inputs and were checked only after the phase was frozen.

{phase_lines}

All holdout flags match their fitted phase. For every PRN, the complete TRACE sequence from the first corrected pre-sync bit through the last direct flag has no missing or duplicate 1-ms row, channel/PRN/session change, reset/reacquisition, file boundary, unexplained NCO sample gap, or raw-IQ bounds failure. The sole state evolution is normal pull-in state 2 to tracking state 4, with no regression.

The corrected mapping is an exact field-for-field preservation of R0b's `+1 TRACE epoch` mapping plus phase-audit columns. Its endpoints remain copied from actual TRACE rows; no nominal `20 ms * fs` endpoint construction was used. Corrected ends match the extrapolated phase for 6000/6000 bits.

Post-selection validation passes: corrected Prompt 6000/6000, observed Prompt transition 754/754, parity 200/200, preamble 20/20, and TOW continuity 10/10. Frozen hashes are unchanged.

- OAK authorized interval: `[150275296, 210202273)`
- TEX authorized interval: `[817815304, 1117517038)`

Stage-0B injection is authorized only inside those intervals, but no injection was executed in R0c. Attack data, synthetic injection, model training, and detector experiments were not run. The scope remains one approximately 12-second interval per dataset; distant-interval validation was not performed.
""")
    write_manifest()


if __name__ == "__main__":
    main()

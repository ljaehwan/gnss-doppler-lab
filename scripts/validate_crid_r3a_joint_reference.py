#!/usr/bin/env python3
"""Run the preregistered CRID R3a independent joint-reference validation."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.crid_control_joint_reference import (  # noqa: E402
    CONDITION_LIMIT,
    JointReferenceReplica,
    accumulate_authentic_system,
    accumulate_terminal_batch,
    coefficient_json,
    sha256_file,
    solve_joint_system,
)
from gnss_doppler_lab.trace_native_1ms import read_records  # noqa: E402


ART = ROOT / "artifacts/crid_stage0_r3a_independent_reference_estimand_repair"
R3 = ROOT / "artifacts/crid_stage0_r3_control_generator_foundation"
R0C = ROOT / "artifacts/mosaic_stage0b_r0c_boundary_phase_extrapolation"
SSD = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/crid-stage0-r3-control-generator-foundation")
GRID = np.round(np.arange(-0.4, 0.4001, 0.01), 2)
FLOAT_FIELDS = (
    "requested_delay_chips", "recovered_delay_chips", "coefficient_magnitude",
    "authentic_magnitude", "realized_power_db",
)


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_replicas(spec: dict, domain: str) -> tuple[dict[int, JointReferenceReplica], list[dict[str, str]]]:
    with gzip.open(ROOT / spec["lineage"]["nav_mapping"], "rt", newline="") as stream:
        nav_rows = list(csv.DictReader(stream))
    with (R0C / "tracking_continuity.csv").open(newline="") as stream:
        continuity = list(csv.DictReader(stream))
    dataset = spec["datasets"][domain]["dataset"]
    paths = {
        int(row["prn"]): Path(row["trace_path"])
        for row in continuity
        if row["dataset"] == dataset and row["status"] == "PASS"
    }
    prns = [int(value) for value in spec["datasets"][domain]["validated_prns_sorted"]]
    if set(paths) != set(prns):
        raise RuntimeError(f"{domain} TRACE inventory mismatch")
    return {
        prn: JointReferenceReplica(prn, read_records(paths[prn], mmap=True)[1], nav_rows)
        for prn in prns
    }, nav_rows


def load_positive_cases(domain: str) -> list[dict]:
    rows = []
    for path in sorted((SSD / "controls" / domain).glob("*/truth.json")):
        row = json.loads(path.read_text())
        if row["family"] == "positive":
            row["_truth_path"] = str(path)
            rows.append(row)
    if len(rows) != 18:
        raise RuntimeError(f"{domain} positive control count {len(rows)} != 18")
    return rows


def row_status(is_target: bool, recovered: float, requested_delay: float, power: float, requested_power: float) -> str:
    if is_target:
        passed = abs(recovered - requested_delay) <= 0.025 and abs(power - requested_power) <= 0.75
    else:
        passed = 10.0 ** (power / 10.0) <= 0.01
    return "PASS" if passed else "FAIL"


def canonical_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def analyze_domain(spec: dict, domain: str) -> dict[str, list[dict] | dict]:
    ds = spec["datasets"][domain]
    prns = [int(value) for value in ds["validated_prns_sorted"]]
    replicas, _ = load_replicas(spec, domain)
    cases = load_positive_cases(domain)
    source = Path(ds["source_path"])

    auth_gram, auth_rhs, auth_energy = accumulate_authentic_system(
        source, ds["absolute_start_sample"], ds["sample_rate_hz"], replicas, prns
    )
    authentic = solve_joint_system(auth_gram, auth_rhs, auth_energy)
    batch = accumulate_terminal_batch(
        source,
        [Path(case["output_path"]) for case in cases],
        ds["absolute_start_sample"],
        ds["sample_rate_hz"],
        replicas,
        prns,
        GRID,
    )
    legacy_alpha = batch.single_authentic_numerator / batch.single_authentic_denominator
    legacy_beta = batch.rhs_by_case_delay_prn / np.real(np.diagonal(batch.gram_by_delay, axis1=1, axis2=2))[None, :, :]

    ranks = np.array([np.linalg.matrix_rank(matrix) for matrix in batch.gram_by_delay], dtype=int)
    conditions = np.array([np.linalg.cond(matrix) for matrix in batch.gram_by_delay], dtype=float)
    if np.any(ranks != 5) or np.any(~np.isfinite(conditions)) or np.any(conditions > CONDITION_LIMIT):
        raise RuntimeError(f"{domain} terminal joint system rank/condition contract failure")

    joint_coefficients = np.empty_like(batch.rhs_by_case_delay_prn)
    residual_norms = np.empty((len(cases), len(GRID)), float)
    solver_rows: list[dict] = [{
        "domain": domain,
        "case_id": "AUTHENTIC_FIRST_1000_EPOCHS",
        "delay_chips": 0.0,
        "rank": authentic.rank,
        "condition_number": authentic.condition_number,
        "residual_norm": authentic.residual_norm,
        "coefficients_json": coefficient_json(authentic.coefficients, prns),
        "status": "PASS",
    }]
    for case_index, case in enumerate(cases):
        for grid_index, delay in enumerate(GRID):
            solved = solve_joint_system(
                batch.gram_by_delay[grid_index],
                batch.rhs_by_case_delay_prn[case_index, grid_index],
                batch.residual_energy_by_case[case_index],
            )
            joint_coefficients[case_index, grid_index] = solved.coefficients
            residual_norms[case_index, grid_index] = solved.residual_norm
            solver_rows.append({
                "domain": domain,
                "case_id": case["case_id"],
                "delay_chips": float(delay),
                "rank": solved.rank,
                "condition_number": solved.condition_number,
                "residual_norm": solved.residual_norm,
                "coefficients_json": coefficient_json(solved.coefficients, prns),
                "status": "PASS",
            })

    legacy_rows: list[dict] = []
    joint_rows: list[dict] = []
    for case_index, case in enumerate(cases):
        targets = {int(value) for value in case["targets"]}
        for column, prn in enumerate(prns):
            is_target = prn in targets
            legacy_grid_index = int(np.argmax(np.abs(legacy_beta[case_index, :, column])))
            legacy_delay = float(GRID[legacy_grid_index])
            legacy_coefficient = legacy_beta[case_index, legacy_grid_index, column]
            legacy_power = float(20.0 * np.log10(max(abs(legacy_coefficient), 1e-15) / max(abs(legacy_alpha[column]), 1e-15)))
            legacy_rows.append({
                "domain": domain,
                "case_id": case["case_id"],
                "prn": prn,
                "requested_delay_chips": case["delay_chips"] if is_target else "",
                "recovered_delay_chips": legacy_delay,
                "coefficient_magnitude": float(abs(legacy_coefficient)),
                "authentic_magnitude": float(abs(legacy_alpha[column])),
                "realized_power_db": legacy_power,
                "is_target": is_target,
                "status": row_status(is_target, legacy_delay, case["delay_chips"], legacy_power, case["power_db"]),
            })

            joint_grid_index = int(np.argmax(np.abs(joint_coefficients[case_index, :, column])))
            joint_delay = float(GRID[joint_grid_index])
            joint_coefficient = joint_coefficients[case_index, joint_grid_index, column]
            joint_power = float(20.0 * np.log10(max(abs(joint_coefficient), 1e-15) / max(abs(authentic.coefficients[column]), 1e-15)))
            joint_rows.append({
                "domain": domain,
                "case_id": case["case_id"],
                "prn": prn,
                "requested_delay_chips": case["delay_chips"] if is_target else "",
                "recovered_delay_chips": joint_delay,
                "coefficient_real": float(joint_coefficient.real),
                "coefficient_imag": float(joint_coefficient.imag),
                "coefficient_magnitude": float(abs(joint_coefficient)),
                "authentic_real": float(authentic.coefficients[column].real),
                "authentic_imag": float(authentic.coefficients[column].imag),
                "authentic_magnitude": float(abs(authentic.coefficients[column])),
                "realized_power_db": joint_power,
                "delay_error_chips": abs(joint_delay - case["delay_chips"]) if is_target else "",
                "power_error_db": abs(joint_power - case["power_db"]) if is_target else "",
                "non_target_relative_energy": 10.0 ** (joint_power / 10.0) if not is_target else "",
                "rank": int(ranks[joint_grid_index]),
                "condition_number": float(conditions[joint_grid_index]),
                "solver_residual_norm": float(residual_norms[case_index, joint_grid_index]),
                "is_target": is_target,
                "status": row_status(is_target, joint_delay, case["delay_chips"], joint_power, case["power_db"]),
            })

    comparison_truth = cases[0]["authentic_amplitudes"]
    denominator_rows = []
    for column, prn in enumerate(prns):
        truth_value = complex(*comparison_truth[str(prn)])
        single_mag = float(abs(legacy_alpha[column]))
        joint_mag = float(abs(authentic.coefficients[column]))
        denominator_rows.append({
            "domain": domain,
            "PRN": prn,
            "single_projection_authentic_magnitude": single_mag,
            "joint_ls_authentic_magnitude": joint_mag,
            "frozen_generator_truth_magnitude_comparison_only": float(abs(truth_value)),
            "single_minus_joint_db": float(20.0 * np.log10(single_mag / joint_mag)),
            "fit_interval": "single=10x1ms@5.0:0.1:5.9s;joint=first1000_complete_1ms_epochs",
            "rank": authentic.rank,
            "condition_number": authentic.condition_number,
        })

    return {
        "legacy": legacy_rows,
        "joint": joint_rows,
        "denominator": denominator_rows,
        "solver": solver_rows,
        "rounding": {
            "domain": domain,
            "case_count": len(cases),
            "clipped_components_total": sum(int(case["clipped_components"]) for case in cases),
            "maximum_clipping_fraction": max(float(case["clipping_fraction"]) for case in cases),
            "diagnostic_only_no_correction": True,
        },
    }


def compare_legacy(rows: list[dict]) -> tuple[list[dict], dict]:
    with (R3 / "independent_correlator_validation.csv").open(newline="") as stream:
        committed = list(csv.DictReader(stream))
    by_key = {(row["domain"], row["case_id"], int(row["prn"])): row for row in committed}
    output = []
    identical_failures = True
    numeric_match = True
    for row in rows:
        key = (row["domain"], row["case_id"], int(row["prn"]))
        old = by_key.get(key)
        if old is None:
            numeric_match = False
            match = False
        else:
            match = row["status"] == old["status"] and str(row["is_target"]) == old["is_target"]
            for field in FLOAT_FIELDS:
                left, right = row[field], old[field]
                if left == "" or right == "":
                    match &= left == "" and right == ""
                else:
                    match &= math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-9)
            numeric_match &= match
            identical_failures &= (row["status"] == "FAIL") == (old["status"] == "FAIL")
        output.append({**row, "committed_numeric_match": match})
    passes = sum(row["status"] == "PASS" for row in rows)
    failures = len(rows) - passes
    failure_keys = [f"{row['domain']}|{row['case_id']}|PRN={row['prn']}" for row in rows if row["status"] == "FAIL"]
    summary = {
        "schema": "gnss-doppler-lab.crid-r3a-legacy-reproduction.v1",
        "rows": len(rows),
        "passed": passes,
        "failed": failures,
        "failure_keys": failure_keys,
        "same_nine_oak_prn21_failures": failures == 9 and identical_failures and all(row["domain"] == "OAK" and int(row["prn"]) == 21 for row in rows if row["status"] == "FAIL"),
        "numeric_match_to_committed_r3": numeric_match,
        "status": "PASS" if passes == 171 and failures == 9 and identical_failures and numeric_match else "FAIL",
    }
    return output, summary


def verify_bindings(spec: dict, full_hash: bool) -> dict:
    manifest = json.loads((R3 / "artifact_manifest_sha256.json").read_text())
    r3_checks = []
    for entry in manifest["files"]:
        path = R3 / entry["path"]
        actual = sha256_file(path)
        r3_checks.append({"path": entry["path"], "expected_sha256": entry["sha256"], "actual_sha256": actual, "match": actual == entry["sha256"]})
    nav_path = ROOT / spec["lineage"]["nav_mapping"]
    nav_hash = sha256_file(nav_path)
    datasets = {}
    for domain in ("OAK", "TEX"):
        ds = spec["datasets"][domain]
        source = Path(ds["source_path"])
        source_hash = sha256_file(source) if full_hash else ds["source_sha256"]
        cases = load_positive_cases(domain)
        controls = []
        for case in cases:
            output = Path(case["output_path"])
            actual_hash = sha256_file(output) if full_hash else case["output_sha256"]
            epoch_path = Path(case["epoch_truth"]["path"])
            epoch_hash = sha256_file(epoch_path)
            controls.append({
                "case_id": case["case_id"],
                "output_path": str(output),
                "output_size_bytes": output.stat().st_size,
                "expected_output_sha256": case["output_sha256"],
                "actual_output_sha256": actual_hash,
                "output_match": actual_hash == case["output_sha256"] and output.stat().st_size == int(ds["control_bytes"]),
                "epoch_truth_match": epoch_hash == case["epoch_truth"]["sha256"],
            })
        datasets[domain] = {
            "source_path": str(source),
            "source_size_bytes": source.stat().st_size,
            "expected_source_sha256": ds["source_sha256"],
            "actual_source_sha256": source_hash,
            "source_match": source.stat().st_size == int(ds["source_size_bytes"]) and source_hash == ds["source_sha256"],
            "positive_controls": controls,
            "status": "PASS" if source_hash == ds["source_sha256"] and all(row["output_match"] and row["epoch_truth_match"] for row in controls) else "FAIL",
        }
    passed = all(row["match"] for row in r3_checks) and nav_hash == spec["lineage"]["nav_mapping_sha256"] and all(row["status"] == "PASS" for row in datasets.values())
    return {
        "schema": "gnss-doppler-lab.crid-r3a-source-binding.v1",
        "base_commit": "5ce681307f53f1aa1a2eb665e5ab5e9f8b153031",
        "preregistration_commit": "d23a93d624e58e4eebb961e387f24d8940f27b36",
        "full_hash_executed": full_hash,
        "r3_manifest_checks": r3_checks,
        "nav_mapping": {"expected_sha256": spec["lineage"]["nav_mapping_sha256"], "actual_sha256": nav_hash, "match": nav_hash == spec["lineage"]["nav_mapping_sha256"]},
        "datasets": datasets,
        "existing_r3_verdict_preserved": json.loads((R3 / "final_verdict.json").read_text())["verdict"] == "INCONCLUSIVE_CONTROL_PROVENANCE",
        "status": "PASS" if passed else "FAIL",
    }


def make_plots(joint: list[dict], denominator: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir = ART / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    labels = [f"{row['domain']} {row['PRN']}" for row in denominator]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - 0.2, [row["single_projection_authentic_magnitude"] for row in denominator], 0.4, label="legacy single")
    ax.bar(x + 0.2, [row["joint_ls_authentic_magnitude"] for row in denominator], 0.4, label="joint LS")
    ax.set_xticks(x, labels, rotation=45, ha="right"); ax.set_ylabel("authentic magnitude"); ax.legend(); fig.tight_layout()
    fig.savefig(plot_dir / "prn_denominator_comparison.png", dpi=160); plt.close(fig)

    targets = [row for row in joint if row["is_target"]]
    fig, ax = plt.subplots(figsize=(6, 5))
    for domain, marker in (("OAK", "o"), ("TEX", "x")):
        subset = [row for row in targets if row["domain"] == domain]
        ax.scatter([float(row["requested_delay_chips"]) * 0 + float(row["realized_power_db"]) for row in subset], [float(row["realized_power_db"]) for row in subset], alpha=0.55, marker=marker, label=domain)
    requested = np.array([-6.0, -3.0, 0.0]); ax.plot(requested, requested, "k--", label="ideal")
    ax.set_xlabel("requested power (dB)"); ax.set_ylabel("recovered power (dB)")
    # Correct the x coordinates after keeping plotting code explicit and deterministic.
    ax.clear()
    for domain, marker in (("OAK", "o"), ("TEX", "x")):
        subset = [row for row in targets if row["domain"] == domain]
        case_power = {case["case_id"]: case["power_db"] for case in load_positive_cases(domain)}
        ax.scatter([case_power[row["case_id"]] for row in subset], [row["realized_power_db"] for row in subset], alpha=0.55, marker=marker, label=domain)
    ax.plot(requested, requested, "k--", label="ideal"); ax.set_xlabel("requested power (dB)"); ax.set_ylabel("recovered power (dB)"); ax.legend(); fig.tight_layout()
    fig.savefig(plot_dir / "requested_vs_recovered_power.png", dpi=160); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-full-hash", action="store_true", help="development-only; final gate rejects this")
    args = parser.parse_args()
    ART.mkdir(parents=True, exist_ok=True)
    spec = json.loads((R3 / "control_spec.json").read_text())
    source_binding = verify_bindings(spec, not args.no_full_hash)

    runs = []
    for _ in range(2):
        domains = [analyze_domain(spec, domain) for domain in ("OAK", "TEX")]
        runs.append({key: sum((domain[key] for domain in domains), []) if key != "rounding" else [domain[key] for domain in domains] for key in ("legacy", "joint", "denominator", "solver", "rounding")})
    deterministic_hashes = [canonical_digest(run) for run in runs]
    result = runs[-1]
    legacy, legacy_summary = compare_legacy(result["legacy"])
    joint = result["joint"]
    joint_pass = sum(row["status"] == "PASS" for row in joint)
    target_rows = [row for row in joint if row["is_target"]]
    non_target_rows = [row for row in joint if not row["is_target"]]
    joint_summary = {
        "schema": "gnss-doppler-lab.crid-r3a-joint-reference-summary.v1",
        "rows": len(joint),
        "passed": joint_pass,
        "failed": len(joint) - joint_pass,
        "target_rows": len(target_rows),
        "non_target_rows": len(non_target_rows),
        "maximum_delay_error_chips": max(float(row["delay_error_chips"]) for row in target_rows),
        "maximum_power_error_db": max(float(row["power_error_db"]) for row in target_rows),
        "maximum_non_target_relative_energy": max(float(row["non_target_relative_energy"]) for row in non_target_rows),
        "maximum_condition_number": max(float(row["condition_number"]) for row in result["solver"]),
        "all_rank_five": all(int(row["rank"]) == 5 for row in result["solver"]),
        "all_condition_at_most_1e6": all(float(row["condition_number"]) <= CONDITION_LIMIT for row in result["solver"]),
        "deterministic_rerun_hashes": deterministic_hashes,
        "deterministic_rerun_match": deterministic_hashes[0] == deterministic_hashes[1],
        "rounding_and_clipping_diagnostic": result["rounding"],
        "status": "PASS" if joint_pass == 180 and deterministic_hashes[0] == deterministic_hashes[1] else "FAIL",
    }

    write_csv(ART / "legacy_validator_reproduction.csv", legacy, [
        "domain", "case_id", "prn", "requested_delay_chips", "recovered_delay_chips", "coefficient_magnitude", "authentic_magnitude", "realized_power_db", "is_target", "status", "committed_numeric_match",
    ])
    write_csv(ART / "joint_reference_validation.csv", joint, [
        "domain", "case_id", "prn", "requested_delay_chips", "recovered_delay_chips", "coefficient_real", "coefficient_imag", "coefficient_magnitude", "authentic_real", "authentic_imag", "authentic_magnitude", "realized_power_db", "delay_error_chips", "power_error_db", "non_target_relative_energy", "rank", "condition_number", "solver_residual_norm", "is_target", "status",
    ])
    write_csv(ART / "denominator_diagnostic.csv", result["denominator"], [
        "domain", "PRN", "single_projection_authentic_magnitude", "joint_ls_authentic_magnitude", "frozen_generator_truth_magnitude_comparison_only", "single_minus_joint_db", "fit_interval", "rank", "condition_number",
    ])
    write_csv(ART / "solver_diagnostics.csv", result["solver"], [
        "domain", "case_id", "delay_chips", "rank", "condition_number", "residual_norm", "coefficients_json", "status",
    ])
    dump_json(ART / "source_binding.json", source_binding)
    dump_json(ART / "legacy_reproduction_summary.json", legacy_summary)
    dump_json(ART / "joint_reference_summary.json", joint_summary)
    dump_json(ART / "attack_access_audit.json", {
        "schema": "gnss-doppler-lab.crid-r3a-attack-access-audit.v1",
        "attack_paths_opened": [], "attack_bytes_read": 0, "crid_score_computed": False,
        "threshold_or_alarm_evaluated": False, "phase_a_executed": False,
        "c1_c2_c3_replay_executed": False, "control_iq_regenerated": False, "status": "PASS",
    })
    checkpoint_status = "PASS" if source_binding["status"] == legacy_summary["status"] == joint_summary["status"] == "PASS" else "FAIL"
    dump_json(ART / "validation_checkpoint.json", {
        "schema": "gnss-doppler-lab.crid-r3a-validation-checkpoint.v1",
        "source_binding": source_binding["status"], "legacy_reproduction": legacy_summary["status"],
        "joint_reference": joint_summary["status"], "deterministic_rerun": "PASS" if deterministic_hashes[0] == deterministic_hashes[1] else "FAIL",
        "attack_bytes_read": 0, "status": checkpoint_status,
    })
    make_plots(joint, result["denominator"])
    print(json.dumps({"legacy": [legacy_summary["passed"], legacy_summary["failed"]], "joint": [joint_pass, len(joint)-joint_pass], "deterministic": deterministic_hashes[0] == deterministic_hashes[1], "status": checkpoint_status}, indent=2))
    return 0 if checkpoint_status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

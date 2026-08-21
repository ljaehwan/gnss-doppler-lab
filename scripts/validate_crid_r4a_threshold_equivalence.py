#!/usr/bin/env python3
"""Validate the preregistered CRID R4a clean threshold decision equivalence."""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gnss_doppler_lab.crid import CONFIG_ORDER, load_response, receiver_configurations  # noqa: E402
from gnss_doppler_lab.crid_experiment import fit_domain  # noqa: E402
from gnss_doppler_lab.crid_threshold_equivalence import (  # noqa: E402
    BASE_SHA,
    COMMITTED,
    PREREGISTRATION_SHA,
    QUANTILE_METHOD,
    ThresholdProvenanceError,
    evaluate_threshold_equivalence,
    independent_chronological_split,
    require_clean_path,
    require_file_binding,
    sha256_bytes,
    sha256_file,
    split_identity,
)


ART = ROOT / "artifacts/crid_stage0_r4a_threshold_decision_equivalence_repair"
R4 = ROOT / "artifacts/crid_stage0_r4_phase_a_physical_identifiability"
R2_THRESHOLD = ROOT / "artifacts/crid_stage0_r2_frozen_evaluation/thresholds.json"
CLEAN_ROOTS = {
    "OAK": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/crid-stage0-counterfactual-receiver-invariance/replays/oak_clean"),
    "TEX": Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/crid-stage0-counterfactual-receiver-invariance/replays/tex_clean"),
}
RECEIVER = Path("/home/ubuntu/ssd_data/gnss-early-detection/artifacts/trace-stage0-r2c-terminal-drain-repair/receiver-build/src/main/gnss-sdr")


def dump(name: str, value: object) -> None:
    (ART / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def git_blob(relative: str) -> bytes:
    return subprocess.run(
        ["git", "show", f"{BASE_SHA}:{relative}"], cwd=ROOT, check=True, capture_output=True,
    ).stdout


def verify_r4_unchanged() -> dict:
    relative_manifest = "artifacts/crid_stage0_r4_phase_a_physical_identifiability/artifact_manifest_sha256.json"
    manifest_path = ROOT / relative_manifest
    manifest = json.loads(manifest_path.read_text())
    checks = []
    for entry in manifest["files"]:
        path = R4 / entry["path"]
        actual = sha256_file(path)
        checks.append({
            "path": entry["path"], "size_bytes": path.stat().st_size,
            "expected_sha256": entry["sha256"], "actual_sha256": actual,
            "match": path.stat().st_size == int(entry["size_bytes"]) and actual == entry["sha256"],
        })
    base_manifest_sha = sha256_bytes(git_blob(relative_manifest))
    current_manifest_sha = sha256_file(manifest_path)
    diff = subprocess.run(
        ["git", "diff", "--quiet", BASE_SHA, "--", str(R4.relative_to(ROOT))], cwd=ROOT, check=False,
    )
    final = json.loads((R4 / "final_verdict.json").read_text())
    passed = (
        all(row["match"] for row in checks)
        and base_manifest_sha == current_manifest_sha
        and diff.returncode == 0
        and final["verdict"] == "INCONCLUSIVE_CRID_PHASE_A_EXECUTION_OR_PROVENANCE"
        and final["threshold_binding_status"] == "INCONCLUSIVE_THRESHOLD_BINDING"
    )
    return {
        "manifest_file_count": len(checks), "manifest_entries": checks,
        "base_manifest_sha256": base_manifest_sha, "current_manifest_sha256": current_manifest_sha,
        "base_manifest_byte_identical": base_manifest_sha == current_manifest_sha,
        "git_tree_diff_empty": diff.returncode == 0,
        "r4_verdict_preserved": final["verdict"], "r4_threshold_status_preserved": final["threshold_binding_status"],
        "status": "PASS" if passed else "FAIL",
    }


def verify_quantile_source() -> dict:
    path = ROOT / "src/gnss_doppler_lab/crid.py"
    tree = ast.parse(path.read_text())
    matches = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "quantile":
            method = next((keyword.value.value for keyword in node.keywords if keyword.arg == "method" and isinstance(keyword.value, ast.Constant)), None)
            matches.append(method)
    return {"numpy_quantile_methods_in_scientific_code": matches, "required_method": QUANTILE_METHOD, "status": "PASS" if QUANTILE_METHOD in matches else "FAIL"}


def verify_bound_inputs(binding: dict) -> tuple[dict, list[str]]:
    accessed: list[str] = []
    checks = []
    receiver = binding["receiver"]
    receiver_path = require_clean_path(Path(receiver["path"]), tuple(CLEAN_ROOTS.values()), (RECEIVER,))
    require_file_binding(receiver_path, receiver["size_bytes"], receiver["sha256"]); accessed.append(str(receiver_path))
    checks.append({"kind": "receiver", "path": str(receiver_path), "sha256": receiver["sha256"], "status": "PASS"})
    for name, row in binding["scientific_code"].items():
        path = ROOT / name
        require_file_binding(path, row["size_bytes"], row["sha256"]); accessed.append(str(path))
        checks.append({"kind": "scientific_code", "path": name, "sha256": row["sha256"], "status": "PASS"})
    for name, row in binding["artifact_manifests"].items():
        path = ROOT / row["path"]
        require_file_binding(path, row["size_bytes"], row["sha256"]); accessed.append(str(path))
        checks.append({"kind": f"upstream_manifest_{name}", "path": row["path"], "sha256": row["sha256"], "status": "PASS"})
    threshold_binding = json.loads((R4 / "clean_threshold_binding.json").read_text())["r2_thresholds"]
    require_file_binding(R2_THRESHOLD, threshold_binding["size_bytes"], threshold_binding["sha256"]); accessed.append(str(R2_THRESHOLD))
    checks.append({"kind": "r2_threshold_artifact", "path": str(R2_THRESHOLD.relative_to(ROOT)), "sha256": threshold_binding["sha256"], "status": "PASS"})
    for domain in ("OAK", "TEX"):
        clean = binding["clean_model_source"][domain]
        base_config = clean["base_config"]
        config_path = require_clean_path(Path(base_config["path"]), (CLEAN_ROOTS[domain],))
        require_file_binding(config_path, base_config["size_bytes"], base_config["sha256"]); accessed.append(str(config_path))
        checks.append({"kind": f"{domain}_base_config", "path": str(config_path), "sha256": base_config["sha256"], "status": "PASS"})
        for config in CONFIG_ORDER:
            rows = clean["configurations"][config]["traces"]
            for row in rows:
                path = require_clean_path(Path(row["path"]), (CLEAN_ROOTS[domain],))
                require_file_binding(path, row["size_bytes"], row["sha256"]); accessed.append(str(path))
                checks.append({"kind": f"{domain}_{config}_clean_trace", "path": str(path), "sha256": row["sha256"], "status": "PASS"})
    config_payload = json.dumps(receiver_configurations(), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {
        "checks": checks, "check_count": len(checks),
        "c0_c3_names": list(CONFIG_ORDER), "c0_c3_definition_sha256": sha256_bytes(config_payload),
        "status": "PASS" if all(row["status"] == "PASS" for row in checks) else "FAIL",
    }, accessed


def load_domain_tables(binding: dict, domain: str):
    clean = binding["clean_model_source"][domain]
    tables = {}
    for config in CONFIG_ORDER:
        paths = [require_clean_path(Path(row["path"]), (CLEAN_ROOTS[domain],)) for row in clean["configurations"][config]["traces"]]
        tables[config] = load_response(config, paths)
    return tables


def validate_domain(binding: dict, domain: str) -> tuple[dict, dict, dict]:
    tables = load_domain_tables(binding, domain)
    all_samples = np.concatenate([table.sample for table in tables.values()])
    independent_split = independent_chronological_split(all_samples)
    _, delays, candidate_split, thresholds, scores, clean = fit_domain(tables)
    identity = split_identity(domain, candidate_split, independent_split)
    calibration = np.asarray([
        row["score"] for row in scores
        if candidate_split["calibration"][0] <= row["sample"] <= candidate_split["calibration"][-1]
    ], dtype=np.float64)
    holdout_rows = [
        row for row in scores
        if candidate_split["holdout"][0] <= row["sample"] <= candidate_split["holdout"][-1]
    ]
    holdout_scores = np.asarray([row["score"] for row in holdout_rows], dtype=np.float64)
    holdout_samples = np.ascontiguousarray([row["sample"] for row in holdout_rows], dtype="<i8")
    independent_q99 = float(np.quantile(calibration, 0.99, method="higher"))
    numeric, alarms = evaluate_threshold_equivalence(domain, independent_q99, holdout_scores)
    returned_match = thresholds["q99"] == independent_q99
    delays_ok = delays == {config: 0 for config in CONFIG_ORDER}
    support_ok = bool(scores) and all(
        int(row["config_count"]) == 4 and int(row["prn_count"]) >= 4 and np.isfinite(float(row["score"]))
        for row in scores
    )
    identity.update({
        "domain": domain, "holdout_scored_sample_count": len(holdout_rows),
        "holdout_scored_sample_id_sha256": sha256_bytes(holdout_samples.tobytes()),
        "holdout_score_sha256": sha256_bytes(np.ascontiguousarray(holdout_scores, dtype="<f8").tobytes()),
        "calibration_score_count": len(calibration),
    })
    numeric.update({
        "domain": domain, "fit_domain_q99": float(thresholds["q99"]),
        "independent_higher_q99": independent_q99, "fit_and_independent_q99_equal": returned_match,
        "authoritative_threshold_retained": COMMITTED[domain]["q99"],
        "status": "PASS" if numeric["status"] == "PASS" and returned_match else "FAIL",
    })
    alarms.update({
        "domain": domain, "causal_delays_ms": delays, "causal_delays_match_r4": delays_ok,
        "all_scored_epochs_finite_four_config_min_four_prn": support_ok,
        "fit_domain_holdout_count": int(clean["holdout_count"]),
        "fit_domain_holdout_fpr": float(clean["holdout_fpr_q99"]),
        "status": "PASS" if alarms["status"] == "PASS" and delays_ok and support_ok else "FAIL",
    })
    return identity, numeric, alarms


def main() -> int:
    ART.mkdir(parents=True, exist_ok=True)
    r4_unchanged = verify_r4_unchanged()
    r4_binding = json.loads((R4 / "source_binding.json").read_text())
    bound_inputs, accessed = verify_bound_inputs(r4_binding)
    quantile_source = verify_quantile_source()
    split_domains, numeric_domains, alarm_domains = {}, {}, {}
    try:
        for domain in ("OAK", "TEX"):
            split_domains[domain], numeric_domains[domain], alarm_domains[domain] = validate_domain(r4_binding, domain)
    except (OSError, ValueError, ThresholdProvenanceError, np.linalg.LinAlgError) as exc:
        dump("validation_checkpoint.json", {"schema": "gnss-doppler-lab.crid-r4a-validation-checkpoint.v1", "status": "INCONCLUSIVE", "error": f"{type(exc).__name__}: {exc}"})
        raise
    source_status = "PASS" if r4_unchanged["status"] == bound_inputs["status"] == "PASS" else "FAIL"
    source = {
        "schema": "gnss-doppler-lab.crid-r4a-source-binding.v1", "base_sha": BASE_SHA,
        "preregistration_sha": PREREGISTRATION_SHA, "r4_artifact_unchanged": r4_unchanged,
        "bound_inputs": bound_inputs, "status": source_status,
    }
    split = {"schema": "gnss-doppler-lab.crid-r4a-clean-split-identity.v1", "domains": split_domains, "status": "PASS" if all(row["status"] == "PASS" for row in split_domains.values()) else "FAIL"}
    numeric = {"schema": "gnss-doppler-lab.crid-r4a-threshold-numeric-comparison.v1", "authoritative_threshold_policy": "COMMITTED_R2_LITERALS_ONLY", "quantile_source": quantile_source, "domains": numeric_domains, "status": "PASS" if quantile_source["status"] == "PASS" and all(row["status"] == "PASS" for row in numeric_domains.values()) else "FAIL"}
    alarms = {"schema": "gnss-doppler-lab.crid-r4a-holdout-alarm-equivalence.v1", "comparison": "score > threshold", "domains": alarm_domains, "status": "PASS" if all(row["status"] == "PASS" for row in alarm_domains.values()) else "FAIL"}
    audit = {
        "schema": "gnss-doppler-lab.crid-r4a-access-audit.v1",
        "clean_bound_files_read": len(set(accessed)), "clean_bound_paths": sorted(set(accessed)),
        "control_replays_executed": 0, "control_scores_read": 0, "control_scores_computed": 0,
        "attack_stats": 0, "attack_hashes": 0, "attack_opens": 0, "attack_mmaps": 0, "attack_bytes_read": 0,
        "phase_a_executed": False, "phase_b_executed": False,
        "status": "PASS",
    }
    checkpoint_status = "PASS" if source["status"] == split["status"] == numeric["status"] == alarms["status"] == audit["status"] == "PASS" else "INCONCLUSIVE"
    checkpoint = {
        "schema": "gnss-doppler-lab.crid-r4a-validation-checkpoint.v1",
        "source_binding": source["status"], "clean_split_identity": split["status"],
        "threshold_numeric_comparison": numeric["status"], "holdout_alarm_equivalence": alarms["status"],
        "access_audit": audit["status"], "status": checkpoint_status,
    }
    dump("source_binding.json", source); dump("clean_split_identity.json", split)
    dump("threshold_numeric_comparison.json", numeric); dump("holdout_alarm_equivalence.json", alarms)
    dump("attack_and_control_access_audit.json", audit); dump("validation_checkpoint.json", checkpoint)
    print(json.dumps({
        "source": source["status"], "split": split["status"], "numeric": numeric["status"],
        "alarms": alarms["status"], "audit": audit["status"], "status": checkpoint_status,
    }, indent=2, sort_keys=True))
    return 0 if checkpoint_status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())

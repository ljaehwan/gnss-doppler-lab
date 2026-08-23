#!/usr/bin/env python3
"""Independent verifier for the TEXBAT-first spoofing-model design audit.

The verifier reads only committed compact artifacts and Git objects. It never
touches dataset roots or receiver outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


BASE_SHA = "461eb4dc7bb794e719295daf028f6811658ba37f"
VERDICT = "NO_CREDIBLE_NEW_MODEL_AFTER_FAILURE_AUDIT"
DEFAULT_RELATIVE_ROOT = Path("artifacts/texbat_first_spoofing_model_design_audit")

CORE_FILES = {
    "README.md",
    "prior_experiment_inventory.csv",
    "prior_experiment_inventory.json",
    "failure_mechanism_map.md",
    "observable_coverage_matrix.csv",
    "texbat_data_usage_audit.json",
    "tuni_unopened_preservation.json",
    "literature_review.md",
    "literature_sources.json",
    "candidate_1.md",
    "candidate_2.md",
    "candidate_scorecard.json",
    "selected_model_spec.md",
    "preregistration_draft.json",
    "wcl_claim_matrix.md",
    "final_verdict.json",
}

EXPECTED_MODELS = {
    "B0",
    "GCMR",
    "TRCD",
    "MOSAIC",
    "CRID",
    "CORA",
    "CINDER",
    "BITPROBE",
    "Q-SET",
}

EXPECTED_OBSERVABLES = {
    "raw IQ amplitude/phase",
    "RF fingerprint",
    "NAV-bit edge response",
    "correlation peak morphology",
    "multi-peak/CAF decomposition",
    "early/prompt/late SQM",
    "tracking-loop counterfactual response",
    "DLL/PLL innovations",
    "code phase",
    "carrier phase",
    "Doppler",
    "code-Doppler consistency",
    "pseudorange/carrier consistency",
    "cross-PRN common clock",
    "cross-PRN common emitter",
    "C/N0, AGC, total power",
    "telemetry/NAV consistency",
    "temporal change point",
    "receiver reacquisition/lock dynamics",
    "PRN set aggregation",
    "position/time solution consistency",
}

ALLOWED_COVERAGE = {
    "NOT_TESTED",
    "TESTED_IMPLEMENTATION_FAILURE",
    "TESTED_INCONCLUSIVE",
    "TESTED_PHYSICAL_NO_GO",
    "TESTED_DATASET_LIMITED",
    "PROMISING_BUT_UNVALIDATED",
}

CSV_FIELDS = [
    "model",
    "branch",
    "final_sha",
    "hypothesis",
    "input_data",
    "physical_observables",
    "features",
    "model_structure",
    "training",
    "threshold",
    "attack_data_used",
    "gate",
    "terminal_result",
    "failure_cause",
    "failure_type",
    "texbat_repeat_risk",
    "substantive_duplicates",
    "evidence",
]

CRITICAL_RESULTS = {
    "CMTE-A2": "PRIMARY_INVALID_NO_GO",
    "CORA": "NO_GO_CORA_COMMON_ORIGIN_HYPOTHESIS",
    "CINDER": "NO_GO_CINDER_CLEAN_IDENTIFIABILITY",
    "CRID": "EXPLORATORY_TEXBAT_DS3_NO_USEFUL_SIGNAL",
    "MOSAIC": "NO_GO_MOSAIC_MULTI_PRN_RECOVERY",
    "BITPROBE": (
        "INCONCLUSIVE_BITPROBE_STAGE0A_R0A_INFERENCE_REPAIR; "
        "NO_ROBUST_SIGNAL_UNDER_RELAXED_GATES"
    ),
    "Q-SET": "RECEIVER_REPAIR_FAILED_CLEAN_REGRESSION",
}


class VerificationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot parse {path.name}: {exc}") from exc


def load_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise VerificationError(f"cannot parse {path.name}: {exc}") from exc


def inventory_csv_text(inventory: dict[str, Any]) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    for model in inventory["models"]:
        row = {}
        for field in CSV_FIELDS:
            value = model[field]
            if field == "attack_data_used":
                value = "true" if value else "false"
            elif field == "evidence":
                value = " | ".join(value)
            row[field] = value
        writer.writerow(row)
    return handle.getvalue()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_evidence_objects(
    repo_root: Path, models: list[dict[str, Any]], resolve_evidence: bool
) -> int:
    evidence_count = 0
    for model in models:
        require(
            isinstance(model["evidence"], list) and model["evidence"],
            f"{model['model']}: no evidence",
        )
        for item in model["evidence"]:
            evidence_count += 1
            match = re.fullmatch(r"([0-9a-f]{40}):(.+)", item)
            require(match is not None, f"{model['model']}: malformed evidence {item}")
            sha, object_path = match.groups()
            require(not object_path.startswith("/"), f"{model['model']}: absolute evidence")
            if resolve_evidence and "(local-only)" not in model["branch"]:
                result = subprocess.run(
                    ["git", "cat-file", "-e", f"{sha}:{object_path}"],
                    cwd=repo_root,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                require(
                    result.returncode == 0,
                    f"{model['model']}: unresolved Git evidence {item}",
                )
    return evidence_count


def verify_manifest(root: Path) -> int:
    manifest_path = root / "artifact_manifest_sha256.json"
    require(manifest_path.is_file(), "artifact_manifest_sha256.json missing")
    manifest = load_json(manifest_path)
    require(manifest["algorithm"] == "sha256", "manifest algorithm drift")
    entries = manifest["files"]
    require(isinstance(entries, dict) and entries, "empty manifest")

    excluded = {"artifact_manifest_sha256.json", "verifier_output.txt"}
    actual = {
        path.name
        for path in root.iterdir()
        if path.is_file() and path.name not in excluded
    }
    require(set(entries) == actual, "manifest file set mismatch")
    for name, expected in entries.items():
        path = root / name
        require(path.is_file(), f"manifest target missing: {name}")
        require(path.stat().st_size == expected["size_bytes"], f"size mismatch: {name}")
        require(sha256_file(path) == expected["sha256"], f"SHA-256 mismatch: {name}")
    return len(entries)


def verify(
    root: Path,
    *,
    check_manifest: bool = True,
    resolve_evidence: bool = False,
) -> dict[str, int]:
    root = root.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    missing = sorted(name for name in CORE_FILES if not (root / name).is_file())
    require(not missing, f"missing core files: {missing}")
    require(not (root / "candidate_3.md").exists(), "fabricated candidate_3.md")

    inventory = load_json(root / "prior_experiment_inventory.json")
    models = inventory["models"]
    require(len(models) == 33, "audited model count must be 33")
    require(inventory["scope"]["model_records"] == len(models), "model count drift")
    require(inventory["scope"]["remote_research_refs"] == 96, "research-ref count drift")
    names = [model["model"] for model in models]
    require(len(set(names)) == len(names), "duplicate inventory model")
    require(EXPECTED_MODELS <= set(names), "minimum detector families missing")
    for model in models:
        require(
            set(model) == set(CSV_FIELDS),
            f"{model['model']}: inventory fields inconsistent",
        )
        require(re.fullmatch(r"[0-9a-f]{40}", model["final_sha"]) is not None,
                f"{model['model']}: bad final SHA")
        for field in CSV_FIELDS:
            if field not in {"attack_data_used", "evidence"}:
                require(str(model[field]).strip() != "", f"{model['model']}: empty {field}")

    by_name = {model["model"]: model for model in models}
    for name, expected in CRITICAL_RESULTS.items():
        require(
            by_name[name]["terminal_result"] == expected,
            f"{name}: terminal verdict semantics changed",
        )
    require(
        by_name["CMTE-A2"]["failure_type"] == "IMPLEMENTATION_FAILURE",
        "CMTE invalid grouping must not be called a physical no-go",
    )
    require(
        "INCONCLUSIVE" in by_name["BITPROBE"]["terminal_result"],
        "BITPROBE formal inconclusive verdict was erased",
    )
    evidence_count = verify_evidence_objects(repo_root, models, resolve_evidence)
    require(evidence_count == 88, "evidence record count drift")
    local_only_evidence = sum(
        len(model["evidence"])
        for model in models
        if "(local-only)" in model["branch"]
    )
    require(local_only_evidence == 9, "local-only evidence count drift")

    expected_csv = inventory_csv_text(inventory)
    actual_csv = (root / "prior_experiment_inventory.csv").read_text(encoding="utf-8")
    require(actual_csv == expected_csv, "inventory JSON/CSV mismatch")
    require(len(load_csv(root / "prior_experiment_inventory.csv")) == 33,
            "inventory CSV row count drift")

    coverage_rows = load_csv(root / "observable_coverage_matrix.csv")
    require(len(coverage_rows) == 21, "observable count must be 21")
    require(
        {row["observable"] for row in coverage_rows} == EXPECTED_OBSERVABLES,
        "observable set mismatch",
    )
    require(
        all(row["status"] in ALLOWED_COVERAGE for row in coverage_rows),
        "invalid observable status",
    )
    require(
        all(row["principal_evidence"] and row["reason"] for row in coverage_rows),
        "observable evidence/reason missing",
    )

    scorecard = load_json(root / "candidate_scorecard.json")
    weights = scorecard["weights_percent"]
    require(sum(weights.values()) == 100, "candidate weights do not sum to 100")
    require(len(scorecard["candidates"]) == 2, "candidate count must be two")
    eligible_count = 0
    for candidate in scorecard["candidates"]:
        scores = candidate["scores"]
        require(set(scores) == set(weights), f"{candidate['id']}: score dimensions drift")
        require(all(isinstance(value, int) and 0 <= value <= 5 for value in scores.values()),
                f"{candidate['id']}: score outside 0..5")
        total = sum(weights[key] * scores[key] / 5.0 for key in weights)
        require(abs(total - candidate["weighted_total"]) < 1e-9,
                f"{candidate['id']}: weighted score arithmetic mismatch")
        dimensions_pass = (
            scores["physical_identifiability"] >= 4
            and scores["independence_from_prior_failures"] >= 4
            and scores["clean_only_falsifiability"] >= 4
        )
        total_pass = total >= 75
        require(candidate["mandatory_dimensions_pass"] is dimensions_pass,
                f"{candidate['id']}: mandatory pass mismatch")
        require(candidate["weighted_total_pass"] is total_pass,
                f"{candidate['id']}: total pass mismatch")
        require(candidate["eligible"] is (dimensions_pass and total_pass),
                f"{candidate['id']}: eligibility mismatch")
        eligible_count += int(candidate["eligible"])
    require(eligible_count == 0, "an eligible candidate contradicts terminal verdict")
    require(scorecard["selected_model"] is None, "scorecard selects a model")
    require(scorecard["selection_verdict"] == VERDICT, "scorecard verdict drift")

    final = load_json(root / "final_verdict.json")
    require(final["verdict"] == VERDICT, "final verdict drift")
    require(final["audit_status"] == "COMPLETE", "audit is not complete")
    require(final["selected_model"] is None, "final verdict selects a model")
    require(final["candidate_count"] == 2, "final candidate count drift")
    require(final["eligible_candidate_count"] == 0, "eligible count drift")
    require(final["audited_model_count"] == 33, "final model count drift")
    require(final["remote_research_refs_audited"] == 96, "final ref count drift")
    require(final["base_sha"] == BASE_SHA, "base SHA drift")
    require(final["implementation_or_training_performed"] is False,
            "implementation/training was claimed")
    require(final["attack_evaluation_performed"] is False,
            "attack evaluation was claimed")
    require(final["next_implementation_authorized"] is False,
            "implementation was authorized")
    access_values = final["current_audit_raw_access"].values()
    require(all(value == 0 for value in access_values), "nonzero final raw access")
    expected_tuni = {"SS-3", "SS-5", "SS-11", "SS-12", "SS-13"}
    require(set(final["tuni_preserved_unopened"]) == expected_tuni,
            "final Tuni preservation set drift")

    texbat = load_json(root / "texbat_data_usage_audit.json")
    require(all(value == 0 for value in texbat["current_audit_access"].values()
                if isinstance(value, int)), "nonzero current TEXBAT access")
    historical = texbat["historical_project_exposure"]
    for scenario in ("DS1", "DS2", "DS3", "DS4", "DS7", "DS8"):
        require("PREVIOUSLY_OPENED" in historical[scenario]["status"],
                f"{scenario}: historical exposure hidden")
    for scenario in ("DS5", "DS6"):
        require(historical[scenario]["status"] == "NO_COMMITTED_EVIDENCE_OF_PAYLOAD_USE",
                f"{scenario}: unsupported usage claim")

    tuni = load_json(root / "tuni_unopened_preservation.json")
    require(set(tuni["preserved_scenarios"]) == expected_tuni, "Tuni set drift")
    require(all(value == 0 for value in tuni["current_audit_operations"].values()),
            "nonzero Tuni attack access")
    require(tuni["future_contract"]["post_tuni_tuning"] is False,
            "post-Tuni tuning permitted")
    require("GPS L1 C/A" in tuni["future_contract"]["gps_to_galileo_warning"]
            and "Galileo E1" in tuni["future_contract"]["gps_to_galileo_warning"],
            "GPS/Galileo transfer warning missing")

    prereg = load_json(root / "preregistration_draft.json")
    require(prereg["status"] == "NOT_CREATED_NO_SELECTED_MODEL",
            "unexpected implementation preregistration")
    require(prereg["selected_model"] is None, "preregistration selects a model")
    require(prereg["implementation_authorized"] is False,
            "preregistration authorizes implementation")
    contract = prereg["task_contract_preserved"]
    require(contract["causal"] is True, "causal contract missing")
    for key in (
        "attack_labels_used_as_features",
        "prn_identity_used_as_feature",
        "scenario_or_filename_used_as_feature",
        "known_onset_used_as_feature",
        "post_decision_future_data",
    ):
        require(contract[key] is False, f"forbidden feature/causality drift: {key}")
    require(
        set(contract["primary_metrics"])
        == {
            "clean_false_alarm_rate",
            "first_detection_delay_after_official_onset",
            "attack_detection_probability",
            "recording-safe_block_bootstrap_confidence_interval",
        },
        "primary metric contract drift",
    )

    literature = load_json(root / "literature_sources.json")
    require(literature["cutoff_date"] == "2026-08-23", "literature cutoff drift")
    sources = literature["sources"]
    require(len(sources) >= 17, "literature source count too small")
    require(len({source["id"] for source in sources}) == len(sources),
            "duplicate literature source")
    for source in sources:
        require(source["url"].startswith("https://"), f"{source['id']}: non-HTTPS source")
        require(source["title"] and source["input"] and source["method"]
                and source["audit_relevance"], f"{source['id']}: incomplete source")
    candidate_one_dois = {
        source["doi"] for source in sources
        if "candidate_1" in source["closest_candidates"] and source["doi"]
    }
    require(
        {
            "10.1109/TST.2013.6678905",
            "10.33012/2017.15107",
            "10.33012/2022.18251",
            "10.3390/s26020397",
        }
        <= candidate_one_dois,
        "candidate 1 closest-prior DOI set incomplete",
    )
    require(
        literature["policy"]["novelty_rule"].startswith("Novelty is assessed from method"),
        "title-absence novelty rule missing",
    )

    all_docs = "\n".join(
        (root / name).read_text(encoding="utf-8")
        for name in sorted(CORE_FILES)
        if name.endswith(".md")
    )
    require("59/100" not in all_docs, "stale candidate score in documents")
    require(VERDICT in (root / "README.md").read_text(encoding="utf-8"),
            "README verdict missing")
    selected_spec = (root / "selected_model_spec.md").read_text(encoding="utf-8")
    require("No model is selected." in selected_spec, "selection contradiction")
    require("Candidate 1 scores 73/100" in selected_spec
            and "Candidate 2 scores 61/100" in selected_spec,
            "selected-model score contradiction")
    candidate_one_doc = re.sub(
        r"\s+", " ", (root / "candidate_1.md").read_text(encoding="utf-8")
    )
    candidate_two_doc = re.sub(
        r"\s+", " ", (root / "candidate_2.md").read_text(encoding="utf-8")
    )
    require("not selected" in candidate_one_doc,
            "candidate 1 selection ambiguity")
    require("not selected" in candidate_two_doc,
            "candidate 2 selection ambiguity")
    for token in (
        "e_clk[k]",
        "e_cc[i,k]",
        "trailing 10 s window",
        "FPR at most 0.01",
        "median first-alarm delay at most 10 s",
    ):
        require(token in candidate_one_doc, f"candidate 1 formula/gate drift: {token}")
    for token in (
        "q[i,w,t]",
        "trailing 30 s causal window",
        "q99 of source-distinct clean block maxima",
        "FPR at most 0.01",
        "median first-alarm delay at most 30 s",
    ):
        require(token in candidate_two_doc, f"candidate 2 formula/gate drift: {token}")

    manifest_count = verify_manifest(root) if check_manifest else 0
    return {
        "models": len(models),
        "evidence": evidence_count,
        "remote_evidence": evidence_count - local_only_evidence,
        "local_only_snapshot_evidence": local_only_evidence,
        "observables": len(coverage_rows),
        "literature_sources": len(sources),
        "candidates": len(scorecard["candidates"]),
        "eligible_candidates": eligible_count,
        "manifest_files": manifest_count,
    }


def write_derived(root: Path) -> None:
    inventory = load_json(root / "prior_experiment_inventory.json")
    (root / "prior_experiment_inventory.csv").write_text(
        inventory_csv_text(inventory), encoding="utf-8", newline=""
    )
    excluded = {"artifact_manifest_sha256.json", "verifier_output.txt"}
    entries = {}
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.name not in excluded:
            entries[path.name] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    manifest = {
        "schema": "gnss-doppler-lab.texbat-first-design-audit.manifest.v1",
        "algorithm": "sha256",
        "excluded_self_referential_files": sorted(excluded),
        "files": entries,
    }
    (root / "artifact_manifest_sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_RELATIVE_ROOT)
    parser.add_argument("--skip-manifest", action="store_true")
    parser.add_argument("--resolve-evidence", action="store_true")
    parser.add_argument("--write-derived", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    try:
        if args.write_derived:
            inventory = load_json(root / "prior_experiment_inventory.json")
            (root / "prior_experiment_inventory.csv").write_text(
                inventory_csv_text(inventory), encoding="utf-8", newline=""
            )
            verify(root, check_manifest=False, resolve_evidence=args.resolve_evidence)
            write_derived(root)
        summary = verify(
            root,
            check_manifest=not args.skip_manifest,
            resolve_evidence=args.resolve_evidence,
        )
    except (VerificationError, KeyError, TypeError, ValueError) as exc:
        print(f"status=FAIL\nreason={exc}")
        return 1

    print("status=PASS")
    print(f"verdict={VERDICT}")
    for key, value in summary.items():
        print(f"{key}={value}")
    print("raw_attack_bytes=0")
    print("next_implementation_authorized=false")
    return 0


if __name__ == "__main__":
    sys.exit(main())

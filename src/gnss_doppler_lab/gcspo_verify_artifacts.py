"""Recompute final GCSPO numerical evidence from whitelisted artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .gcspo_statistics import (RELATION_POLICY, exact_b0_full_contrast, exact_contrast_support, paired_block_bootstrap,
                               paired_score_loss_bootstrap, scenario_phase_balanced_pauc,
                               scheduled_persistence, validate_mandatory_relation_evidence)
from .gcspo_capabilities import validate_preaccess_capabilities


def _csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _score_rows(path):
    rows = []
    for raw in _csv(path):
        row = dict(raw)
        for key in ("window_start_s", "availability_s", "score", "threshold_q99",
                    "phase_start_s", "phase_end_s"):
            row[key] = float(row[key])
        row["effective_dof"] = None if row.get("effective_dof") in (None, "", "None") else float(row["effective_dof"])
        row["label"] = str(row["label"]).lower() == "true"
        row["prns"] = tuple(json.loads(row.pop("prns_json")))
        row["epoch_ids"] = tuple(json.loads(row.pop("epoch_ids_json")))
        row["epoch_prn_support"] = tuple((int(epoch), tuple(prns)) for epoch, prns in
                                          json.loads(row.pop("epoch_prn_support_json")))
        rows.append(row)
    return rows


def independently_load_clean_contrast_rows(root, identities):
    """Independently authenticate and reconstruct the cleanStatic negative cell."""
    root = Path(root).resolve(); required = ("clean_only_report.json", "clean_ablation_report.json",
                                             "clean_a5_report.json", "clean_reproduction_evidence.json")
    identity_map = {str(Path(row.get("path", "")).resolve()): row for row in identities if isinstance(row, dict)}
    documents, source = {}, {}
    for name in required:
        path = (root / name).resolve(); identity = identity_map.get(str(path))
        if not path.is_file() or identity is None:
            raise ValueError(f"clean contrast identity absent: {name}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest(); size = path.stat().st_size
        if digest != identity.get("sha256") or size != identity.get("size_bytes"):
            raise ValueError(f"clean contrast identity mismatch: {name}")
        documents[name] = json.loads(path.read_text())
        source[name] = {"sha256": digest, "size_bytes": size}
    clean, ablation, a5, reproduction = (documents[name] for name in required)
    expected_status = ((clean, "CLEAN_ONLY_PASS"), (ablation, "CLEAN_ABLATIONS_PASS"), (a5, "CLEAN_A5_PASS"))
    if any(doc.get("run_status") != status or doc.get("attack_access_count") != 0 or
           doc.get("protected_attack_rows_read") is not False for doc, status in expected_status):
        raise ValueError("clean contrast identity/status mismatch")
    try:
        inputs = {"Full": clean["scores"]["Full_holdout"], "A1": clean["scores"]["A1_holdout"],
                  "A2": ablation["methods"]["A2"]["holdout"], "A5": a5["holdout"]}
    except (KeyError, TypeError):
        raise ValueError("clean contrast cell is missing") from None
    expected_count = reproduction.get("counts", {}).get("clean_contrast_holdout_windows")
    if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count < 1:
        raise ValueError("clean contrast expected count binding is absent")
    if any(len(rows) != expected_count for rows in inputs.values()):
        raise ValueError("clean contrast cell count mismatch")
    source["clean_reproduction_evidence.json"]["expected_holdout_windows"] = expected_count
    result = []
    for method, rows in inputs.items():
        for raw in rows:
            try:
                prns = tuple(map(int, raw["prns"])); epochs = tuple(map(int, raw["epoch_ids"]))
                support = tuple((int(epoch), tuple(map(int, values))) for epoch, values in raw["epoch_prn_support"])
                score = float(raw["score"])
            except (KeyError, TypeError, ValueError):
                raise ValueError("clean contrast cell support is malformed") from None
            result.append({**raw, "prns": prns, "epoch_ids": epochs, "epoch_prn_support": support,
                           "score": score, "scenario": "cleanStatic", "family": "cleanStatic", "phase": "holdout",
                           "method": method, "label": False, "phase_start_s": 350., "phase_end_s": 470.})
    return result, source


def _paired_pauc(paired, method):
    rows = []
    for row in paired:
        family = "DS7_DS8" if row["scenario"] in {"DS7", "DS8"} else row["scenario"]
        label = "positive" if row["label"] else "negative"
        rows.append({**row, "score": row["scores"][method],
                     "logical_cell": f"{label}:{family}:{row['phase']}"})
    return scenario_phase_balanced_pauc(rows)


def reconstruct_relation_evidence(document, *, scenarios=None, capabilities=None):
    """Reconstruct the evaluator's scenario-specific paired score-loss policy."""
    relation_rows = document.get("rows")
    if not isinstance(relation_rows, list) or not relation_rows:
        raise ValueError("relation score rows are absent")
    selected = tuple(RELATION_POLICY) if scenarios is None else tuple(scenarios)
    unavailable = {} if capabilities is None else capabilities.get("unavailable", {})
    available = set(selected) if capabilities is None else set(capabilities.get("available", {}))
    required = sorted(available & {"DS3", "DS7", "DS8"})
    results = {"policy": RELATION_POLICY, "required_available_scenarios": required,
               "scenario_results": {}}
    for scenario in selected:
        policy = RELATION_POLICY[scenario]
        if scenario not in available:
            disposition = unavailable.get(scenario, {})
            results["scenario_results"][scenario] = {
                "status": disposition.get("status", "UNAVAILABLE"), "primary": policy["primary"],
                "mandatory": False, "reason": disposition.get("reason", "CAPABILITY_UNAVAILABLE")}
            continue
        rows = [row for row in relation_rows if row.get("scenario") == scenario]
        established = any(row.get("phase") == "established" for row in rows)
        if policy["requires_established"] and not established:
            results["scenario_results"][scenario] = {
                "status": "LIMITED_TRANSITION_ONLY", "primary": policy["primary"],
                "mandatory": False, "reason": "AUTHENTICATED_ESTABLISHED_PULL_OFF_COVERAGE_ABSENT"}
            continue
        try:
            report = paired_score_loss_bootstrap(rows, "Full", policy["primary"])
        except ValueError as exc:
            results["scenario_results"][scenario] = {
                "status": "UNAVAILABLE", "primary": policy["primary"],
                "mandatory": scenario in {"DS3", "DS7", "DS8"}, "reason": str(exc)}
            continue
        results["scenario_results"][scenario] = {
            "status": "AVAILABLE", "primary": policy["primary"],
            "mandatory": scenario in {"DS3", "DS7", "DS8"},
            "lcb": report["lcb_95"], "interval_95": report["interval_95"],
            "median_relative_loss": report["median_relative_loss"],
            "replicates": report["replicates"], "contrast": report["contrast"]}
    validate_mandatory_relation_evidence(results, required_scenarios=required)
    return results

def reconstruct_final_evidence(root):
    root = Path(root); scores = _score_rows(root / "per_epoch_scores.csv")
    freeze = json.loads((root / "implementation_manifest.json").read_text())
    clean_rows, clean_source = independently_load_clean_contrast_rows(root, freeze.get("clean_scientific_artifacts", []))
    source = clean_rows + [row for row in scores if row["scenario"] in {"DS3", "DS4", "DS7", "DS8"}
                           and row["phase"] != "pre_onset_replay"]
    incremental = {f"Full-{method}": paired_block_bootstrap(source, "Full", method)["lcb_95"]
                   for method in ("A1", "A2")}
    relation_document = json.loads((root / "relation_destruction_metrics.json").read_text())
    capabilities = validate_preaccess_capabilities(json.loads((root / "protected_capabilities.json").read_text()))
    destruction = reconstruct_relation_evidence(relation_document, capabilities=capabilities)
    persistence = {}; individual = {}
    for scenario in tuple(name for name in ("DS3", "DS7", "DS8") if name in capabilities["available"]):
        rows = sorted([row for row in scores if row["scenario"] == scenario and row["method"] == "Full"
                       and row["phase"] == "established"], key=lambda row: row["availability_s"])
        if not rows: raise ValueError(f"persistence rows absent: {scenario}")
        flags = scheduled_persistence(rows, threshold=rows[0]["threshold_q99"])
        first = next((row["availability_s"] for row, flag in zip(rows, flags) if flag), None)
        individual[scenario] = {"ratio": float(np.mean(flags)),
                                "delay_s": float("inf") if first is None else first - rows[0]["phase_start_s"]}
    family = [name for name in ("DS7", "DS8") if name in individual]
    if "DS3" not in individual or not family: raise ValueError("mandatory persistence capability is absent")
    persistence["DS3"] = individual["DS3"]
    persistence["DS7_DS8"] = {"ratio": min(individual[name]["ratio"] for name in family),
                               "delay_s": max(individual[name]["delay_s"] for name in family),
                               "available_members": family}
    external = {row["scenario"]: float(row["fpr_q99"]) for row in _csv(root / "external_static_fpr.csv")}
    clean_fpr = external.pop("cleanStatic_holdout")
    controls = json.loads((root / "physical_controls.json").read_text()).get("results")
    if not isinstance(controls, list) or not controls: raise ValueError("control evidence rows are absent")
    shared_support = exact_contrast_support(source, ("Full", "A5"))
    support_keys = {(row["scenario"], row["phase"], row["availability_s"], tuple(row["prns"])) for row in shared_support}
    full_edf = [row["effective_dof"] for row in source if row["method"] == "Full" and
                (row["scenario"], row["phase"], row["availability_s"], tuple(row["prns"])) in support_keys]
    a5_edf = [row["effective_dof"] for row in source if row["method"] == "A5" and
              (row["scenario"], row["phase"], row["availability_s"], tuple(row["prns"])) in support_keys]
    shared = {"full_pauc": _paired_pauc(shared_support, "Full"), "a5_pauc": _paired_pauc(shared_support, "A5"),
              "full_median_edf": float(np.median(full_edf)), "a5_median_edf": float(np.median(a5_edf))}
    b0_exact = exact_b0_full_contrast(scores, required_scenarios=sorted(set(capabilities["available"]) & {"DS3", "DS7", "DS8"}))
    return {"clean_holdout_fpr": clean_fpr, "clean_contrast_source": clean_source,
            "external_pre_fpr": external,
            "incremental_lcb": incremental, "b0_exact_support": b0_exact,
            "destruction": destruction, "persistence": persistence, "controls": controls, "shared": shared}

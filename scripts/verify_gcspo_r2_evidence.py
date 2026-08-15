#!/usr/bin/env python3
"""Independent verifier for the GCSPO Stage-0 R2 evidence bundle.

Inputs are copied evidence files under artifacts/gcspo_stage0_r2_runner_simulation_evidence.
The verifier does not import gcspo_r2_runner.py and does not rerun any model. It
recomputes metrics from CSV/JSON evidence and checks runner/source/provenance
records.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

DEFAULT_EVIDENCE_DIR = Path("artifacts/gcspo_stage0_r2_runner_simulation_evidence")
EXPECTED_SOURCE_MANIFEST_SHA256 = "ad6bbcd34c3889aa393d8699eec4e48c2dcc59095a5a0e3e632442b0bc7205cd"
REQUIRED_PHASES = [
    "preflight",
    "cleanstatic-normal-model",
    "ds3-evaluation",
    "ds7-evaluation",
    "ds4-ds8-conditional-evaluation",
    "relation-destruction-physical-controls",
    "final-statistics-plots-verification",
]
TOL = 1e-9


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def open_text(path: Path):
    return gzip.open(path, "rt", encoding="utf-8", newline="") if path.suffix == ".gz" else path.open("r", encoding="utf-8", newline="")


def read_json(base: Path, name: str):
    p = base / name
    if not p.exists() and (base / (name + ".gz")).exists():
        p = base / (name + ".gz")
    with open_text(p) as f:
        return json.load(f)


def read_csv_rows(base: Path, name: str) -> list[dict]:
    p = base / name
    if not p.exists() and (base / (name + ".gz")).exists():
        p = base / (name + ".gz")
    with open_text(p) as f:
        return list(csv.DictReader(f))


def as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() == "true"


def weighted_low_fpr_pauc(scores: Iterable[float], labels: Iterable[bool], cells: Iterable[str], *, alpha: float = 0.05) -> float:
    score = np.asarray(list(scores), dtype=np.float64)
    label = np.asarray(list(labels), dtype=bool)
    cell = np.asarray(list(cells)).astype(str)
    pos = sorted(set(cell[label]))
    neg = sorted(set(cell[~label]))
    if not pos or not neg:
        raise ValueError("weighted pAUC requires both classes")
    weights = np.zeros(len(score), dtype=np.float64)
    for positive, names in ((True, pos), (False, neg)):
        for name in names:
            idx = np.flatnonzero((label == positive) & (cell == name))
            weights[idx] = 1.0 / (len(names) * len(idx))
    order = np.argsort(-score, kind="stable")
    points = [(0.0, 0.0)]
    tp = fp = 0.0
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and score[order[j]] == score[order[i]]:
            j += 1
        group = order[i:j]
        tp += float(np.sum(weights[group][label[group]]))
        fp += float(np.sum(weights[group][~label[group]]))
        points.append((fp, tp))
        i = j
    clipped = [points[0]]
    for left, right in zip(points, points[1:]):
        if right[0] <= alpha:
            clipped.append(right)
            continue
        if left[0] < alpha:
            fraction = (alpha - left[0]) / (right[0] - left[0])
            clipped.append((alpha, left[1] + fraction * (right[1] - left[1])))
        break
    if clipped[-1][0] < alpha:
        clipped.append((alpha, clipped[-1][1]))
    x = np.asarray([p[0] for p in clipped], dtype=np.float64)
    y = np.asarray([p[1] for p in clipped], dtype=np.float64)
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) * 0.5, dtype=np.float64) / alpha)


def scheduled_persistence(rows: list[dict], *, threshold: float, slot_s: float = 0.5) -> list[bool]:
    ordered = sorted(rows, key=lambda r: float(r["availability_s"]))
    out, run, prev = [], [], None
    for row in ordered:
        cur = float(row["availability_s"])
        if prev is None or not math.isclose(cur - prev, slot_s, rel_tol=0.0, abs_tol=1e-12):
            run = []
        run.append(float(row["score"]) > threshold)
        if len(run) > 5:
            run.pop(0)
        out.append(len(run) == 5 and sum(run) >= 3)
        prev = cur
    return out


def support_identity(row: dict):
    return (
        row["scenario"], row["phase"], float(row["window_start_s"]), float(row["availability_s"]),
        tuple(json.loads(row["epoch_ids_json"])),
        tuple(json.loads(row["prns_json"])),
        tuple((int(e), tuple(p)) for e, p in json.loads(row["epoch_prn_support_json"])),
    )


def assert_close(checks: list[dict], name: str, actual, expected, tol: float = TOL):
    ok = actual == expected if isinstance(expected, str) else abs(float(actual) - float(expected)) <= tol
    checks.append({"name": name, "actual": actual, "expected": expected, "tolerance": tol if not isinstance(expected, str) else None, "status": "PASS" if ok else "FAIL"})
    return ok


def verify(base: Path = DEFAULT_EVIDENCE_DIR) -> dict:
    checks: list[dict] = []
    failures: list[str] = []

    manifest = read_json(base, "evidence_bundle_manifest.json")
    source_integrity = read_json(base, "source_artifact_integrity.json")
    if not assert_close(checks, "source manifest sha256", source_integrity["actual_manifest_sha256"], EXPECTED_SOURCE_MANIFEST_SHA256, 0):
        failures.append("SOURCE_ARTIFACT_CHANGED")
    if not assert_close(checks, "source file count", source_integrity["actual_file_count"], 68, 0):
        failures.append("SOURCE_ARTIFACT_CHANGED")
    if not assert_close(checks, "source integrity status", source_integrity["status"], "PASS"):
        failures.append("SOURCE_ARTIFACT_CHANGED")

    thresholds = read_json(base, "thresholds.json")
    clean = read_json(base, "clean_only_report.json")
    b0 = read_json(base, "clean_b0_report.json")
    full_holdout = clean["scores"]["Full_holdout"]
    full_q99 = float(thresholds["Full"]["q99"])
    full_q995 = float(thresholds["Full"]["q995"])
    b0_q99 = float(thresholds["A0_B0"]["q99"])
    full_fpr_q99 = float(np.mean([float(r["score"]) > full_q99 for r in full_holdout]))
    full_fpr_q995 = float(np.mean([float(r["score"]) > full_q995 for r in full_holdout]))
    b0_holdout = b0["scores"]["holdout"]
    b0_fpr_q99 = float(np.mean([float(r["btail_max_507080_ewma075"]) > b0_q99 for r in b0_holdout]))
    assert_close(checks, "cleanStatic Full q99 holdout FPR", full_fpr_q99, 0.07172995780590717)
    assert_close(checks, "cleanStatic Full q99.5 holdout FPR", full_fpr_q995, 0.06751054852320675)
    assert_close(checks, "cleanStatic B0 q99 holdout FPR", b0_fpr_q99, 0.12745098039215685)
    assert_close(checks, "Full q99 calibration threshold", full_q99, -1942.747871032354)
    assert_close(checks, "Full q99.5 calibration threshold", full_q995, -1926.8027680519062)
    assert_close(checks, "B0 q99 calibration threshold", b0_q99, 1.5536241677134872)
    assert_close(checks, "Full holdout row count", len(full_holdout), 237, 0)
    assert_close(checks, "B0 holdout row count", len(b0_holdout), 204, 0)

    rows = read_csv_rows(base, "per_epoch_scores.csv")
    by = defaultdict(list)
    for r in rows:
        by[(r["scenario"], r["method"])].append(r)
    scenario_metrics = {(r["scenario"], r["method"]): r for r in read_csv_rows(base, "scenario_metrics.csv")}
    recomputed_metrics = {}
    for scenario in ["DS3", "DS7"]:
        selected = by[(scenario, "Full")]
        pre = [r for r in selected if r["phase"].startswith("pre_")]
        transition = [r for r in selected if r["phase"] == "transition"]
        established = [r for r in selected if r["phase"] == "established"]
        attack = transition + established
        scores = [float(r["score"]) for r in pre + attack]
        labels = [False] * len(pre) + [True] * len(attack)
        cells = ["negative"] * len(pre) + ["positive"] * len(attack)
        q99 = float(selected[0]["threshold_q99"])
        ordered = sorted(selected, key=lambda r: float(r["availability_s"]))
        flags = scheduled_persistence(ordered, threshold=q99)
        onset, pull = {"DS3": (118.9, 195.0), "DS7": (110.0, 150.0)}[scenario]
        first_alarm = next((float(r["availability_s"]) for r in attack if float(r["score"]) > q99), None)
        first_persistent = next((float(r["availability_s"]) for r, f in zip(ordered, flags) if f and r in attack), None)
        metric = {
            "full_events": len(selected),
            "pre_onset_events": len(pre),
            "attack_events": len(attack),
            "pre_onset_fpr_q99": float(np.mean([float(r["score"]) > q99 for r in pre])),
            "roc_auc": float(roc_auc_score(labels, scores)),
            "low_fpr_pauc": weighted_low_fpr_pauc(scores, labels, cells, alpha=0.05),
            "pr_auc": float(average_precision_score(labels, scores)),
            "attack_detection_rate_q99": float(np.mean([float(r["score"]) > q99 for r in attack])),
            "transition_detection_rate_q99": float(np.mean([float(r["score"]) > q99 for r in transition])),
            "established_detection_rate_q99": float(np.mean([float(r["score"]) > q99 for r in established])),
            "first_alarm_delay_from_onset_s": None if first_alarm is None else first_alarm - onset,
            "first_alarm_delay_from_pull_off_s": None if first_alarm is None else first_alarm - pull,
            "first_persistent_alarm_s": first_persistent,
            "persistent_alarm_ratio": float(np.mean(flags)),
        }
        recomputed_metrics[scenario] = metric
        reported = scenario_metrics[(scenario, "Full")]
        for key in ["roc_auc", "low_fpr_pauc", "pr_auc", "pre_onset_fpr_q99", "attack_detection_rate_q99", "transition_detection_rate_q99", "established_detection_rate_q99", "first_alarm_delay_from_onset_s", "first_alarm_delay_from_pull_off_s", "first_persistent_alarm_s", "persistent_alarm_ratio"]:
            assert_close(checks, f"{scenario} Full {key}", metric[key], float(reported[key]))
        assert_close(checks, f"{scenario} Full events", metric["full_events"], int(reported["windows"]), 0)

    assert_close(checks, "DS3 Full events expected", recomputed_metrics["DS3"]["full_events"], 907, 0)
    assert_close(checks, "DS3 Full pre-onset FPR expected", recomputed_metrics["DS3"]["pre_onset_fpr_q99"], 0.14893617021276595)
    assert_close(checks, "DS3 Full ROC-AUC expected", recomputed_metrics["DS3"]["roc_auc"], 0.8267983789260385)
    assert_close(checks, "DS7 Full events expected", recomputed_metrics["DS7"]["full_events"], 533, 0)
    assert_close(checks, "DS7 Full ROC-AUC expected", recomputed_metrics["DS7"]["roc_auc"], 0.5991366738610512)
    external = {r["scenario"]: r for r in read_csv_rows(base, "external_static_fpr.csv")}
    assert_close(checks, "DS7 pre-110 replay exclusion", external["DS7_pre_onset"]["status"], "REPLAY_EXCLUDED_FROM_EXTERNAL_FPR")

    # Exact support accounting and complete identity check.
    identities = {("DS3", "Full"): set(), ("DS3", "A0"): set(), ("DS7", "Full"): set(), ("DS7", "A0"): set()}
    malformed = 0
    for r in rows:
        key = (r["scenario"], r["method"])
        if key not in identities:
            continue
        try:
            ident = support_identity(r)
            epochs = ident[4]
            support = ident[6]
            prns = ident[5]
            if tuple(e for e, _ in support) != epochs:
                malformed += 1
            union = tuple(sorted(set().union(*(set(p) for _, p in support)))) if support else ()
            if union != prns:
                malformed += 1
            identities[key].add(ident)
        except Exception:
            malformed += 1
    assert_close(checks, "complete epoch+PRN+window identity malformed rows", malformed, 0, 0)
    for scenario, expected in [("DS3", 643), ("DS7", 396)]:
        full, b0s = identities[(scenario, "Full")], identities[(scenario, "A0")]
        common = full & b0s
        assert_close(checks, f"{scenario} B0 exact-common", len(common), expected, 0)
        inv = {r["scenario"]: r for r in read_csv_rows(base, f"phase_outputs/{scenario.lower()}-evaluation/common_support_inventory.csv")}
        # phase_outputs common_support_inventory has method rows only; top-level has both scenarios.
    common_rows = {(r["scenario"], r["method"]): r for r in read_csv_rows(base, "common_support_inventory.csv")}
    for scenario in ["DS3", "DS7"]:
        row = common_rows[(scenario, "A0")]
        full, b0s = identities[(scenario, "Full")], identities[(scenario, "A0")]
        assert_close(checks, f"{scenario} A0 method_total", len(b0s), int(row["method_total"]), 0)
        assert_close(checks, f"{scenario} Full total", len(full), int(row["full_total"]), 0)
        assert_close(checks, f"{scenario} A0 method_only", len(b0s - full), int(row["method_only"]), 0)
        assert_close(checks, f"{scenario} A0 full_only", len(full - b0s), int(row["full_only"]), 0)

    relation = read_json(base, "relation_destruction_metrics.json")
    controls = read_json(base, "physical_controls_audit.json")
    assert_close(checks, "DS3 LOS shuffle status", relation["results"]["DS3"]["LOS_SHUFFLE"]["status"], "AVAILABLE")
    assert_close(checks, "DS3 temporal desync LCB", relation["results"]["DS3"]["PER_PRN_TEMPORAL_SHIFT"]["lcb_95"], -0.013567801537405258)
    aggregate = {r["id"]: r for r in controls["aggregate"]}
    for cid, blocks in [("EMPIRICAL_NOISE", 18), ("ONE_PRN_DISTURBANCE", 3), ("PRN_DROP_ONLY", 12)]:
        assert_close(checks, f"{cid} blocks_with_persistent_alarm", aggregate[cid]["blocks_with_persistent_alarm"], blocks, 0)
        assert_close(checks, f"{cid} max_consecutive_alarms", aggregate[cid]["max_consecutive_alarms"], 17, 0)
        assert_close(checks, f"{cid} max_persistent_alarm_ratio", aggregate[cid]["max_persistent_alarm_ratio"], 0.7647058823529411)
    assert_close(checks, "physical controls false alarm status", controls["scientific_false_alarm_status"], "FAIL")

    runner = read_json(base, "runner_phase_evidence.json")
    assert_close(checks, "runner all required latest successful", runner["all_required_latest_successful"], True)
    for phase in REQUIRED_PHASES:
        info = runner["required_phases"][phase]
        assert_close(checks, f"runner {phase} status", info["status"], "succeeded")
        assert_close(checks, f"runner {phase} exit_code", info["exit_code"], 0, 0)
        assert_close(checks, f"runner {phase} heartbeat", info["final_heartbeat_exists"], True)
        assert_close(checks, f"runner {phase} command", info["contract_command_exists"], True)
        assert_close(checks, f"runner {phase} terminal evidence", info["phase_terminal_evidence_ok"], True)

    provenance = read_json(base, "provenance_audit.json")
    assert_close(checks, "provenance metric effect", provenance["metric_effect"], "none identified")
    assert_close(checks, "provenance claim", provenance["provenance_claim"], "weakened")
    assert_close(checks, "live remote sync", provenance["live_remote_sync"], "not independently demonstrated at execution time")

    failed = [c for c in checks if c["status"] != "PASS"]
    if failed and "SOURCE_ARTIFACT_CHANGED" in failures:
        judgement = "SOURCE_ARTIFACT_CHANGED"
    elif failed:
        judgement = "EVIDENCE_RECOMPUTATION_MISMATCH"
    elif manifest.get("final_evidence_judgement") != "EVIDENCE_VERIFIED":
        judgement = "EVIDENCE_INCOMPLETE"
    else:
        judgement = "EVIDENCE_VERIFIED"
    return {
        "schema": "gnss-doppler-lab.gcspo-stage0-r2.independent-evidence-verification.v1",
        "evidence_dir": str(base),
        "final_evidence_judgement": judgement,
        "checks_total": len(checks),
        "checks_failed": len(failed),
        "failed_checks": failed,
        "checks": checks,
        "recomputed_metrics": recomputed_metrics,
        "science_verdict_preserved": {
            "detector": "NO-GO under current configuration",
            "neural_stage1": "not allowed",
            "shared_pull_off_physics": "incomplete",
            "paper_model_continuation": "not recommended",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", default=str(DEFAULT_EVIDENCE_DIR))
    parser.add_argument("--write-json", default=None)
    args = parser.parse_args()
    result = verify(Path(args.evidence_dir))
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.write_json:
        Path(args.write_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.write_json).write_text(text, encoding="utf-8")
    print(text)
    return 0 if result["final_evidence_judgement"] == "EVIDENCE_VERIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())

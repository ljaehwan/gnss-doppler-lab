#!/usr/bin/env python3
"""Fresh-clone verifier for committed CRISP Stage-0 artifacts."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
ARTIFACT = REPO_ROOT / "artifacts/crisp_stage0_static"
REQUIRED = {
    "README.md", "config.json", "preregistration.json", "source_commit.json",
    "data_inventory.json", "source_lineage.json", "clean_split_audit.json",
    "invariance_tests.json", "normal_model_summary.json", "thresholds.json",
    "scenario_metrics.csv", "ablation_metrics.csv", "per_epoch_scores.csv.gz",
    "per_block_scores.csv.gz", "per_prn_metrics.csv", "external_static_fpr.csv",
    "control_metrics.json", "bootstrap_intervals.csv", "final_verdict.json",
    "artifact_manifest_sha256.json", "runtime_summary.json",
    "reproduction_validation.json", "supplemental_metrics.json",
    "oak_os3_os4_pooled_metrics.json",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def main() -> None:
    missing = sorted(name for name in REQUIRED if not (ARTIFACT / name).is_file())
    if missing:
        fail(f"missing required files: {missing}")
    manifest = json.loads((ARTIFACT / "artifact_manifest_sha256.json").read_text())
    paths = {row["path"] for row in manifest["files"]}
    for row in manifest["files"]:
        path = ARTIFACT / row["path"]
        if not path.is_file() or path.stat().st_size != row["size_bytes"] or sha256(path) != row["sha256"]:
            fail(f"manifest mismatch: {row['path']}")
    if not (REQUIRED - {"artifact_manifest_sha256.json"}) <= paths:
        fail("manifest does not cover required files")
    prereg = json.loads((ARTIFACT / "preregistration.json").read_text())
    if prereg["status"] != "CLEAN_ONLY_PREREGISTRATION" or prereg["attack_scores_viewed"]:
        fail("clean-only preregistration seal invalid")
    audit = json.loads((ARTIFACT / "clean_split_audit.json").read_text())
    if audit["attack_rows_read_for_model_or_threshold"] != 0 or audit["raw_sample_overlap"] or not audit["chronological"]:
        fail("clean split/leakage audit failed")
    inventory = json.loads((ARTIFACT / "data_inventory.json").read_text())
    if inventory["status"] != "PASS" or any(row["status"] != "PASS" for row in inventory["scenarios"].values()):
        fail("input provenance unavailable")
    invariance = json.loads((ARTIFACT / "invariance_tests.json").read_text())
    if not invariance["all_pass"] or not invariance["projector_properties"]["pass"]:
        fail("algebraic invariance failed")
    reproduction = json.loads((ARTIFACT / "reproduction_validation.json").read_text())
    if not reproduction["pass"] or not reproduction["final_report_renderer"]["pass"]:
        fail("deterministic reproduction failed")
    supplemental = json.loads((ARTIFACT / "supplemental_metrics.json").read_text())
    if set(supplemental) != {"TEXBAT.DS3", "TEXBAT.DS7", "OAKBAT.OS3", "OAKBAT.OS4"}:
        fail("supplemental scenario set mismatch")
    pooled = json.loads((ARTIFACT / "oak_os3_os4_pooled_metrics.json").read_text())
    if "Full" not in pooled or not 0.0 <= pooled["Full"]["pauc_fpr_le_0_05"] <= 1.0:
        fail("OAK pooled metrics invalid")
    thresholds = json.loads((ARTIFACT / "thresholds.json").read_text())
    for dataset in ("TEXBAT", "OAKBAT"):
        for method, row in thresholds[dataset]["methods"].items():
            if "cleanStatic calibration only" not in row["source"] or row["calibration_blocks"] <= 0:
                fail(f"threshold leakage/empty calibration: {dataset}/{method}")
    with (ARTIFACT / "scenario_metrics.csv").open(newline="") as stream:
        scenario_rows = list(csv.DictReader(stream))
    if {row["scenario"] for row in scenario_rows} != {
        "TEXBAT.cleanStatic", "TEXBAT.DS3", "TEXBAT.DS7",
        "OAKBAT.cleanStatic", "OAKBAT.OS3", "OAKBAT.OS4",
    }:
        fail("scenario set mismatch")
    with gzip.open(ARTIFACT / "per_epoch_scores.csv.gz", "rt", newline="") as stream:
        epoch_reader = csv.DictReader(stream)
        first = next(epoch_reader, None)
    with gzip.open(ARTIFACT / "per_block_scores.csv.gz", "rt", newline="") as stream:
        block_reader = csv.DictReader(stream)
        first_block = next(block_reader, None)
    if first is None or first_block is None:
        fail("empty score artifact")
    verdict = json.loads((ARTIFACT / "final_verdict.json").read_text())
    allowed = {"GO_FOR_CRISP_NEURAL_STAGE1", "NO_GO_CRISP_PHYSICAL_HYPOTHESIS", "INCONCLUSIVE_INPUT_PROVENANCE"}
    if verdict["verdict"] not in allowed:
        fail("invalid verdict")
    recomputed = "GO_FOR_CRISP_NEURAL_STAGE1" if all(verdict["go_checks"].values()) else "NO_GO_CRISP_PHYSICAL_HYPOTHESIS"
    if verdict["verdict"] != recomputed:
        fail(f"verdict mismatch: expected {recomputed}")
    prereg_sha = verdict["preregistration_sha"]
    try:
        subprocess.run(["git", "merge-base", "--is-ancestor", prereg_sha, "HEAD"], check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        fail("preregistration SHA is not an ancestor of result")
    plots = sorted((ARTIFACT / "plots").glob("*.png"))
    if len(plots) != 10 or any(path.stat().st_size == 0 for path in plots):
        fail("required plots missing")
    print(f"PASS: CRISP artifact {verdict['verdict']}; {len(manifest['files'])} checksums")


if __name__ == "__main__":
    main()

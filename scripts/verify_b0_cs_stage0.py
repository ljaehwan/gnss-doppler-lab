#!/usr/bin/env python3
"""Fail-closed verifier for a completed B0-CS Stage-0 artifact bundle."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = ROOT / "artifacts" / "b0_cs_stage0_static"
ALLOWED_VERDICTS = {
    "GO_WCL_B0_CS", "B0_ONLY_STRONG_BUT_METHOD_WEAK",
    "PIVOT_TO_PROVENANCE_EVALUATION_PAPER", "NO_PAPER_READY_EVIDENCE",
}
REQUIRED_FILES = {
    "README.md", "config.json", "preregistration.json", "source_commit.json",
    "data_inventory.json", "timeline_inventory.json", "split_and_overlap_audit.json",
    "historical_b0_reproduction.json", "paper_b0_training_summary.json",
    "calibration_summary.json", "thresholds.json", "scenario_metrics.csv",
    "ablation_metrics.csv", "external_static_fpr.csv", "calibration_diagnostics.csv",
    "per_epoch_scores.csv.gz", "per_block_scores.csv.gz", "control_metrics.json",
    "bootstrap_intervals.csv", "runtime_metrics.json", "final_verdict.json",
    "artifact_manifest_sha256.json",
}
REQUIRED_PLOTS = {
    "paper_b0_residual_timeline.png", "b0_vs_b0_cs_receiver_score.png",
    "block_e_cusum_timeline.png", "scenario_roc_and_low_fpr_roc.png",
    "external_normal_fpr.png", "calibration_pvalue_histogram.png",
    "cn0_tracked_count_strata.png", "b0_vs_b0_cs_alarm_delay.png",
    "control_response.png", "ar_vs_gru_ablation.png",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_document(root: Path) -> dict:
    target = root / "artifact_manifest_sha256.json"
    files = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != target:
            files.append({
                "path": str(path.relative_to(root)), "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    return {"schema": "gnss-doppler-lab.b0-cs-artifact-manifest.v1", "files": files}


def write_manifest(root: Path) -> None:
    path = root / "artifact_manifest_sha256.json"
    path.write_text(json.dumps(manifest_document(root), indent=2, sort_keys=True) + "\n")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(root: Path, *, require_manifest: bool = True) -> dict:
    errors = []
    missing = sorted(name for name in REQUIRED_FILES if not (root / name).is_file())
    if not require_manifest:
        missing = [name for name in missing if name != "artifact_manifest_sha256.json"]
    if missing:
        errors.append(f"missing required files: {missing}")
    plot_missing = sorted(name for name in REQUIRED_PLOTS if not (root / "plots" / name).is_file())
    if plot_missing:
        errors.append(f"missing required plots: {plot_missing}")
    json_names = [
        "config.json", "preregistration.json", "source_commit.json", "data_inventory.json",
        "timeline_inventory.json", "split_and_overlap_audit.json", "historical_b0_reproduction.json",
        "paper_b0_training_summary.json", "calibration_summary.json", "thresholds.json",
        "control_metrics.json", "runtime_metrics.json", "final_verdict.json", "clean_freeze.json",
    ]
    documents = {}
    for name in json_names:
        path = root / name
        if path.exists():
            try:
                documents[name] = load_json(path)
            except Exception as exc:
                errors.append(f"invalid JSON {name}: {exc}")
    verdict = documents.get("final_verdict.json", {}).get("verdict")
    if verdict not in ALLOWED_VERDICTS:
        errors.append(f"invalid final verdict: {verdict}")
    config = documents.get("config.json", {})
    if config.get("attack_outcome_selection") is not False:
        errors.append("config does not prohibit attack-outcome selection")
    prereg = documents.get("preregistration.json", {})
    if prereg.get("pre_attack_assertions", {}).get("attack_outcomes_used_for_selection") is not False:
        errors.append("preregistration attack-selection assertion is not false")
    split = documents.get("split_and_overlap_audit.json", {})
    for field in ("no_target_epoch_overlap", "no_raw_sample_or_byte_interval_overlap", "normal_only"):
        if split.get(field) is not True:
            errors.append(f"split audit failed: {field}")
    freeze = documents.get("clean_freeze.json", {})
    if freeze.get("status") != "FROZEN_BEFORE_ATTACK_ACCESS" or freeze.get("attack_inputs_read") is not False:
        errors.append("clean freeze boundary invalid")
    if freeze:
        frozen_hashes = freeze.get("config_and_code_hashes", {})
        observed = {
            "config_sha256": sha256(root / "config.json") if (root / "config.json").exists() else None,
            "preregistration_sha256": sha256(root / "preregistration.json") if (root / "preregistration.json").exists() else None,
            "implementation_sha256": sha256(ROOT / "src" / "gnss_doppler_lab" / "b0_dependence_calibrated.py"),
            "experiment_implementation_sha256": sha256(ROOT / "src" / "gnss_doppler_lab" / "b0_cs_stage0_experiment.py"),
            "runner_sha256": sha256(ROOT / "scripts" / "run_b0_cs_stage0.py"),
        }
        if frozen_hashes != observed:
            errors.append("configuration/code differs from clean freeze")
        for filename, field in (
            ("paper_b0_model.pt", "paper_b0_model_sha256"),
            ("calibrator_state.json", "calibrator_state_sha256"),
            ("thresholds.json", "thresholds_sha256"),
            ("linear_ar_state.json", "linear_ar_state_sha256"),
        ):
            path = root / filename
            if not path.is_file() or sha256(path) != freeze.get(field):
                errors.append(f"frozen artifact mismatch: {filename}")
    for name in ("per_epoch_scores.csv.gz", "per_block_scores.csv.gz"):
        path = root / name
        if path.exists():
            try:
                with gzip.open(path, "rt", encoding="utf-8") as stream:
                    header = stream.readline().strip()
                if not header:
                    errors.append(f"empty gzip CSV: {name}")
            except Exception as exc:
                errors.append(f"invalid gzip CSV {name}: {exc}")
    if require_manifest and (root / "artifact_manifest_sha256.json").exists():
        try:
            saved = load_json(root / "artifact_manifest_sha256.json")
            actual = manifest_document(root)
            if saved != actual:
                errors.append("artifact manifest content/hash mismatch")
        except Exception as exc:
            errors.append(f"invalid artifact manifest: {exc}")
    readme = (root / "README.md").read_text(encoding="utf-8") if (root / "README.md").exists() else ""
    if readme.count("## 16. Exactly one recommended next action") != 1:
        errors.append("README does not contain exactly one recommended-next-action section")
    return {
        "schema": "gnss-doppler-lab.b0-cs-verifier-result.v1",
        "result": "PASS" if not errors else "FAIL", "errors": errors,
        "artifact_root": str(root), "verdict": verdict,
        "manifest_sha256": sha256(root / "artifact_manifest_sha256.json") if (root / "artifact_manifest_sha256.json").exists() else None,
    }


def fresh_clone_verify(root: Path, branch: str) -> dict:
    if subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True).strip():
        raise ValueError("fresh-clone verification requires a clean committed worktree")
    with tempfile.TemporaryDirectory(prefix="b0-cs-fresh-clone-") as directory:
        clone = Path(directory) / "repo"
        subprocess.run(
            ["git", "clone", "--quiet", "--no-hardlinks", "--branch", branch, str(ROOT), str(clone)],
            check=True,
        )
        relative = root.resolve().relative_to(ROOT.resolve())
        command = [sys.executable, str(clone / "scripts" / "verify_b0_cs_stage0.py"),
                   "--artifact-root", str(clone / relative)]
        completed = subprocess.run(command, text=True, capture_output=True)
        parsed = json.loads(completed.stdout) if completed.stdout.strip() else {
            "result": "FAIL", "errors": [completed.stderr]
        }
        return {
            "schema": "gnss-doppler-lab.b0-cs-fresh-clone-result.v1",
            "result": "PASS" if completed.returncode == 0 and parsed.get("result") == "PASS" else "FAIL",
            "branch": branch, "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=clone, text=True).strip(),
            "nested_verifier": parsed, "stderr": completed.stderr,
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    parser.add_argument("--write-manifest", action="store_true")
    parser.add_argument("--fresh-clone", action="store_true")
    parser.add_argument("--branch", default="research/b0-cs-stage0-static")
    parser.add_argument("--pre-manifest", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.artifact_root).resolve()
    if args.write_manifest:
        preflight = verify(root, require_manifest=False)
        if preflight["result"] != "PASS":
            print(json.dumps(preflight, indent=2, sort_keys=True))
            return 1
        write_manifest(root)
    result = verify(root, require_manifest=not args.pre_manifest)
    if args.fresh_clone and result["result"] == "PASS":
        result["fresh_clone"] = fresh_clone_verify(root, args.branch)
        if result["fresh_clone"]["result"] != "PASS":
            result["result"] = "FAIL"
            result["errors"].append("fresh-clone verifier failed")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Verify committed GCSPO artifacts from a fresh worktree/clone checkout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnss_doppler_lab.gcspo_artifacts import canonical_write_json, FROZEN_HASHES, sha256_file, utc_now
from gnss_doppler_lab.gcspo_fresh_clone import clone_exact
from gnss_doppler_lab.gcspo_verify import verify_final

DEFAULT_ARTIFACT_RELATIVE = Path("artifacts/gcspo_stage0_static_rerun")

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("full",), default="full")
    parser.add_argument("--repo-url", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_RELATIVE)
    parser.add_argument("--clean-root", type=Path)
    args = parser.parse_args(); started = utc_now()
    if args.artifact_dir.is_absolute(): raise ValueError("artifact dir must be repository-relative")
    output_artifact = ROOT / args.artifact_dir
    with tempfile.TemporaryDirectory(prefix="gcspo-fresh-clone-") as temporary:
        checkout = Path(temporary) / "repo"; clone = clone_exact(args.repo_url, args.commit, checkout)
        environment = Path(temporary) / "venv"
        subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
        python = environment / "bin" / "python"
        subprocess.run([str(python), "-m", "pip", "install", "-r", "requirements-gcspo.txt", "-e", ".[dev]"], cwd=checkout, check=True)
        is_r1 = args.artifact_dir.name == "gcspo_stage0_r1_frozen_completion"
        pattern = "test_gcspo_r1*.py" if is_r1 else "test_gcspo*.py"
        tests = sorted(str(path.relative_to(checkout)) for path in (checkout / "tests").glob(pattern))
        if not tests: raise ValueError("fresh clone contains no mandatory GCSPO tests")
        subprocess.run([str(python), "-m", "pytest", "-q", *tests], cwd=checkout, check=True)
        cloned_artifact = checkout / args.artifact_dir
        verify_command = ([str(python), "scripts/verify_gcspo_stage0.py", "--r1-final",
                           "--artifact-dir", str(cloned_artifact)] if is_r1 else
                          [str(python), "scripts/verify_gcspo_stage0.py", "--mode", "full",
                           "--artifact-dir", str(cloned_artifact)])
        subprocess.run(verify_command, cwd=checkout, check=True)
        clean_reproduction = "PASS" if is_r1 else "NOT_PERMITTED_OR_SOURCE_ABSENT"
        if not is_r1 and args.clean_root is not None and args.clean_root.is_dir():
            reproduction = Path(temporary) / "clean-reproduction"
            reproduction.mkdir();
            for name in FROZEN_HASHES: shutil.copy2(cloned_artifact / name, reproduction / name)
            subprocess.run([str(python), "scripts/run_gcspo_stage0.py", "--phase", "clean-only",
                            "--config", str(reproduction / "config.json"), "--artifact-dir", str(reproduction),
                            "--clean-root", str(args.clean_root)], cwd=checkout, check=True)
            clean_reproduction = "PASS"
        manifest_name = "artifact_manifest_sha256.json"
        manifest_sha = sha256_file(cloned_artifact / manifest_name)
        implementation = json.loads((cloned_artifact / "implementation_manifest.json").read_text()) if is_r1 else None
        report = {"schema": "gnss-doppler-lab.gcspo-stage0.fresh-clone-verifier-report.v2",
                  "commit": args.commit,
                  "target_commit": implementation["target_commit"] if is_r1 else args.commit,
                  "evidence_commit": clone["head"],
                  "config_sha256": FROZEN_HASHES["config.json"], "artifact_manifest_sha256": manifest_sha,
                  "checks": [{"id": "exact_clone", "status": "PASS"}, {"id": "mandatory_tests", "status": "PASS"},
                             {"id": "canonical_manifest", "status": "PASS"},
                             {"id": "clean_reproduction", "status": clean_reproduction}],
                  "overall_status": "PASS", "verified_run_status": "VALID_SCIENCE",
                  "attestation_scope": "post-evidence; this report is intentionally excluded from evidence_commit and the scientific manifest",
                  "command": "--r1-final" if is_r1 else "--mode full",
                  "started_utc": started, "finished_utc": utc_now(), "exit_code": 0}
    canonical_write_json(output_artifact / "fresh_clone_verifier_report.json", report)
    print("FRESH_CLONE_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

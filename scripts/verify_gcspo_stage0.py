#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import subprocess

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gnss_doppler_lab.gcspo_artifacts import canonical_write_json, FROZEN_HASHES, sha256_file, utc_now
from gnss_doppler_lab.gcspo_verify import verify_clean_ready, verify_final

def _artifact_manifest_sha(artifact_dir, phase):
    path = Path(artifact_dir) / "artifact_manifest_sha256.json"
    if phase == "clean-ready" and not path.is_file():
        return None
    return sha256_file(path)



def _r1_scope_for_artifact(artifact_dir: Path):
    from contextlib import nullcontext

    preregistration = artifact_dir / "completion_preregistration.json"
    try:
        preregistration.lstat()
    except FileNotFoundError:
        return nullcontext(False)
    from gnss_doppler_lab.gcspo_r1_runner import verify_effective_preregistration
    from gnss_doppler_lab.gcspo_r1_support import r1_support_adapter_scope

    verify_effective_preregistration(artifact_dir)
    return r1_support_adapter_scope()


def _check_r1_adapter_manifests(artifact_dir: Path) -> None:
    from gnss_doppler_lab.gcspo_r1_runner import (
        _AUDIT_RECEIPT_SHA256, _AUDIT_RECEIPT_SIZE, _strict_object,
    )
    with _r1_scope_for_artifact(artifact_dir) as active:
        if not active:
            raise ValueError("R1 completion preregistration is absent")
        freeze = _strict_object(
            (artifact_dir / "frozen_science_hashes.json").read_bytes(),
            "R1 frozen science hashes",
        )
        audits = freeze.get("audited_artifacts")
        if (freeze.get("schema") !=
                "gnss-doppler-lab.gcspo-stage0-r1-frozen-completion.science-hashes.v1" or
                type(audits) is not dict or
                set(audits) != {"physical_controls_audit", "physical_controls_audit_receipt"}):
            raise ValueError("R1 frozen-science manifest shape mismatch")
        for name, expected_name in (
                ("physical_controls_audit", "physical_controls_audit.json"),
                ("physical_controls_audit_receipt", "physical_controls_audit_receipt.json")):
            identity = audits[name]
            audit_path = artifact_dir / expected_name
            try:
                info = audit_path.lstat()
            except FileNotFoundError as exc:
                raise ValueError("R1 physical-controls audit manifest binding mismatch") from exc
            if (type(identity) is not dict or not audit_path.is_file() or audit_path.is_symlink() or
                    sha256_file(audit_path) != identity.get("sha256") or
                    info.st_size != identity.get("size_bytes")):
                raise ValueError("R1 physical-controls audit manifest binding mismatch")
        receipt = audits["physical_controls_audit_receipt"]
        if (receipt["sha256"] != _AUDIT_RECEIPT_SHA256 or
                receipt["size_bytes"] != _AUDIT_RECEIPT_SIZE):
            raise ValueError("R1 physical-controls audit receipt identity mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=ROOT / "artifacts/gcspo_stage0_static_rerun")
    parser.add_argument("--mode", choices=("full",))
    parser.add_argument("--phase", choices=("clean-ready", "final"))
    parser.add_argument("--r1-adapter-check", action="store_true")
    parser.add_argument("--r1-preaccess", action="store_true")
    parser.add_argument("--r1-final", action="store_true")
    parser.add_argument("--r1-final-manifest-only", action="store_true")
    parser.add_argument("--no-remote", action="store_true")
    args = parser.parse_args()
    started = utc_now()
    if args.r1_adapter_check:
        if (args.mode or args.phase or args.r1_preaccess or args.r1_final or
                args.r1_final_manifest_only or args.no_remote):
            parser.error("--r1-adapter-check must be used alone")
        _check_r1_adapter_manifests(args.artifact_dir)
        print("R1_ADAPTER_CHECK_PASS")
        return 0
    if args.r1_preaccess:
        if args.mode or args.phase or args.r1_final or args.r1_final_manifest_only:
            parser.error("--r1-preaccess cannot be combined with another verification mode")
        from gnss_doppler_lab.gcspo_r1_runner import (
            _verify_adapter_bindings, install_and_verify_adapter,
            verify_execution_freeze,
        )
        with install_and_verify_adapter():
            _verify_adapter_bindings()
            checked = verify_execution_freeze(ROOT, check_remote=not args.no_remote)
        print(f"R1_PREACCESS_VERIFIER_PASS target={checked['target_commit']}")
        return 0
    if args.r1_final:
        if args.mode or args.phase or args.r1_final_manifest_only or args.no_remote:
            parser.error("--r1-final must be used alone")
        from gnss_doppler_lab.gcspo_r1_runner import verify_r1_final
        with _r1_scope_for_artifact(args.artifact_dir) as active:
            if not active:
                raise ValueError("R1 completion preregistration is absent")
            result = verify_r1_final(args.artifact_dir)
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                              text=True, capture_output=True).stdout.strip()
        implementation = json.loads((args.artifact_dir / "implementation_manifest.json").read_text())
        report = {
            "schema": "gnss-doppler-lab.gcspo-stage0.r1-verifier-report.v1",
            "target_commit": implementation["target_commit"], "evidence_commit": head,
            "artifact_manifest_sha256": sha256_file(args.artifact_dir / "artifact_manifest_sha256.json"),
            "checks": [{"id": "r1_final", "status": "PASS"},
                       {"id": "r1_exact_inventory", "status": "PASS"}],
            "overall_status": "PASS", "verified_run_status": "VALID_SCIENCE",
            "attestation_scope": "post-evidence; report excluded from the scientific artifact manifest",
            "command": "--r1-final", "started_utc": started, "finished_utc": utc_now(), "exit_code": 0,
        }
        canonical_write_json(args.artifact_dir / "verifier_report.json", report)
        print(f"R1_FINAL_VERIFIER_PASS verdict={result['verdict']}")
        return 0
    if args.r1_final_manifest_only:
        if args.mode or args.phase or args.no_remote:
            parser.error("--r1-final-manifest-only must be used alone")
        from gnss_doppler_lab.gcspo_r1_runner import verify_r1_final_manifest
        verify_r1_final_manifest(args.artifact_dir)
        print("R1_FINAL_MANIFEST_VERIFIER_PASS")
        return 0
    if args.no_remote:
        parser.error("--no-remote requires --r1-preaccess")
    if not args.mode and not args.phase:
        parser.error("--mode full or --phase is required")
    phase = "final" if args.mode == "full" else args.phase
    with _r1_scope_for_artifact(args.artifact_dir) as r1_active:
        if r1_active and phase == "final":
            from gnss_doppler_lab.gcspo_r1_runner import verify_r1_final
            result = verify_r1_final(args.artifact_dir)
            head = subprocess.run(["/usr/bin/git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                                  text=True, capture_output=True).stdout.strip()
            implementation = json.loads((args.artifact_dir / "implementation_manifest.json").read_text())
            report = {
                "schema": "gnss-doppler-lab.gcspo-stage0.r1-verifier-report.v1",
                "commit": implementation["target_commit"],
                "target_commit": implementation["target_commit"], "evidence_commit": head,
                "artifact_manifest_sha256": sha256_file(args.artifact_dir / "artifact_manifest_sha256.json"),
                "checks": [{"id": "r1_final", "status": "PASS"},
                           {"id": "r1_exact_inventory", "status": "PASS"}],
                "overall_status": "PASS", "verified_run_status": "VALID_SCIENCE",
                "attestation_scope": "post-evidence; report excluded from the scientific artifact manifest",
                "command": "--mode full", "started_utc": started, "finished_utc": utc_now(), "exit_code": 0,
            }
            canonical_write_json(args.artifact_dir / "verifier_report.json", report)
            print(f"R1_FINAL_VERIFIER_PASS verdict={result['verdict']}")
            return 0
        result = (verify_clean_ready(args.artifact_dir) if phase == "clean-ready"
                  else verify_final(args.artifact_dir, strict=True))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()
    freeze_path = args.artifact_dir / "implementation_manifest.json"
    target = head if not freeze_path.is_file() else __import__("json").loads(freeze_path.read_text()).get("target_commit", head)
    report = {"schema": "gnss-doppler-lab.gcspo-stage0.verifier-report.v2", "commit": target,
              "target_commit": target, "evidence_commit": head, "config_sha256": FROZEN_HASHES["config.json"],
              "artifact_manifest_sha256": _artifact_manifest_sha(args.artifact_dir, phase),
              "checks": [{"id": "semantic_reconstruction", "status": result["status"]}],
              "overall_status": "PASS", "verified_run_status": "VALID_SCIENCE" if phase == "final" else "CLEAN_ONLY_PASS",
              "command": "--mode full" if args.mode else f"--phase {phase}", "started_utc": started,
              "finished_utc": utc_now(), "exit_code": 0}
    canonical_write_json(args.artifact_dir / "verifier_report.json", report)
    print(f"VERIFIER_PASS phase={phase}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

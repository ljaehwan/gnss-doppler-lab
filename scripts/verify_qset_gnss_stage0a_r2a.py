#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from gnss_doppler_lab import qset_stage0a_r2a as Q

class VerificationError(RuntimeError): pass
def need(value: bool, message: str) -> None:
    if not value: raise VerificationError(message)
def validate_manifest(root: Path, manifest: dict) -> None:
    for row in manifest["files"]:
        path = root / row["path"]; need(path.is_file(), f"missing {path}"); need(path.stat().st_size == row["size_bytes"], f"size drift {path}"); need(Q.sha256_file(path) == row["sha256"], f"hash drift {path}")
    need(Q.canonical_sha(manifest["files"]) == manifest["aggregate_sha256"], "aggregate drift")
def main() -> int:
    need(Q.ARTIFACT.is_dir(), "artifact absent")
    required = ["README.md", "preregistration.json", "source_binding.json", "original_r2_evidence.json", "baseline_receiver_forensics.json", "acquisition_attempt_timeline.csv", "channel_occupancy_timeline.csv", "receiver_variant_matrix.json", "receiver_variant_results.json", "ss1_support_audit.json", "clean_regression.json", "access_audit.json", "deterministic_reproduction.json", "final_verdict.json", "artifact_manifest_sha256.json", "test_output.txt"]
    for name in required: need((Q.ARTIFACT / name).is_file(), f"missing artifact {name}")
    prereg = Q.read_json(Q.ARTIFACT / "preregistration.json"); need(prereg["score_operations_before_selection"] == 0, "preselection scoring")
    freeze = Q.read_json(Q.ARTIFACT / "freeze_commit.json")
    need(freeze["base_sha"] == Q.BASE_SHA and not freeze["variant_or_gate_changes_after_freeze"] and not freeze["scientific_changes"], "freeze binding drift")
    for row in freeze["executable_final_bindings"]:
        path = Q.ROOT / row["path"]; need(path.is_file(), f"missing executable {path}"); need(path.stat().st_size == row["size_bytes"] and Q.sha256_file(path) == row["sha256"], f"executable drift {path}")
    access = Q.read_json(Q.ARTIFACT / "access_audit.json"); need(access["forbidden_raw_access"] == {"stats": 0, "hashes": 0, "opens": 0, "mmaps": 0, "bytes_read": 0}, "forbidden raw access")
    need(access["ss1_spoofing_score_operations"] == 0, "SS-1 score forbidden")
    results = Q.read_json(Q.ARTIFACT / "receiver_variant_results.json"); variants = {row["variant"]: row for row in results["variants"]}; need(set(variants) == set(Q.VARIANTS), "variant set drift")
    need([name for name,row in variants.items() if row["support_gate_pass"]] == ["V3"], "support gate selection drift")
    need(variants["V3"]["m_ge_5_windows"] == 132 and variants["V3"]["qualified_prn_count"] == 6, "V3 support drift")
    for name in ("V1", "V2", "V3"):
        row=variants[name]; need(row["exit_code"] == 0 and row["terminal_drain"], f"{name} execution drift"); need(row["finite_failures"] == row["cadence_failures"] == row["causal_failures"] == 0, f"{name} trace validation drift")
    clean=Q.read_json(Q.ARTIFACT / "clean_regression.json"); need(clean["status"] == "FAIL" and not clean["refit"] and not clean["gate"]["pass"], "clean regression drift")
    need(clean["model_sha256"] == "a38d8bb79ca817c9241f4146e0b41759d7cfcf62c609f0b2e5ec5fbcc09f2bda", "model drift")
    need(clean["threshold_sha256"] == "2b223096c96af7a97fd9d5a5da6048c5f65f55b75fa05e410bcce9daf55e0490", "threshold drift")
    verdict = Q.read_json(Q.ARTIFACT / "final_verdict.json"); need(verdict["verdict"] == "RECEIVER_REPAIR_FAILED_CLEAN_REGRESSION" and not verdict["unopened_attack_confirmation_authorized"] and not verdict["ss1_spoofing_score_computed"], "invalid verdict")
    validate_manifest(Q.ARTIFACT, Q.read_json(Q.ARTIFACT / "artifact_manifest_sha256.json"))
    print(json.dumps({"status": "PASS", "verdict": verdict["verdict"], "score_operations": 0, "forbidden_raw_bytes": 0}, sort_keys=True)); return 0
if __name__ == "__main__":
    try: raise SystemExit(main())
    except VerificationError as exc: print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True)); raise SystemExit(1)

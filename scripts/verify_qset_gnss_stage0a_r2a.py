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
    access = Q.read_json(Q.ARTIFACT / "access_audit.json"); need(access["forbidden_raw_access"] == {"stats": 0, "hashes": 0, "opens": 0, "mmaps": 0, "bytes_read": 0}, "forbidden raw access")
    need(access["ss1_spoofing_score_operations"] == 0, "SS-1 score forbidden")
    results = Q.read_json(Q.ARTIFACT / "receiver_variant_results.json"); need(set(results) == set(Q.VARIANTS), "variant set drift")
    verdict = Q.read_json(Q.ARTIFACT / "final_verdict.json"); need(verdict["verdict"] in {"READY_FOR_QSET_R2B_UNOPENED_ATTACK_CONFIRMATION", "DATASET_SUPPORT_LIMIT_CONFIRMED", "RECEIVER_REPAIR_FAILED_CLEAN_REGRESSION", "INCONCLUSIVE_RECEIVER_SUPPORT_ROOT_CAUSE"}, "invalid verdict")
    validate_manifest(Q.ARTIFACT, Q.read_json(Q.ARTIFACT / "artifact_manifest_sha256.json"))
    print(json.dumps({"status": "PASS", "verdict": verdict["verdict"], "score_operations": 0, "forbidden_raw_bytes": 0}, sort_keys=True)); return 0
if __name__ == "__main__":
    try: raise SystemExit(main())
    except VerificationError as exc: print(json.dumps({"status": "FAIL", "error": str(exc)}, sort_keys=True)); raise SystemExit(1)
